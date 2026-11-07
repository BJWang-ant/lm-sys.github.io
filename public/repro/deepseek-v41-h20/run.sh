#!/usr/bin/env bash
set -euo pipefail

dry_run=0
if [[ $# -gt 0 && $1 == --dry-run ]]; then
    dry_run=1
    shift
fi
if (( $# == 0 )); then
    echo "Usage: bash run.sh [--dry-run] 0 1 2 3 4 5 6 7 8 colocate-tp pd0 pd1 pd2 pd3 pd4 pd5 pd6 pd7 pd8 pd9" >&2
    exit 2
fi

: "${MODEL_PATH:?Set MODEL_PATH to the local DeepSeek-V4.1-Flash checkpoint}"
: "${DATASET_PATH:?Set DATASET_PATH to ShareGPT_V3_unfiltered_cleaned_split.json}"
: "${LOG_ROOT:?Set LOG_ROOT to a new benchmark output directory}"
python=${PYTHON_BIN:-python3}
read -r -a concurrencies <<< "${CONCURRENCIES:-24}"
prompts=${NUM_PROMPTS:-256}
warmup=${WARMUP_REQUESTS:-10}
stages=(baseline deepgemm bcg decode-fp8 k-reuse-bf16 prefill-selection tp-moe score-budget query-dp)
pd_stages=(tp-baseline cp-dp deepgemm bcg tp-moe prefill-budget prefill-selection k-reuse nvlink-layer-batch final-poll)
mixed_retained=${MIXED_RETAINED:-}
mixed_control=${MIXED_CONTROL:-}
pd_retained=${PD_RETAINED:-}
pd_control=${PD_CONTROL:-}

for stage in "$@"; do
    [[ $stage == colocate-tp || $stage =~ ^([0-8]|pd[0-9])$ ]] || {
        echo "Invalid stage: $stage" >&2
        exit 2
    }
done
(( ${#concurrencies[@]} > 0 )) || exit 2
for value in "$prompts" "$warmup" "${concurrencies[@]}"; do
    [[ $value =~ ^[1-9][0-9]*$ ]] || { echo "Expected a positive integer: $value" >&2; exit 2; }
done

pids=()
workers_alive() {
    ps -eo pgid=,stat= | awk -v pgid="$1" '
        $1 == pgid && $2 !~ /^Z/ { alive = 1 }
        END { exit !alive }
    '
}

stop_workers() {
    [[ -n ${pids[0]-} ]] || return 0
    local pid
    for pid in "${pids[@]}"; do
        kill -TERM -- "-$pid" 2>/dev/null || true
    done
    local deadline=$((SECONDS + 180))
    for pid in "${pids[@]}"; do
        while workers_alive "$pid" && (( SECONDS < deadline )); do sleep 1; done
        if workers_alive "$pid"; then
            echo "Worker process group $pid is still active" >&2
            return 1
        fi
        wait "$pid" 2>/dev/null || true
    done
    pids=()
}
trap stop_workers EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

launch() {
    local role=$1
    shift
    printf '%s: ' "$role"
    printf '%q ' "$@"
    printf '\n'
    (( dry_run == 0 )) || return 0
    printf '%q ' "$@" > "$run_dir/$role.command"
    printf '\n' >> "$run_dir/$role.command"
    setsid "$@" > "$run_dir/$role.log" 2>&1 < /dev/null &
    pids+=("$!")
    printf '%s\n' "$!" > "$run_dir/$role.pid"
}

check_ports() {
    "$python" - "$@" <<'PY'
import socket
import sys

for port in map(int, sys.argv[1:]):
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", port))
PY
}

healthy() {
    "$python" - "$1" "$2" <<'PY'
import http.client
import sys

connection = http.client.HTTPConnection("127.0.0.1", int(sys.argv[1]), timeout=3)
try:
    connection.request("GET", sys.argv[2])
    raise SystemExit(connection.getresponse().status != 200)
except (OSError, http.client.HTTPException):
    raise SystemExit(1)
finally:
    connection.close()
PY
}

wait_ready() {
    local port=$1 path=$2 pid
    local deadline=$((SECONDS + 3600))
    until healthy "$port" "$path"; do
        for pid in "${pids[@]}"; do
            kill -0 "$pid" 2>/dev/null || { echo "Worker $pid exited; see $run_dir" >&2; return 1; }
        done
        (( SECONDS < deadline )) || { echo "Startup timed out on port $port; see $run_dir" >&2; return 1; }
        sleep 3
    done
}

enabled() {
    [[ " $active " == *" $1 "* ]]
}

switch() {
    if enabled "$1"; then echo 1; else echo 0; fi
}

compare() {
    "$python" - "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path

control, trial = [json.loads(Path(path).read_text()) for path in sys.argv[1:]]
for field in ("completed", "total_output_tokens", "total_input_tokens", "max_concurrency"):
    if control[field] != trial[field]:
        raise SystemExit(f"Unmatched benchmark inputs or completion counts: {field}")
e2e = trial["mean_e2e_latency_ms"] < control["mean_e2e_latency_ms"]
output = trial["output_throughput"] > control["output_throughput"]
print("keep" if e2e or output else "drop")
PY
}

if (( dry_run == 0 )); then
    [[ -f $MODEL_PATH/config.json && -f $DATASET_PATH ]]
    mkdir -p -- "$LOG_ROOT"
fi

for stage in "$@"; do
    mode=mixed
    if [[ $stage == pd* ]]; then
        mode=pd
        step=${stage#pd}
        run_dir="$LOG_ROOT/$stage-${pd_stages[$step]}"
        active="$pd_retained $step"
        control=$pd_control
    elif [[ $stage == colocate-tp ]]; then
        run_dir="$LOG_ROOT/$stage"
        active=
        control=
    else
        step=$stage
        run_dir="$LOG_ROOT/$stage-${stages[$stage]}"
        active="$mixed_retained $step"
        control=$mixed_control
    fi
    if (( dry_run == 0 )); then
        if [[ $mode == pd ]]; then check_ports 9090 9091 9092 8998; else check_ports 9090; fi
        mkdir -- "$run_dir"
        printf '%s\n' "$run_dir"
        printf '%s\n' "$active" > "$run_dir/active-stages"
        printf '%s\n' "$control" > "$run_dir/control"
        nvidia-smi --query-gpu=index,name,memory.total,memory.used,clocks.current.sm \
            --format=csv > "$run_dir/gpus.csv"
    fi

    scorer=auto
    torch_prefill=0
    ep=4
    cp_decode=tp
    graph=(--disable-prefill-cuda-graph)
    if [[ $mode == pd ]]; then
        deepgemm=2 bcg=3 tp_moe=4 fusion=6 k_reuse=7
        decode_fp8=0 large_budget=0 prefill_budget=$(switch 5)
        scorer=deepgemm
        torch_prefill=1
    else
        deepgemm=1 bcg=2 tp_moe=6 fusion=5 k_reuse=4
        decode_fp8=$(switch 3) large_budget=$(switch 7) prefill_budget=0
    fi
    if enabled "$deepgemm"; then
        scorer=deepgemm
        torch_prefill=0
    fi
    enabled "$tp_moe" && ep=1
    if enabled "$bcg"; then
        graph=(--cuda-graph-backend-prefill breakable --cuda-graph-max-bs-prefill 8192
            --cuda-graph-prefill-max-context 262144)
    fi
    shared_env=(env
        SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank
        'SGLANG_HUMMING_INPUT_QUANT_CONFIG={"dtype":"float8e4m3"}'
        SGLANG_USE_CUDA_IPC_TRANSPORT=1 SGLANG_MM_FEATURE_CACHE_MB=512 SGLANG_VLM_CACHE_SIZE_MB=512
        "SGLANG_DSV41_SM90_PREFILL_INDEXER=$scorer" "SGLANG_DSV41_TORCH_PREFILL_INDEXER=$torch_prefill"
        "SGLANG_OPT_DSV41_DECODE_INDEXER_FP8=$decode_fp8"
        "SGLANG_OPT_DSV41_VERIFY_INDEXER_K_REUSE=$(switch "$k_reuse")"
        "SGLANG_OPT_DSV41_VERIFY_INDEXER_BF16_K_REUSE=$(switch "$k_reuse")"
        "SGLANG_OPT_DSV41_TRITON_PREFILL_SELECTION=$(switch "$fusion")"
        "SGLANG_OPT_DSV41_LARGE_INDEXER_SCORE_BUDGET=$large_budget"
        "SGLANG_OPT_DSV41_LARGE_PREFILL_SCORE_BUDGET=$prefill_budget")
    common=("$python" -m sglang.launch_server
        --model-path "$MODEL_PATH" --served-model-name llm --trust-remote-code
        --host 127.0.0.1 --attention-backend dsv4 --moe-runner-backend humming
        --mm-feature-transport cuda_ipc --reasoning-parser auto --tool-call-parser auto
        --speculative-algorithm DSPARK --speculative-dspark-block-size 5
        --enable-cache-report --enable-metrics --enable-strict-thinking --random-seed 42)
    cp=(--enable-prefill-cp --cp-strategy interleave)
    hicache_ratio=2
    if [[ $mode == pd ]]; then
        hicache_ratio=0.25
    fi
    hicache=(--enable-hierarchical-cache --hicache-ratio "$hicache_ratio"
        --hicache-io-backend kernel --hicache-write-policy write_through)

    if [[ $mode == pd ]]; then
        prefill_parallel=(--tp 4)
        decode_parallel=(--tp 4)
        if enabled 1; then
            prefill_parallel+=("${cp[@]}")
            decode_parallel+=(--dp 4 --enable-dp-attention --enable-dp-lm-head)
        fi
        poll_ms=0
        enabled 9 && poll_ms=${PD_FINAL_POLL_MS:-5}
        mooncake=(SGLANG_HOST_IP=127.0.0.1 SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK
            MC_INTRA_NVLINK=true MC_INTRANODE_NVLINK=true SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER=1
            "SGLANG_OPT_MOONCAKE_NVLINK_LAYER_BATCH=$(switch 8)"
            "SGLANG_MOONCAKE_FINAL_POLL_TIMEOUT_MS=$poll_ms"
            SGLANG_SET_CPU_AFFINITY=1 SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1)
        workers=(--tokenizer-worker-num 8 --detokenizer-worker-num 8)
        launch prefill "${shared_env[@]}" "${mooncake[@]}" CUDA_VISIBLE_DEVICES=0,1,2,3 \
            "${common[@]}" "${prefill_parallel[@]}" "${graph[@]}" "${hicache[@]}" "${workers[@]}" \
            --language-model-only --ep-size "$ep" --moe-a2a-backend none \
            --chunked-prefill-size 8192 --max-prefill-tokens 8192 --enable-decoder-swa-bounded-replay \
            --swa-full-tokens-ratio 0.1 --mem-fraction-static 0.8 \
            --disaggregation-mode prefill --disaggregation-transfer-backend mooncake \
            --disaggregation-bootstrap-port 8998 --port 9091
        launch decode "${shared_env[@]}" "${mooncake[@]}" CUDA_VISIBLE_DEVICES=4,5,6,7 \
            "${common[@]}" "${workers[@]}" "${decode_parallel[@]}" \
            --ep-size "$ep" --moe-a2a-backend none --mem-fraction-static 0.85 --max-running-requests 256 \
            --disaggregation-mode decode --disaggregation-transfer-backend mooncake --port 9092
        if (( dry_run == 0 )); then
            wait_ready 9091 /health
            wait_ready 9092 /health
        fi
        launch router "$python" -m sglang_router.launch_router --pd-disaggregation \
            --prefill http://127.0.0.1:9091 8998 --decode http://127.0.0.1:9092 \
            --host 127.0.0.1 --port 9090 --worker-startup-timeout-secs 3600 \
            --disable-circuit-breaker --prefill-policy round_robin --decode-policy round_robin
    else
        colocate_parallel=(--tp 4)
        if [[ $stage != colocate-tp ]]; then
            enabled 8 && cp_decode=dp
            colocate_parallel+=("${cp[@]}" --enable-cp-decode-attn-tp --cp-decode-attn-mode "$cp_decode")
        fi
        launch colocate "${shared_env[@]}" CUDA_VISIBLE_DEVICES=0,1,2,3 \
            "${common[@]}" "${colocate_parallel[@]}" "${hicache[@]}" "${graph[@]}" \
            --ep-size "$ep" --moe-a2a-backend none --mem-fraction-static 0.8 \
            --chunked-prefill-size 8192 --max-prefill-tokens 8192 --max-running-requests 64 \
            --cuda-graph-max-bs-decode 64 --enable-decoder-swa-bounded-replay --port 9090
    fi

    benchmark_model=(--model "$MODEL_PATH")
    if [[ $mode == pd ]]; then
        benchmark_model+=(--served-model-name llm)
        (( dry_run == 1 )) || wait_ready 9090 /readiness
    else
        (( dry_run == 1 )) || wait_ready 9090 /health
    fi

    for concurrency in "${concurrencies[@]}"; do
        result="$run_dir/c$concurrency.jsonl"
        bench=("$python" -m sglang.bench_serving
            "${benchmark_model[@]}" --host 127.0.0.1 --port 9090 --backend sglang-oai
            --dataset-name generated-shared-prefix --dataset-path "$DATASET_PATH"
            --num-prompts "$prompts" --gsp-num-groups 1 --gsp-prompts-per-group "$prompts"
            --gsp-system-prompt-len 61440 --gsp-question-len 4096 --gsp-output-len 500
            --max-concurrency "$concurrency" --warmup-requests "$warmup" --seed 42
            --flush-cache --cache-report --output-file "$result")
        printf '%q ' "${bench[@]}"
        printf '\n'
        if (( dry_run == 0 )); then
            printf '%q ' "${bench[@]}" > "$run_dir/c$concurrency.command"
            printf '\n' >> "$run_dir/c$concurrency.command"
            NO_PROXY=127.0.0.1 no_proxy=127.0.0.1 "${bench[@]}" > "$run_dir/c$concurrency.log" 2>&1
            "$python" - "$result" "$prompts" <<'PY'
import json
import sys
from pathlib import Path

row = json.loads(Path(sys.argv[1]).read_text().splitlines()[-1])
expected = int(sys.argv[2])
if row["completed"] != expected or row["total_output_tokens"] != expected * 500:
    raise SystemExit("Incomplete benchmark. Check the benchmark and server logs.")
print(
    f"E2E {row['mean_e2e_latency_ms']:.1f} ms, "
    f"TTFT {row['mean_ttft_ms']:.1f} ms, TPOT {row['mean_tpot_ms']:.2f} ms, "
    f"output {row['output_throughput']:.1f} tok/s"
)
PY
        fi
    done
    if (( dry_run == 0 )) && [[ $stage != colocate-tp ]]; then
        decision=baseline
        if [[ -n $control ]]; then
            result_name="c${concurrencies[0]}.jsonl"
            decision=$(compare "$LOG_ROOT/$control/$result_name" "$run_dir/$result_name")
        fi
        printf '%s\n' "$decision" > "$run_dir/decision"
        printf '%s %s vs %s\n' "${run_dir##*/}" "$decision" "${control:-none}"
        if [[ $decision != drop ]]; then
            if [[ $mode == pd ]]; then
                pd_retained=$active
                pd_control=${run_dir##*/}
            else
                mixed_retained=$active
                mixed_control=${run_dir##*/}
            fi
        fi
    fi
    stop_workers
done
