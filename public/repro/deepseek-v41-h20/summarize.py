"""Print matched benchmark results without copying server metadata or raw logs."""

import argparse
import json
from pathlib import Path


STAGES = (
    "baseline",
    "deepgemm",
    "bcg",
    "decode-fp8",
    "k-reuse-bf16",
    "prefill-selection",
    "tp-moe",
    "score-budget",
    "query-dp",
)
PD_STAGES = (
    "tp-baseline",
    "cp-dp",
    "deepgemm",
    "bcg",
    "tp-moe",
    "prefill-budget",
    "prefill-selection",
    "k-reuse",
    "nvlink-layer-batch",
    "final-poll",
)
COLUMNS = "Mean TTFT (ms) | Mean TPOT (ms/token) | Mean E2E (ms) | P99 E2E (ms) | Output (tok/s) | Cache hit"


def read_result(directory, concurrency, prompts):
    path = directory / f"c{concurrency}.jsonl"
    if not path.exists():
        return None
    lines = path.read_text().splitlines()
    if len(lines) != 1:
        raise ValueError(f"Expected one benchmark result in {path}")
    result = json.loads(lines[0])
    if (
        result["max_concurrency"] != concurrency
        or result["completed"] != prompts
        or result["total_output_tokens"] != prompts * 500
        or result["mean_e2e_latency_ms"] <= 0
        or result["output_throughput"] <= 0
    ):
        raise ValueError(f"Result failed the workload or completion checks in {path}")
    return result


def metrics(result):
    cache = result.get("cache_report", {}).get("cache_hit_rate_pct")
    cache_text = f"{cache:.2f}%" if cache is not None else "—"
    return (
        f"{result['mean_ttft_ms']:.2f} | {result['mean_tpot_ms']:.2f} | "
        f"{result['mean_e2e_latency_ms']:.2f} | {result['p99_e2e_latency_ms']:.2f} | "
        f"{result['output_throughput']:.2f} | {cache_text}"
    )


def deltas(result, control):
    e2e = 1 - result["mean_e2e_latency_ms"] / control["mean_e2e_latency_ms"]
    throughput = result["output_throughput"] / control["output_throughput"] - 1
    return f"{e2e:+.2%} | {throughput:+.2%}"


def report(log_root, stages, concurrency, prompts):
    print(f"| Stage | Control | Decision | {COLUMNS} | E2E reduction | Output gain |")
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    completed = {}
    retained = None
    for stage in stages:
        directory = log_root / stage
        result = read_result(directory, concurrency, prompts)
        if result is None:
            print(f"| {stage} | — | Pending | — | — | — | — | — | — | — | — |")
            continue
        control_path = directory / "control"
        control = control_path.read_text().strip() if control_path.exists() else retained
        change = "— | —"
        decision = "baseline"
        if control:
            previous = completed[control]
            change = deltas(result, previous)
            decision = (
                "keep"
                if result["mean_e2e_latency_ms"] < previous["mean_e2e_latency_ms"]
                or result["output_throughput"] > previous["output_throughput"]
                else "drop"
            )
        print(f"| {stage} | {control or '—'} | {decision} | {metrics(result)} | {change} |")
        completed[stage] = result
        if decision != "drop":
            retained = stage
    if retained is not None and len(completed) == len(stages):
        e2e, throughput = deltas(completed[retained], completed[stages[0]]).split(" | ")
        print(f"\nRetained stage {retained} gives {e2e} E2E reduction and {throughput} output gain.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_root", type=Path)
    parser.add_argument("--pd-log-root", type=Path)
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--prompts", type=int, default=256)
    args = parser.parse_args()
    if not args.log_root.is_dir():
        parser.error("log_root must be an existing run directory")
    pd_log_root = args.pd_log_root or args.log_root
    if not pd_log_root.is_dir():
        parser.error("--pd-log-root must be an existing run directory")

    report(args.log_root, [f"{i}-{stage}" for i, stage in enumerate(STAGES)], args.concurrency, args.prompts)
    tp = read_result(args.log_root / "colocate-tp", args.concurrency, args.prompts)
    if tp is not None:
        print(f"\n| Colocated TP baseline | {COLUMNS} |")
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        print(f"| TP4 + DSpark | {metrics(tp)} |")
    print("\n## Disaggregated serving on 4P+4D")
    print()
    report(pd_log_root, [f"pd{i}-{stage}" for i, stage in enumerate(PD_STAGES)], args.concurrency, args.prompts)


if __name__ == "__main__":
    main()
