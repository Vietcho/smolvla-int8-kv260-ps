"""Rank PS bottlenecks and estimate accelerator potential with Amdahl's law."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def amdahl(share: float, kernel_speedup: float) -> float:
    return 1.0 / ((1.0 - share) + share / kernel_speedup)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "report",
        type=Path,
        nargs="?",
        default=HERE / "results" / "w2_ps_int8_layer_profile.json",
    )
    parser.add_argument("--assumed-pl-speedup", type=float, default=10.0)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    vla_mean = report["phase_timings"]["vla_ms"]["mean_ms"]
    iterations = report["phase_timings"]["vla_ms"]["count"]
    groups = []
    for name, values in report["linear_profile"]["groups"].items():
        mean_group_ms = values["total_ms"] / iterations
        share = min(1.0, mean_group_ms / vla_mean)
        groups.append(
            {
                "group": name,
                "mean_profiled_linear_ms": mean_group_ms,
                "share_of_vla": share,
                "macs_per_inference": values["macs"] / iterations,
                "ideal_max_system_speedup": amdahl(share, float("inf")),
                "estimated_system_speedup_at_assumed_pl_factor": amdahl(
                    share, args.assumed_pl_speedup
                ),
            }
        )
    groups.sort(key=lambda row: row["mean_profiled_linear_ms"], reverse=True)
    top_layers = []
    for row in report["linear_profile"]["top_layers"]:
        top_layers.append(
            {
                "name": row["name"],
                "group": row["group"],
                "mean_total_ms_per_inference": row["total_ms"] / iterations,
                "calls_per_inference": row["calls"] / iterations,
                "macs_per_inference": row["macs"] / iterations,
                "input_shape": row["input_shape"],
                "output_shape": row["output_shape"],
            }
        )
    result = {
        "warning": (
            "Hook timing is for ranking only. Confirm candidates with PS-only versus PS+PL A/B runs."
        ),
        "vla_mean_ms": vla_mean,
        "assumed_pl_kernel_speedup": args.assumed_pl_speedup,
        "group_ranking": groups,
        "top_layer_candidates": top_layers,
        "recommended_first_target": groups[0]["group"] if groups else None,
    }
    output = HERE / "results" / "w2_bottleneck_analysis.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"ANALYSIS={output}")


if __name__ == "__main__":
    main()
