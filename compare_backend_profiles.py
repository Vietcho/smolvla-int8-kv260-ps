"""Combine QNNPACK/oneDNN SmolVLA profile reports into comparison CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV={path}")


def percentage(part: float, whole: float) -> float:
    return 100.0 * part / whole if whole else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple SmolVLA backend profile reports")
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output-prefix", type=Path, default=Path("results/backend_survey"))
    args = parser.parse_args()

    summary_rows: list[dict] = []
    phase_rows: list[dict] = []
    block_rows: list[dict] = []
    for report_path in args.reports:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        environment = report["environment"]
        engine = environment["quantized_engine"]
        phases = report["phase_timings"]
        vla_ms = phases["vla_ms"]["mean_ms"]
        e2e_ms = phases["e2e_ms"]["mean_ms"]
        accuracy_section = report.get("accuracy", {})
        accuracy = accuracy_section.get(
            "w2_vs_w1_pc_int8",
            accuracy_section.get("comparisons", {}).get("raw_vs_w1_pc_int8", {}),
        )
        memory = report.get("memory", {})
        summary_rows.append({
            "engine": engine,
            "status": report["status"],
            "hostname": environment.get("hostname"),
            "torch": environment.get("torch"),
            "threads": environment.get("threads"),
            "vla_mean_ms": vla_ms,
            "e2e_mean_ms": e2e_ms,
            "peak_rss_mib": memory.get("peak_rss_sampled_mib"),
            "w1_max_abs": accuracy.get("max_abs"),
            "w1_mean_abs": accuracy.get("mean_abs"),
            "w1_rmse": accuracy.get("rmse"),
            "w1_cosine": accuracy.get("cosine_similarity"),
            "report": str(report_path),
        })
        for phase, values in phases.items():
            phase_rows.append({
                "engine": engine,
                "phase": phase,
                "mean_ms": values["mean_ms"],
                "share_of_e2e_pct": percentage(values["mean_ms"], e2e_ms),
                "count": values["count"],
                "min_ms": values["min_ms"],
                "max_ms": values["max_ms"],
            })
        current_blocks = []
        for block in report.get("block_profile", {}).get("blocks", []):
            current_blocks.append({
                "engine": engine,
                "block": block["name"],
                "mean_ms_per_inference": block["mean_ms_per_inference"],
                "share_of_vla_pct": percentage(block["mean_ms_per_inference"], vla_ms),
                "calls_per_inference": block["calls_per_inference"],
                "mean_ms_per_call": block["mean_ms_per_call"],
                "logical_input_bytes_per_inference": block["logical_input_bytes_per_inference"],
                "logical_output_bytes_per_inference": block["logical_output_bytes_per_inference"],
            })
        current_blocks.sort(key=lambda row: row["mean_ms_per_inference"], reverse=True)
        for rank, row in enumerate(current_blocks, start=1):
            row["rank_within_engine"] = rank
        block_rows.extend(current_blocks)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(Path(f"{args.output_prefix}_summary.csv"), summary_rows)
    write_csv(Path(f"{args.output_prefix}_phases.csv"), phase_rows)
    write_csv(Path(f"{args.output_prefix}_blocks.csv"), block_rows)


if __name__ == "__main__":
    main()
