"""Export one SmolVLA runner JSON report to analysis-friendly CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULT_DIR = HERE / "results"


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
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


def safe_ratio(numerator: float, denominator: float) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a SmolVLA JSON report to CSV tables")
    parser.add_argument("report", nargs="?", type=Path)
    args = parser.parse_args()
    if args.report is None:
        candidates = sorted(RESULT_DIR.glob("*_profile.json"), key=lambda path: path.stat().st_mtime)
        if not candidates:
            raise SystemExit("No *_profile.json found; pass a report path explicitly")
        report_path = candidates[-1]
    else:
        report_path = args.report

    report = json.loads(report_path.read_text(encoding="utf-8"))
    prefix = report_path.with_suffix("")
    environment = report.get("environment", {})
    engine = environment.get("quantized_engine")
    phase_timings = report.get("phase_timings", {})
    e2e_ms = phase_timings.get("e2e_ms", {}).get("mean_ms", 0.0)
    vla_ms = phase_timings.get("vla_ms", {}).get("mean_ms", 0.0)

    phase_rows = []
    for name, values in phase_timings.items():
        mean_ms = values.get("mean_ms", 0.0)
        phase_rows.append({
            "phase": name,
            "engine": engine,
            "share_of_e2e_pct": safe_ratio(mean_ms, e2e_ms),
            **values,
        })
    phase_rows.sort(key=lambda row: row.get("mean_ms", 0.0), reverse=True)
    for rank, row in enumerate(phase_rows, start=1):
        row["rank_by_mean_ms"] = rank
    write_rows(Path(f"{prefix}_phases.csv"), phase_rows)

    block_rows = []
    for row in report.get("block_profile", {}).get("blocks", []):
        enriched = dict(row)
        enriched["engine"] = engine
        enriched["share_of_vla_pct"] = safe_ratio(row.get("mean_ms_per_inference", 0.0), vla_ms)
        block_rows.append(enriched)
    block_rows.sort(key=lambda row: row.get("mean_ms_per_inference", 0.0), reverse=True)
    for rank, row in enumerate(block_rows, start=1):
        row["rank_by_ms_per_inference"] = rank
    write_rows(Path(f"{prefix}_blocks.csv"), block_rows)

    group_rows = []
    for name, values in report.get("linear_profile", {}).get("groups", {}).items():
        mean_ms = values.get("mean_ms_per_inference", 0.0)
        group_rows.append({
            "group": name,
            "engine": engine,
            "share_of_vla_pct": safe_ratio(mean_ms, vla_ms),
            **values,
        })
    group_rows.sort(key=lambda row: row.get("mean_ms_per_inference", 0.0), reverse=True)
    for rank, row in enumerate(group_rows, start=1):
        row["rank_by_ms_per_inference"] = rank
    write_rows(Path(f"{prefix}_linear_groups.csv"), group_rows)

    layer_rows = [
        dict(row, engine=engine)
        for row in report.get("linear_profile", {}).get("all_layers", [])
    ]
    write_rows(Path(f"{prefix}_layers.csv"), layer_rows)

    movement = report.get("data_movement", {})
    boundary_rows = [
        {"boundary": name, "engine": engine, "bytes": value}
        for name, value in movement.get("tensor_boundaries_last_inference", {}).items()
    ]
    write_rows(Path(f"{prefix}_data_boundaries.csv"), boundary_rows)

    load_rows = [
        {"load_stage": name, "engine": engine, "seconds": value}
        for name, value in report.get("load_timings", {}).items()
    ]
    write_rows(Path(f"{prefix}_load.csv"), load_rows)

    accuracy_rows = [
        {"reference": reference, "engine": engine, **values}
        for reference, values in report.get("accuracy", {}).get("comparisons", {}).items()
    ]
    write_rows(Path(f"{prefix}_accuracy.csv"), accuracy_rows)

    memory = report.get("memory", {})
    accuracy_section = report.get("accuracy", {})
    comparison = accuracy_section.get(
        "w2_vs_w1_pc_int8",
        accuracy_section.get("comparisons", {}).get("raw_vs_w1_pc_int8", {}),
    )
    write_rows(Path(f"{prefix}_summary.csv"), [{
        "status": report.get("status"),
        "hostname": environment.get("hostname"),
        "machine": environment.get("machine"),
        "python": environment.get("python"),
        "torch": environment.get("torch"),
        "engine": engine,
        "threads": environment.get("threads"),
        "vla_mean_ms": vla_ms,
        "e2e_mean_ms": e2e_ms,
        "peak_rss_mib": memory.get("peak_rss_sampled_mib"),
        "w1_max_abs": comparison.get("max_abs"),
        "w1_mean_abs": comparison.get("mean_abs"),
        "w1_cosine": comparison.get("cosine_similarity"),
    }])


if __name__ == "__main__":
    main()
