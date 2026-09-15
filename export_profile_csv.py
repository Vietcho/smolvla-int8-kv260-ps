"""Export phase, block, layer and data tables from a runner JSON report."""

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
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV={path}")


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
    phase_rows = [{"phase": name, **values} for name, values in report.get("phase_timings", {}).items()]
    write_rows(Path(f"{prefix}_phases.csv"), phase_rows)
    write_rows(Path(f"{prefix}_blocks.csv"), report.get("block_profile", {}).get("blocks", []))
    write_rows(Path(f"{prefix}_layers.csv"), report.get("linear_profile", {}).get("all_layers", []))
    movement = report.get("data_movement", {})
    boundary_rows = [
        {"boundary": name, "bytes": value}
        for name, value in movement.get("tensor_boundaries_last_inference", {}).items()
    ]
    write_rows(Path(f"{prefix}_data_boundaries.csv"), boundary_rows)


if __name__ == "__main__":
    main()
