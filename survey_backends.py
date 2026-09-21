"""Run comparable block profiles for selected INT8 backends and export CSV."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULT_DIR = HERE / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Survey SmolVLA QNNPACK and oneDNN on KV260")
    parser.add_argument("--engines", nargs="+", default=["qnnpack", "onednn"])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--prefix", default="kv260_backend_survey")
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    reports: list[Path] = []
    for engine in args.engines:
        result_prefix = f"{args.prefix}_{engine}"
        print(f"\n===== RUNNING {engine.upper()} =====", flush=True)
        subprocess.run([
            sys.executable,
            str(HERE / "run_w2_ps_int8.py"),
            "--engine", engine,
            "--threads", str(args.threads),
            "--warmup", str(args.warmup),
            "--iterations", str(args.iterations),
            "--skip-layer-profile",
            "--profile-blocks",
            "--result-prefix", result_prefix,
        ], check=True)
        report = RESULT_DIR / f"{result_prefix}_profile.json"
        reports.append(report)
        subprocess.run([sys.executable, str(HERE / "export_profile_csv.py"), str(report)], check=True)

    output_prefix = RESULT_DIR / args.prefix
    subprocess.run([
        sys.executable,
        str(HERE / "compare_backend_profiles.py"),
        *map(str, reports),
        "--output-prefix", str(output_prefix),
    ], check=True)
    print(f"\nSurvey complete. Open {output_prefix}_blocks.csv to see the ranked bottlenecks.")


if __name__ == "__main__":
    main()
