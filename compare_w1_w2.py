"""Compare an action chunk produced on KV260 PS against W1 and W0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def metrics(candidate: np.ndarray, reference: np.ndarray) -> dict:
    a, b = candidate.astype(np.float64), reference.astype(np.float64)
    delta = a - b
    denominator = np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel())
    return {
        "shape_equal": a.shape == b.shape,
        "finite": bool(np.isfinite(a).all()),
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "cosine_similarity": float(np.dot(a.ravel(), b.ravel()) / denominator),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    candidate = np.load(args.candidate, allow_pickle=False)
    package = HERE / "package"
    result = {
        "candidate": str(args.candidate.resolve()),
        "vs_w1_pc_int8": metrics(
            candidate,
            np.load(package / "reference" / "w1_int8_action_chunk_raw.npy", allow_pickle=False),
        ),
        "vs_w0_original": metrics(
            candidate,
            np.load(package / "inputs" / "w0a_action_chunk_raw.npy", allow_pickle=False),
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
