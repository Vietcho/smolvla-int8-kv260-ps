"""Small dependency-free helpers for SmolVLA runtime profiling.

The block timers are intentionally installed only for the measured iterations.
Their totals are inclusive, so nested blocks overlap and must not be added.
"""

from __future__ import annotations

import functools
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


def tree_nbytes(value: Any, _seen: set[int] | None = None) -> int:
    """Return logical payload bytes for tensors/arrays inside a Python tree."""
    seen = set() if _seen is None else _seen
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, dict):
        marker = id(value)
        if marker in seen:
            return 0
        seen.add(marker)
        return sum(tree_nbytes(item, seen) for item in value.values())
    if isinstance(value, (tuple, list)):
        marker = id(value)
        if marker in seen:
            return 0
        seen.add(marker)
        return sum(tree_nbytes(item, seen) for item in value)
    return 0


def timed_call(function: Callable, *args, **kwargs):
    started = time.perf_counter_ns()
    result = function(*args, **kwargs)
    return result, (time.perf_counter_ns() - started) / 1_000_000


def file_record(path: Path, elapsed_ms: float) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "read_ms": elapsed_ms}


class BlockProfiler:
    """Instrument selected bound methods without changing packaged model source."""

    def __init__(self) -> None:
        self._rows: dict[str, dict] = defaultdict(
            lambda: {
                "calls": 0,
                "total_ms": 0.0,
                "samples_ms": [],
                "logical_input_bytes": 0,
                "logical_output_bytes": 0,
            }
        )
        self._restore: list[tuple[object, str, Callable]] = []

    def wrap_method(
        self,
        owner: object,
        method_name: str,
        label: str | Callable[[tuple, dict], str],
    ) -> None:
        original = getattr(owner, method_name)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            name = label(args, kwargs) if callable(label) else label
            input_bytes = tree_nbytes((args, kwargs))
            started = time.perf_counter_ns()
            result = original(*args, **kwargs)
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
            row = self._rows[name]
            row["calls"] += 1
            row["total_ms"] += elapsed_ms
            row["samples_ms"].append(elapsed_ms)
            row["logical_input_bytes"] += input_bytes
            row["logical_output_bytes"] += tree_nbytes(result)
            return result

        setattr(owner, method_name, wrapped)
        self._restore.append((owner, method_name, original))

    def install_smolvla(self, policy) -> None:
        flow = policy.model
        joint = flow.vlm_with_expert
        self.wrap_method(policy, "prepare_images", "01_prepare_images")
        self.wrap_method(policy, "prepare_state", "02_prepare_state")
        self.wrap_method(joint, "embed_image", "03_vision_encoder_connector")
        self.wrap_method(joint, "embed_language_tokens", "04_language_embedding")
        self.wrap_method(flow, "embed_prefix", "05_embed_prefix_total")

        def forward_label(_args: tuple, kwargs: dict) -> str:
            return (
                "06_vlm_prefix_kv_prefill"
                if kwargs.get("fill_kv_cache")
                else "08_vlm_action_expert_step"
            )

        self.wrap_method(joint, "forward", forward_label)
        self.wrap_method(flow, "embed_suffix", "07_embed_suffix_step")
        self.wrap_method(flow, "denoise_step", "09_denoise_step_total")
        self.wrap_method(flow, "sample_actions", "10_sample_actions_total")

    def remove(self) -> None:
        for owner, method_name, original in reversed(self._restore):
            setattr(owner, method_name, original)
        self._restore.clear()

    def report(self, iterations: int) -> dict:
        rows = []
        for name in sorted(self._rows):
            value = self._rows[name]
            samples = value["samples_ms"]
            rows.append(
                {
                    "name": name,
                    "calls": value["calls"],
                    "calls_per_inference": value["calls"] / iterations,
                    "total_ms": value["total_ms"],
                    "mean_ms_per_call": statistics.fmean(samples),
                    "mean_ms_per_inference": value["total_ms"] / iterations,
                    "min_ms_per_call": min(samples),
                    "max_ms_per_call": max(samples),
                    "logical_input_bytes_total": value["logical_input_bytes"],
                    "logical_output_bytes_total": value["logical_output_bytes"],
                    "logical_input_bytes_per_inference": value["logical_input_bytes"] / iterations,
                    "logical_output_bytes_per_inference": value["logical_output_bytes"] / iterations,
                }
            )
        return {
            "instrumented": True,
            "iterations": iterations,
            "warning": (
                "Inclusive Python timers overlap for nested blocks and add overhead; "
                "do not sum rows. Use a separate uninstrumented run for publishable latency."
            ),
            "data_note": (
                "Byte counts are logical tensor payloads at software boundaries. "
                "This PS-only runner performs no AXI DMA or PS-PL transfer."
            ),
            "blocks": rows,
        }


def print_timing_table(title: str, rows: list[dict]) -> None:
    print(f"\n=== {title} ===")
    if not rows:
        print("(disabled)")
        return
    print(f"{'block':36s} {'calls':>7s} {'ms/infer':>12s} {'input/infer':>14s} {'output/infer':>14s}")
    for row in rows:
        print(
            f"{row['name']:36s} {row['calls']:7d} "
            f"{row['mean_ms_per_inference']:12.3f} "
            f"{row['logical_input_bytes_per_inference']:14.0f} "
            f"{row['logical_output_bytes_per_inference']:14.0f}"
        )


def print_golden_table(comparisons: dict[str, dict]) -> None:
    print("\n=== GOLDEN OUTPUT COMPARISON ===")
    print(f"{'reference':30s} {'max_abs':>12s} {'mean_abs':>12s} {'rmse':>12s} {'cosine':>12s} {'finite':>8s}")
    for name, values in comparisons.items():
        cosine = values.get("cosine_similarity")
        cosine_text = "n/a" if cosine is None else f"{cosine:.9f}"
        print(
            f"{name:30s} {values['max_abs']:12.6g} {values['mean_abs']:12.6g} "
            f"{values['rmse']:12.6g} {cosine_text:>12s} {str(values['finite']):>8s}"
        )
