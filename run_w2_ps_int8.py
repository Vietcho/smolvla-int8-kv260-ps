"""W2: load full SmolVLA Linear INT8 directly and profile it on KV260 PS.

The loader avoids materializing the original 906 MB checkpoint and an FP32
copy of every Linear. It builds the architecture on the meta device, removes
all Linear parameters, loads the remaining FP32 tensors, then streams packed
INT8 weights into the ARM quantized backend.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
PACKAGE = HERE / "package"
MODEL_DIR = PACKAGE / "model"
INPUT_DIR = PACKAGE / "inputs"
REFERENCE_DIR = PACKAGE / "reference"
RESULT_DIR = HERE / "results"
SOURCE_DIR = PACKAGE / "source"
HF_CONFIG_DIR = MODEL_DIR / "hf_smolvlm_config"

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(SOURCE_DIR))

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from torch import nn  # noqa: E402
from torch.ao.nn.quantized.dynamic import Linear as TorchDynamicLinear  # noqa: E402

import w0_common as common  # noqa: E402
from runtime_profile import (  # noqa: E402
    BlockProfiler, file_record, print_golden_table, print_timing_table, timed_call, tree_nbytes,
)


class LinearStub(nn.Module):
    def __init__(self, in_features: int, out_features: int, has_bias: bool):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = has_bias

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("LinearStub must be replaced before inference")


class Int8DynamicLinearCompat(nn.Module):
    def __init__(self, kernel: TorchDynamicLinear, in_features: int, out_features: int):
        super().__init__()
        self.kernel = kernel
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("_dtype_anchor", torch.empty(0, dtype=torch.float32), persistent=False)

    @property
    def weight(self) -> torch.Tensor:
        return self._dtype_anchor

    @property
    def bias(self) -> torch.Tensor | None:
        return self.kernel.bias()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.kernel(inputs.to(torch.float32))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="W2 KV260 PS-only SmolVLA INT8 profiler")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--top-layers", type=int, default=30)
    parser.add_argument("--skip-layer-profile", action="store_true")
    parser.add_argument("--profile-blocks", action="store_true", help="Time algorithm blocks; adds overhead.")
    parser.add_argument("--w1-max-abs-tolerance", type=float, default=0.05)
    parser.add_argument("--w1-mean-abs-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--result-prefix",
        type=str,
        default="w2_ps_int8",
        help="Prefix for output files, for example pc_int8 or w2_ps_int8.",
    )
    return parser.parse_args()


def choose_quantized_engine() -> str:
    supported = list(torch.backends.quantized.supported_engines)
    for candidate in ("qnnpack", "xnnpack", "onednn", "fbgemm"):
        if candidate in supported:
            torch.backends.quantized.engine = candidate
            return candidate
    raise RuntimeError(f"No usable dynamic INT8 backend; supported={supported}")


def replace_linears_with_stubs(module: nn.Module) -> dict[str, LinearStub]:
    stubs: dict[str, LinearStub] = {}

    def visit(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear):
                stub = LinearStub(child.in_features, child.out_features, child.bias is not None)
                setattr(parent, child_name, stub)
                stubs[name] = stub
            else:
                visit(child, name)

    visit(module)
    return stubs


def set_submodule(root: nn.Module, qualified_name: str, value: nn.Module) -> None:
    parent_name, child_name = qualified_name.rsplit(".", 1)
    parent = root.get_submodule(parent_name)
    setattr(parent, child_name, value)


def set_named_buffer(root: nn.Module, qualified_name: str, value: torch.Tensor) -> None:
    if "." in qualified_name:
        parent_name, buffer_name = qualified_name.rsplit(".", 1)
        parent = root.get_submodule(parent_name)
    else:
        parent, buffer_name = root, qualified_name
    parent._buffers[buffer_name] = value


def load_policy_direct() -> tuple[nn.Module, object, object, dict, list[dict]]:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    timings: dict[str, float] = {}
    mark = time.perf_counter()
    config = PreTrainedConfig.from_pretrained(MODEL_DIR)
    config.device = "cpu"
    config.compile_model = False
    config.load_vlm_weights = False
    config.vlm_model_name = str(HF_CONFIG_DIR.resolve())
    with torch.device("meta"):
        policy = SmolVLAPolicy(config)
    stubs = replace_linears_with_stubs(policy)
    timings["meta_architecture_seconds"] = time.perf_counter() - mark

    quant_manifest = json.loads((MODEL_DIR / "quantization_manifest.json").read_text(encoding="utf-8"))
    layer_records = quant_manifest["layers"]
    expected_names = {record["name"] for record in layer_records}
    if set(stubs) != expected_names:
        missing = sorted(expected_names - set(stubs))
        extra = sorted(set(stubs) - expected_names)
        raise RuntimeError(f"Architecture/manifest mismatch: missing={missing[:5]}, extra={extra[:5]}")

    mark = time.perf_counter()
    policy.to_empty(device="cpu")
    non_linear = load_file(str(MODEL_DIR / "non_linear_fp32.safetensors"), device="cpu")
    incompatible = policy.load_state_dict(non_linear, strict=False, assign=True)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected non-Linear keys: {incompatible.unexpected_keys[:10]}")
    del non_linear
    buffers = load_file(str(MODEL_DIR / "runtime_buffers.safetensors"), device="cpu")
    for name, value in buffers.items():
        set_named_buffer(policy, name, value)
    del buffers
    policy.model.global_image_start_token = torch.tensor(
        [policy.model.fake_image_token, policy.model.global_image_token], dtype=torch.long
    )
    policy.model.image_end_token = torch.tensor([policy.model.fake_image_token], dtype=torch.long)
    timings["non_linear_load_seconds"] = time.perf_counter() - mark

    mark = time.perf_counter()
    int8_path = MODEL_DIR / "linear_int8.safetensors"
    with safe_open(int8_path, framework="pt", device="cpu") as stream:
        artifact_keys = set(stream.keys())
        for index, record in enumerate(layer_records, start=1):
            name = record["name"]
            stub = stubs[name]
            integer_weight = stream.get_tensor(f"{name}.weight_int8").to(torch.int8)
            scales = stream.get_tensor(f"{name}.scale_fp32").to(torch.float64)
            zero_points = stream.get_tensor(f"{name}.zero_point_int32").to(torch.int64)
            qweight = torch._make_per_channel_quantized_tensor(
                integer_weight, scales, zero_points, int(record["axis"])
            )
            bias_key = f"{name}.bias_fp32"
            bias = stream.get_tensor(bias_key).to(torch.float32) if bias_key in artifact_keys else None
            kernel = TorchDynamicLinear(
                stub.in_features, stub.out_features, bias_=bias is not None, dtype=torch.qint8
            )
            kernel.set_weight_bias(qweight, bias)
            set_submodule(
                policy,
                name,
                Int8DynamicLinearCompat(kernel, stub.in_features, stub.out_features),
            )
            if index % 50 == 0:
                print(f"Loaded INT8 Linear {index}/{len(layer_records)}", flush=True)
            del integer_weight, scales, zero_points, qweight, bias, kernel
    timings["int8_linear_load_seconds"] = time.perf_counter() - mark

    meta_left = [name for name, value in policy.named_parameters() if value.is_meta]
    meta_left += [name for name, value in policy.named_buffers() if value.is_meta]
    if meta_left:
        raise RuntimeError(f"Meta tensors remain after direct load: {meta_left[:20]}")

    mark = time.perf_counter()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        str(MODEL_DIR),
        preprocessor_overrides={
            "device_processor": {"device": "cpu"},
            "tokenizer_processor": {"tokenizer_name": str(HF_CONFIG_DIR.resolve())},
        },
    )
    timings["processor_load_seconds"] = time.perf_counter() - mark
    policy.eval()
    return policy, preprocess, postprocess, timings, layer_records


def error_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict:
    candidate64 = candidate.astype(np.float64)
    reference64 = reference.astype(np.float64)
    delta = candidate64 - reference64
    flat_a, flat_b = candidate64.reshape(-1), reference64.reshape(-1)
    denominator = float(np.linalg.norm(flat_a) * np.linalg.norm(flat_b))
    reference_norm = float(np.linalg.norm(flat_b))
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
        "relative_l2": float(np.linalg.norm(delta) / reference_norm) if reference_norm else None,
        "cosine_similarity": float(np.dot(flat_a, flat_b) / denominator) if denominator else None,
        "finite": bool(np.isfinite(candidate64).all()),
    }


def summarize(values: list[float]) -> dict:
    ordered = sorted(values)
    percentile_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "p95_ms": ordered[percentile_index],
    }


def main() -> None:
    args = parse_arguments()
    if args.iterations < 1 or args.warmup < 0:
        raise ValueError("iterations must be >=1 and warmup must be >=0")
    engine = choose_quantized_engine()
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(args.threads)
    except RuntimeError:
        pass

    tracker = common.PeakRamTracker()
    tracker.start()
    total_mark = time.perf_counter()
    policy, preprocess, postprocess, load_timings, layer_records = load_policy_direct()
    load_timings["total_model_load_seconds"] = time.perf_counter() - total_mark

    storage_reads = []
    fixture_path = INPUT_DIR / "fixture.npz"
    fixture, elapsed_ms = timed_call(common.load_fixture, fixture_path)
    storage_reads.append(file_record(fixture_path, elapsed_ms))
    raw_observation = common.build_raw_observation(fixture)
    noise_path = INPUT_DIR / "w0a_noise.npy"
    noise_array, elapsed_ms = timed_call(np.load, noise_path, allow_pickle=False)
    storage_reads.append(file_record(noise_path, elapsed_ms))
    noise = torch.from_numpy(noise_array).float()

    def infer_once() -> tuple[torch.Tensor, np.ndarray, dict[str, float], dict[str, int]]:
        phases: dict[str, float] = {}
        with torch.inference_mode():
            mark = time.perf_counter()
            frame = common.build_frame(raw_observation)
            phases["build_frame_ms"] = (time.perf_counter() - mark) * 1000
            mark = time.perf_counter()
            batch = preprocess(frame)
            phases["preprocess_ms"] = (time.perf_counter() - mark) * 1000
            policy.reset()
            mark = time.perf_counter()
            chunk = policy.predict_action_chunk(batch, noise=noise)
            phases["vla_ms"] = (time.perf_counter() - mark) * 1000
            mark = time.perf_counter()
            post = postprocess(chunk.squeeze(0))
            post_array = post.detach().cpu().to(torch.float32).numpy()
            phases["postprocess_ms"] = (time.perf_counter() - mark) * 1000
            mark = time.perf_counter()
            cpu_chunk = chunk.detach().cpu().to(torch.float32)
            phases["output_materialize_ms"] = (time.perf_counter() - mark) * 1000
        phases["e2e_ms"] = sum(phases.values())
        boundaries = {
            "raw_observation_bytes": tree_nbytes(raw_observation),
            "frame_bytes": tree_nbytes(frame),
            "preprocessed_batch_bytes": tree_nbytes(batch),
            "noise_bytes": tree_nbytes(noise),
            "model_output_bytes": tree_nbytes(chunk),
            "raw_cpu_output_bytes": tree_nbytes(cpu_chunk),
            "postprocessed_output_bytes": tree_nbytes(post_array),
        }
        return cpu_chunk, post_array, phases, boundaries

    print(f"W2 engine={engine}; warmup={args.warmup}; iterations={args.iterations}", flush=True)
    for index in range(args.warmup):
        print(f"Warmup {index + 1}/{args.warmup}", flush=True)
        infer_once()

    block_profiler = BlockProfiler() if args.profile_blocks else None
    if block_profiler is not None:
        block_profiler.install_smolvla(policy)

    profile: dict[str, dict] = defaultdict(lambda: {"calls": 0, "total_ms": 0.0, "macs": 0})
    hooks = []
    layer_lookup = {record["name"]: record for record in layer_records}
    if not args.skip_layer_profile:
        for name, module in policy.named_modules():
            if isinstance(module, Int8DynamicLinearCompat):
                def pre_hook(current, inputs, layer_name=name):
                    current._profile_started_ns = time.perf_counter_ns()
                    shape = tuple(inputs[0].shape)
                    tokens = math.prod(shape[:-1]) if len(shape) > 1 else 1
                    profile[layer_name]["macs"] += tokens * current.in_features * current.out_features

                def post_hook(current, inputs, output, layer_name=name):
                    elapsed = (time.perf_counter_ns() - current._profile_started_ns) / 1_000_000
                    profile[layer_name]["calls"] += 1
                    profile[layer_name]["total_ms"] += elapsed
                    profile[layer_name]["input_shape"] = list(inputs[0].shape)
                    profile[layer_name]["output_shape"] = list(output.shape)

                hooks.append(module.register_forward_pre_hook(pre_hook))
                hooks.append(module.register_forward_hook(post_hook))

    phase_samples: dict[str, list[float]] = defaultdict(list)
    final_chunk = None
    final_post = None
    final_boundaries = {}
    for index in range(args.iterations):
        print(f"Measure {index + 1}/{args.iterations}", flush=True)
        final_chunk, final_post, phases, final_boundaries = infer_once()
        for name, value in phases.items():
            phase_samples[name].append(value)
    for hook in hooks:
        hook.remove()
    if block_profiler is not None:
        block_profiler.remove()

    layer_rows = []
    for name, values in profile.items():
        record = layer_lookup[name]
        group = record.get("group")
        if group is None:
            group = (
                "vision" if ".vlm.model.vision_model." in name
                else "vlm" if ".vlm." in name
                else "action_expert"
            )
        layer_rows.append({
            "name": name, "group": group, "calls": values["calls"],
            "calls_per_inference": values["calls"] / args.iterations,
            "total_ms": values["total_ms"],
            "mean_ms_per_call": values["total_ms"] / values["calls"],
            "mean_ms_per_inference": values["total_ms"] / args.iterations,
            "macs": values["macs"], "macs_per_inference": values["macs"] / args.iterations,
            "input_shape": values["input_shape"], "output_shape": values["output_shape"],
            "weight_numel": record["weight_numel"],
        })
    layer_rows.sort(key=lambda row: row["total_ms"], reverse=True)
    group_profile = {}
    for group in ("vision", "vlm", "action_expert"):
        selected = [row for row in layer_rows if row["group"] == group]
        group_profile[group] = {
            "linear_count": len(selected), "calls": sum(row["calls"] for row in selected),
            "total_ms": sum(row["total_ms"] for row in selected),
            "macs": sum(row["macs"] for row in selected),
            "mean_ms_per_inference": sum(row["total_ms"] for row in selected) / args.iterations,
            "macs_per_inference": sum(row["macs"] for row in selected) / args.iterations,
        }

    assert final_chunk is not None and final_post is not None
    candidate = final_chunk.numpy()
    reference_paths = {
        "raw_vs_w1_pc_int8": REFERENCE_DIR / "w1_int8_action_chunk_raw.npy",
        "raw_vs_w0_original": INPUT_DIR / "w0a_action_chunk_raw.npy",
        "postprocessed_vs_w1_pc_int8": REFERENCE_DIR / "w1_int8_action_chunk_postprocessed.npy",
        "postprocessed_vs_w0_original": INPUT_DIR / "w0a_action_chunk_postprocessed.npy",
    }
    references = {}
    for name, path in reference_paths.items():
        references[name], elapsed_ms = timed_call(np.load, path, allow_pickle=False)
        storage_reads.append(file_record(path, elapsed_ms))
    comparisons = {
        "raw_vs_w1_pc_int8": error_metrics(candidate, references["raw_vs_w1_pc_int8"]),
        "raw_vs_w0_original": error_metrics(candidate, references["raw_vs_w0_original"]),
        "postprocessed_vs_w1_pc_int8": error_metrics(final_post, references["postprocessed_vs_w1_pc_int8"]),
        "postprocessed_vs_w0_original": error_metrics(final_post, references["postprocessed_vs_w0_original"]),
    }
    versus_w1 = comparisons["raw_vs_w1_pc_int8"]
    numerical_pass = bool(
        versus_w1["finite"]
        and versus_w1["max_abs"] <= args.w1_max_abs_tolerance
        and versus_w1["mean_abs"] <= args.w1_mean_abs_tolerance
    )
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    safe_prefix = "".join(
        character if character.isalnum() or character in ("-", "_") else "_"
        for character in args.result_prefix
    ).strip("_")
    if not safe_prefix:
        raise ValueError("result-prefix must contain at least one letter or digit")
    output_path = RESULT_DIR / f"{safe_prefix}_action_chunk_raw.npy"
    post_output_path = RESULT_DIR / f"{safe_prefix}_action_chunk_postprocessed.npy"
    _, raw_save_ms = timed_call(np.save, output_path, candidate)
    _, post_save_ms = timed_call(np.save, post_output_path, final_post)
    memory = tracker.stop()
    block_report = (
        block_profiler.report(args.iterations)
        if block_profiler is not None else {"instrumented": False, "blocks": []}
    )
    report = {
        "status": "PASS" if numerical_pass else "REVIEW",
        "mode": "W2_KV260_PS_ONLY_FULL_LINEAR_DYNAMIC_INT8",
        "environment": {
            "hostname": platform.node(), "platform": platform.platform(), "machine": platform.machine(),
            "python": platform.python_version(), "torch": torch.__version__,
            "quantized_engine": engine, "threads": args.threads,
            "ram_total_mib": psutil.virtual_memory().total / 2**20,
        },
        "load_timings": load_timings,
        "phase_timings": {name: summarize(values) for name, values in phase_samples.items()},
        "memory": memory,
        "accuracy": {
            "comparisons": comparisons,
            "w2_vs_w1_pc_int8": comparisons["raw_vs_w1_pc_int8"],
            "w2_vs_w0_original": comparisons["raw_vs_w0_original"],
            "w1_max_abs_tolerance": args.w1_max_abs_tolerance,
            "w1_mean_abs_tolerance": args.w1_mean_abs_tolerance,
            "numerical_pass": numerical_pass,
        },
        "block_profile": block_report,
        "linear_profile": {
            "instrumented": not args.skip_layer_profile,
            "warning": "Python hook timings include profiler overhead; use ranking, not absolute layer sum.",
            "groups": group_profile, "top_layers": layer_rows[:args.top_layers], "all_layers": layer_rows,
        },
        "data_movement": {
            "scope": "PS storage and CPU-memory tensor boundaries; no PS-PL DMA in this runner",
            "storage_reads": storage_reads,
            "tensor_boundaries_last_inference": final_boundaries,
            "output_writes": [
                {"path": str(output_path), "bytes": output_path.stat().st_size, "write_ms": raw_save_ms},
                {"path": str(post_output_path), "bytes": post_output_path.stat().st_size, "write_ms": post_save_ms},
            ],
        },
        "outputs": {"action_chunk_raw": str(output_path), "action_chunk_postprocessed": str(post_output_path)},
    }
    if args.skip_layer_profile and not args.profile_blocks:
        report_kind = "benchmark"
    elif args.profile_blocks:
        report_kind = "profile"
    else:
        report_kind = "layer_profile"
    report_path = RESULT_DIR / f"{safe_prefix}_{report_kind}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print_golden_table(comparisons)
    print_timing_table("SMOLVLA BLOCK PROFILE", block_report["blocks"])
    print("\n=== PHASE TIMING (mean ms) ===")
    for name, values in report["phase_timings"].items():
        print(f"{name:28s} {values['mean_ms']:12.3f}")
    if not args.skip_layer_profile:
        print("\n=== INT8 LINEAR GROUP PROFILE (inclusive hook time) ===")
        for name, values in group_profile.items():
            print(f"{name:18s} {values['mean_ms_per_inference']:12.3f} ms/infer")
        print("\n=== TOP INT8 LINEAR LAYERS ===")
        for row in layer_rows[:args.top_layers]:
            print(f"{row['total_ms']:12.3f} ms  {row['calls']:5d} calls  {row['name']}")
    print(f"W2_REPORT={report_path}")


if __name__ == "__main__":
    main()
