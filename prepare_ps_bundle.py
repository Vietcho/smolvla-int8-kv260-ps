"""Build the self-contained W2 PS deployment bundle on the Windows PC."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
W1 = ROOT / "quan_int8"
PACKAGE = HERE / "package"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def find_hf_snapshot() -> Path:
    snapshots = (
        ROOT
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct"
        / "snapshots"
    )
    matches = sorted(path for path in snapshots.glob("*") if path.is_dir())
    if not matches:
        raise FileNotFoundError(f"SmolVLM metadata snapshot not found under {snapshots}")
    return matches[-1]


def main() -> None:
    model_dir = PACKAGE / "model"
    inputs_dir = PACKAGE / "inputs"
    reference_dir = PACKAGE / "reference"
    source_dir = PACKAGE / "source"
    for directory in (model_dir, inputs_dir, reference_dir, source_dir):
        directory.mkdir(parents=True, exist_ok=True)

    source_model = W1 / "source_model"
    processor_files = [
        "config.json",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    ]
    for name in processor_files:
        copy_file(source_model / name, model_dir / name)

    hf_source = find_hf_snapshot()
    hf_destination = model_dir / "hf_smolvlm_config"
    hf_destination.mkdir(parents=True, exist_ok=True)
    for source in hf_source.iterdir():
        if source.is_file() and source.name != "model.safetensors":
            copy_file(source, hf_destination / source.name)

    int8_source = W1 / "artifacts" / "smolvla_linear_w8a8_dynamic.safetensors"
    quant_manifest_source = W1 / "artifacts" / "quantization_manifest.json"
    int8_destination = model_dir / "linear_int8.safetensors"
    quant_manifest_destination = model_dir / "quantization_manifest.json"
    copy_file(int8_source, int8_destination)
    copy_file(quant_manifest_source, quant_manifest_destination)

    quant_manifest = json.loads(quant_manifest_source.read_text(encoding="utf-8"))
    excluded = set()
    for layer in quant_manifest["layers"]:
        excluded.add(f"{layer['name']}.weight")
        excluded.add(f"{layer['name']}.bias")

    checkpoint = source_model / "model.safetensors"
    non_linear_path = model_dir / "non_linear_fp32.safetensors"
    non_linear: dict[str, torch.Tensor] = {}
    with safe_open(checkpoint, framework="pt", device="cpu") as stream:
        all_keys = list(stream.keys())
        for key in all_keys:
            if key not in excluded:
                value = stream.get_tensor(key)
                non_linear[key] = (
                    value.to(torch.float32).contiguous() if value.is_floating_point() else value.contiguous()
                )
    save_file(
        non_linear,
        str(non_linear_path),
        metadata={
            "purpose": "FP32 SmolVLA parameters not represented by full Linear INT8 artifact"
        },
    )
    non_linear_tensor_count = len(non_linear)
    non_linear_tensor_bytes = sum(value.numel() * value.element_size() for value in non_linear.values())
    del non_linear

    # Save non-persistent rotary and related buffers. They are absent from the
    # checkpoint but must not remain uninitialized after meta-device creation.
    sys.path.insert(0, str(W1))
    import w1_int8_cpu as w1  # noqa: PLC0415

    policy, _, _ = w1.load_policy()
    runtime_buffers = {
        name: value.detach().cpu().contiguous() for name, value in policy.named_buffers()
    }
    runtime_buffers_path = model_dir / "runtime_buffers.safetensors"
    save_file(runtime_buffers, str(runtime_buffers_path), metadata={"source": "W1 loaded policy"})
    runtime_buffer_count = len(runtime_buffers)
    del runtime_buffers, policy

    for name in (
        "fixture.npz",
        "fixture_manifest.json",
        "w0a_noise.npy",
        "w0a_action_chunk_raw.npy",
        "w0a_action_chunk_postprocessed.npy",
    ):
        copy_file(W1 / "inputs" / name, inputs_dir / name)
    for name in (
        "w1_int8_action_chunk_raw.npy",
        "w1_int8_action_chunk_postprocessed.npy",
        "w1_int8_cpu_bench.json",
    ):
        copy_file(W1 / "results" / name, reference_dir / name)

    lerobot_source = ROOT / "third_party" / "lerobot" / "src" / "lerobot"
    lerobot_destination = source_dir / "lerobot"
    if lerobot_destination.exists():
        shutil.rmtree(lerobot_destination)
    shutil.copytree(
        lerobot_source,
        lerobot_destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    copy_file(W1 / "w0_common.py", source_dir / "w0_common.py")

    files = sorted(path for path in PACKAGE.rglob("*") if path.is_file())
    package_manifest = {
        "format_version": 1,
        "purpose": "W2 KV260 PS-only full SmolVLA Linear INT8 deployment and profiling",
        "quantization": {
            "linear_layers": len(quant_manifest["layers"]),
            "linear_weight_numel": quant_manifest["coverage"]["linear_weight_numel"],
            "linear_weight_fraction": quant_manifest["coverage"]["linear_weight_fraction"],
            "linear_weight": "per-output-channel symmetric INT8",
            "activation": "dynamic INT8 chosen by the selected PyTorch ARM backend",
            "accumulator": "INT32 inside quantized Linear kernel",
            "linear_output": "FP32",
            "other_operators": "FP32/BF16 as stored",
        },
        "memory_strategy": "meta init -> remove Linear -> materialize non-Linear -> stream packed INT8",
        "non_linear_tensor_count": non_linear_tensor_count,
        "non_linear_tensor_bytes": non_linear_tensor_bytes,
        "runtime_buffer_count": runtime_buffer_count,
        "files": [
            {
                "path": path.relative_to(PACKAGE).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        ],
    }
    manifest_path = PACKAGE / "package_manifest.json"
    manifest_path.write_text(json.dumps(package_manifest, indent=2), encoding="utf-8")
    print(json.dumps(package_manifest, indent=2))
    print(f"PS_PACKAGE={PACKAGE}")


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    main()
