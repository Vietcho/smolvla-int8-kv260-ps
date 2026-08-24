"""Hạ tầng dùng chung cho baseline W0-a (đo inference thuần trên CPU x86).

Nguyên tắc bắt buộc của module này:

* Chỉ ĐỌC `models/smolvla_base/` và `third_party/lerobot/`. Không ghi, không sửa,
  không tạo file mới trong hai cây thư mục đó.
* Latency từng module lấy bằng cách bọc bound method trên *instance* đã nạp và
  đăng ký forward hook có thể gỡ bỏ, nên không đụng tới file nguồn upstream.
* Mọi nguồn ngẫu nhiên phải được cố định. Noise flow-matching được sinh từ seed và
  truyền tường minh vào `predict_action_chunk`, nên `VLAFlowMatching.sample_noise`
  (dùng global RNG, không nhận generator) không bao giờ được gọi trong đường đo.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "smolvla_base"
LEROBOT_DIR = ROOT / "third_party" / "lerobot"
CACHE_DIR = ROOT / ".cache" / "huggingface"
FIXTURE_DIR = ROOT / "fixtures" / "w0"
RESULT_DIR = ROOT / "results" / "local_W0_W3"
GOLDEN_DIR = RESULT_DIR / "golden"

# Phải đặt trước khi import torch/lerobot.
os.environ.setdefault("HF_HOME", str(CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(CACHE_DIR / "hub"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402

# Schema observation/action bị khóa cho toàn bộ W0/W1/W2/W3. W0-b (closed loop
# trong Webots) phải dùng lại đúng các hằng số này, nếu không kết quả không so được.
SCHEMA_VERSION = "w0-fixture-1"
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
CAMERA_NAMES = ["camera1", "camera2", "camera3"]
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
TASK = "Move the robot arm toward the red cube."
ROBOT_TYPE = "ur5e_webots"

# Các file checkpoint được kiểm tra toàn vẹn trước/sau mỗi lần chạy.
CHECKPOINT_FILES = [
    "model.safetensors",
    "config.json",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "policy_preprocessor_step_5_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
]


# --------------------------------------------------------------------------
# Hash và toàn vẹn file
# --------------------------------------------------------------------------
def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def content_hash(arrays: dict[str, np.ndarray]) -> str:
    """Hash nội dung mảng, độc lập với container và timestamp của file.

    Đây là định danh chính thức của fixture/golden: file `.npz` chứa timestamp
    trong header zip nên sha256 của file thay đổi giữa các lần ghi, còn hash nội
    dung thì không. Mọi mảng được ép về little-endian nên x86 và aarch64 (KV260)
    cho cùng một giá trị.
    """
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        if array.dtype.byteorder == ">":
            array = array.astype(array.dtype.newbyteorder("<"))
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype.str).encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def snapshot_tree(root: Path) -> dict[str, list]:
    """Chụp trạng thái (kích thước, mtime) mọi file dưới `root`."""
    snapshot = {}
    if not root.exists():
        return snapshot
    for path in sorted(root.rglob("*")):
        if path.is_file():
            info = path.stat()
            snapshot[str(path.relative_to(root)).replace("\\", "/")] = [
                info.st_size,
                info.st_mtime_ns,
            ]
    return snapshot


def diff_tree(before: dict, after: dict) -> dict:
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(name for name in set(before) & set(after) if before[name] != after[name])
    return {
        "unchanged": not (added or removed or changed),
        "added": added,
        "removed": removed,
        "changed": changed,
    }


def checkpoint_fingerprint(hash_files: bool = True) -> dict:
    """Vân tay checkpoint gốc. Dùng để khóa revision theo §15 và để chứng minh
    checkpoint không bị thay đổi sau khi chạy."""
    fingerprint = {}
    for name in CHECKPOINT_FILES:
        path = MODEL_DIR / name
        if not path.is_file():
            fingerprint[name] = {"present": False}
            continue
        entry = {"present": True, "size_bytes": path.stat().st_size}
        if hash_files:
            entry["sha256"] = sha256_file(path)
        fingerprint[name] = entry
    return fingerprint


# --------------------------------------------------------------------------
# Metadata môi trường (khóa phiên bản theo §15)
# --------------------------------------------------------------------------
def git_commit(repo: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def git_is_dirty(repo: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def package_versions() -> dict:
    versions = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}
    for name in ("transformers", "lerobot", "safetensors", "huggingface_hub", "accelerate", "torchvision"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception as error:  # noqa: BLE001 - chỉ ghi nhận, không được làm hỏng report
            versions[name] = f"unavailable: {type(error).__name__}"
    versions["lerobot_commit"] = git_commit(LEROBOT_DIR)
    versions["lerobot_worktree_dirty"] = git_is_dirty(LEROBOT_DIR)
    return versions


def cpu_temperatures() -> dict:
    """Nhiệt độ nếu đo được. Trên Windows psutil thường không có sensor, khi đó
    trả về lý do thay vì im lặng bỏ qua (§12 yêu cầu báo cáo nhiệt độ/throttling)."""
    reader = getattr(psutil, "sensors_temperatures", None)
    if reader is None:
        return {"available": False, "reason": "psutil.sensors_temperatures not implemented on this platform"}
    try:
        readings = reader()
    except Exception as error:  # noqa: BLE001
        return {"available": False, "reason": f"{type(error).__name__}: {error}"}
    if not readings:
        return {"available": False, "reason": "no sensor exposed by the OS"}
    return {
        "available": True,
        "celsius": {
            chip: [entry.current for entry in entries] for chip, entries in readings.items()
        },
    }


def host_info() -> dict:
    memory = psutil.virtual_memory()
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_total_mib": memory.total / (1024 * 1024),
        "ram_available_mib": memory.available / (1024 * 1024),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "env_omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "env_mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
    }


# --------------------------------------------------------------------------
# Peak RAM
# --------------------------------------------------------------------------
class PeakRamTracker:
    """Theo dõi peak RAM. `peak_wset` (Windows) và `ru_maxrss` (Linux) cho peak do
    OS ghi nhận; luồng lấy mẫu cho giá trị chạy được trên mọi nền tảng, dùng lại
    được nguyên vẹn trên KV260."""

    def __init__(self, interval_seconds: float = 0.05):
        self.interval = interval_seconds
        self.process = psutil.Process()
        self.peak_sampled_mib = 0.0
        self.sample_count = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                rss = self.process.memory_info().rss / (1024 * 1024)
            except psutil.Error:
                break
            self.peak_sampled_mib = max(self.peak_sampled_mib, rss)
            self.sample_count += 1
            self._stop.wait(self.interval)

    def start(self) -> None:
        self.peak_sampled_mib = current_rss_mib()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        report = {
            "peak_rss_sampled_mib": self.peak_sampled_mib,
            "sample_interval_seconds": self.interval,
            "sample_count": self.sample_count,
            "rss_final_mib": current_rss_mib(),
        }
        try:
            memory_info = self.process.memory_info()
        except psutil.Error:
            memory_info = None
        if memory_info is not None and hasattr(memory_info, "peak_wset"):
            report["peak_wset_mib"] = memory_info.peak_wset / (1024 * 1024)
        try:
            import resource  # noqa: PLC0415 - chỉ có trên POSIX

            report["ru_maxrss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except ImportError:
            pass
        return report


def current_rss_mib() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


# --------------------------------------------------------------------------
# Đo latency từng phase / từng module
# --------------------------------------------------------------------------
class PhaseTimer:
    """Đo thời gian các phase bên trong model mà không sửa file nguồn.

    Ghi chú quan trọng: các nhãn LỒNG NHAU (xem `NESTING`). Không được cộng dồn
    tất cả nhãn lại rồi coi là tổng, sẽ bị đếm trùng.
    """

    NESTING = {
        "embed_prefix": ["embed_image", "state_proj"],
        "denoise_step": ["embed_suffix", "vlm_decode", "action_out_proj"],
        "embed_suffix": ["action_in_proj", "action_time_mlp_in", "action_time_mlp_out"],
    }

    def __init__(self):
        self.enabled = False
        self.iterations: list[dict[str, list[float]]] = []
        self._current: dict[str, list[float]] = defaultdict(list)
        self._undo: list[tuple] = []
        self._handles: list = []

    def begin_iteration(self) -> None:
        self._current = defaultdict(list)
        self.iterations.append(self._current)

    def _record(self, label: str, seconds: float) -> None:
        if self.enabled:
            self._current[label].append(seconds * 1000.0)

    def wrap_method(self, owner: object, name: str, label: str | None = None, classifier=None) -> None:
        original = getattr(owner, name)
        had_own_attribute = name in vars(owner)
        default_label = label or name

        def wrapper(*args, **kwargs):
            key = default_label if classifier is None else classifier(kwargs, default_label)
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self._record(key, time.perf_counter() - started)

        setattr(owner, name, wrapper)
        self._undo.append((owner, name, original, had_own_attribute))

    def hook_module(self, module, label: str) -> None:
        pending: list[float] = []

        def pre_hook(_module, _inputs):
            pending.append(time.perf_counter())

        def post_hook(_module, _inputs, _output):
            if pending:
                self._record(label, time.perf_counter() - pending.pop())

        self._handles.append(module.register_forward_pre_hook(pre_hook))
        self._handles.append(module.register_forward_hook(post_hook))

    def restore(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for owner, name, original, had_own_attribute in reversed(self._undo):
            if had_own_attribute:
                setattr(owner, name, original)
            else:
                try:
                    delattr(owner, name)
                except AttributeError:
                    pass
        self._undo.clear()

    def summary(self) -> dict:
        labels = sorted({label for iteration in self.iterations for label in iteration})
        summary = {}
        for label in labels:
            per_iteration_totals = []
            per_call = []
            call_counts = []
            for iteration in self.iterations:
                samples = iteration.get(label, [])
                per_iteration_totals.append(float(sum(samples)))
                per_call.extend(samples)
                call_counts.append(len(samples))
            summary[label] = {
                "calls_per_iteration": sorted(set(call_counts)),
                "total_per_iteration_ms": stats(per_iteration_totals),
                "per_call_ms": stats(per_call),
            }
        return summary


def stats(values) -> dict | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "p95": float(np.percentile(array, 95)),
        "stdev": float(array.std(ddof=1)) if array.size > 1 else 0.0,
    }


def instrument_policy(policy, timer: PhaseTimer) -> None:
    """Gắn đồng hồ vào các phase của SmolVLA trên instance đã nạp.

    Ranh giới được chọn khớp với các khối §10 dự kiến đẩy xuống PL:
    vision encode, prefill KV cache, và vòng lặp denoise của Action Expert.
    """
    flow = policy.model
    backbone = flow.vlm_with_expert

    def vlm_classifier(kwargs: dict, _default: str) -> str:
        return "vlm_prefill" if kwargs.get("fill_kv_cache") else "vlm_decode"

    timer.wrap_method(flow, "embed_prefix")
    timer.wrap_method(flow, "embed_suffix")
    timer.wrap_method(flow, "denoise_step")
    timer.wrap_method(backbone, "forward", label="vlm_forward", classifier=vlm_classifier)
    if hasattr(backbone, "embed_image"):
        timer.wrap_method(backbone, "embed_image")

    for attribute, label in (
        ("state_proj", "state_proj"),
        ("action_in_proj", "action_in_proj"),
        ("action_out_proj", "action_out_proj"),
        ("action_time_mlp_in", "action_time_mlp_in"),
        ("action_time_mlp_out", "action_time_mlp_out"),
    ):
        module = getattr(flow, attribute, None)
        if module is not None:
            timer.hook_module(module, label)


# --------------------------------------------------------------------------
# Fixture, model, noise
# --------------------------------------------------------------------------
def fixture_arrays(fixture: dict) -> dict[str, np.ndarray]:
    """Chỉ lấy phần mảng số của fixture để tính content hash."""
    keys = ["observation.state", *[f"observation.images.{name}" for name in CAMERA_NAMES]]
    return {key: fixture[key] for key in keys}


def load_fixture(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        fixture = {key: data[key] for key in data.files}
    state = fixture["observation.state"]
    if state.shape != (len(JOINT_NAMES),):
        raise ValueError(f"Fixture state shape {state.shape} != {(len(JOINT_NAMES),)}")
    for name in CAMERA_NAMES:
        image = fixture[f"observation.images.{name}"]
        expected = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
        if image.shape != expected:
            raise ValueError(f"Fixture {name} shape {image.shape} != {expected}")
        if image.dtype != np.uint8:
            raise ValueError(f"Fixture {name} dtype {image.dtype} != uint8")
    return fixture


def build_raw_observation(fixture: dict) -> dict:
    """Dựng observation thô đúng schema mà controller Webots tạo ra."""
    state = fixture["observation.state"].astype(np.float32)
    observation = {name: np.float32(state[index]) for index, name in enumerate(JOINT_NAMES)}
    observation.update(
        {name: fixture[f"observation.images.{name}"].copy() for name in CAMERA_NAMES}
    )
    return observation


def build_frame(raw_observation: dict):
    from lerobot.datasets.utils import hw_to_dataset_features  # noqa: PLC0415
    from lerobot.policies.utils import build_inference_frame  # noqa: PLC0415

    hardware_features = {name: float for name in JOINT_NAMES}
    hardware_features.update({name: (IMAGE_HEIGHT, IMAGE_WIDTH, 3) for name in CAMERA_NAMES})
    dataset_features = hw_to_dataset_features(hardware_features, "observation", use_video=False)
    return build_inference_frame(
        observation=raw_observation,
        device=torch.device("cpu"),
        ds_features=dataset_features,
        task=TASK,
        robot_type=ROBOT_TYPE,
    )


def load_policy():
    """Nạp checkpoint gốc trên CPU. Chỉ đọc MODEL_DIR."""
    from lerobot.configs.policies import PreTrainedConfig  # noqa: PLC0415
    from lerobot.policies.factory import make_pre_post_processors  # noqa: PLC0415
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: PLC0415

    if not (MODEL_DIR / "model.safetensors").is_file():
        raise FileNotFoundError(f"Missing checkpoint: {MODEL_DIR}")

    started = time.perf_counter()
    config = PreTrainedConfig.from_pretrained(MODEL_DIR)
    config.device = "cpu"
    policy = SmolVLAPolicy.from_pretrained(MODEL_DIR, config=config)
    policy.eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        str(MODEL_DIR),
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    load_seconds = time.perf_counter() - started
    return policy, preprocess, postprocess, load_seconds


def make_noise(seed: int, chunk_size: int, max_action_dim: int, batch_size: int = 1) -> torch.Tensor:
    """Sinh noise flow-matching tất định.

    Phải truyền tensor này vào `predict_action_chunk(batch, noise=...)`. Nếu để
    None, upstream sẽ gọi `sample_noise()` dùng global RNG và golden output mất
    tính tái lập — đúng lỗi đã làm hỏng lần đo W0 trước.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(
        (batch_size, chunk_size, max_action_dim),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
