# W2 — SmolVLA Linear INT8 trên KV260 PS

Hướng dẫn đưa toàn bộ gói lên GitHub bằng Git LFS và clone trên KV260 nằm tại
[`GITHUB_KV260.md`](GITHUB_KV260.md).

Thư mục này là gói triển khai PS-only và profiler. Nó nạp trực tiếp 303 Linear INT8, không nạp checkpoint gốc 906 MB rồi mới lượng tử. Norm, Softmax, SiLU, Conv, Embedding và flow integration vẫn FP32.

## Phạm vi chạy độc lập

Để **verify, nạp model, suy luận, benchmark và profile trên KV260 PS**, toàn bộ
file cần thiết đã nằm trong `quan_int8_PS`. Runner dùng đường dẫn tương đối theo
vị trí của chính `run_w2_ps_int8.py`, vì vậy không cần thư mục `quan_int8`,
`third_party`, checkpoint gốc hay các file khác ở thư mục cha.

Các chuỗi đường dẫn Windows còn xuất hiện trong một số JSON chỉ là thông tin
nguồn gốc của lần tạo artifact; runner không mở các đường dẫn đó.

Ngoại lệ: `prepare_ps_bundle.py` là công cụ **tạo lại** gói trên PC. Chỉ file này
cần `../quan_int8`, `../third_party/lerobot` và cache checkpoint. Không chạy
`prepare_ps_bundle.py` trên KV260. Các thư viện hệ thống/Python vẫn phải được cài
theo `requirements-ps.txt`.

## 1. Nội dung

```text
package/model/linear_int8.safetensors      303 Linear INT8
package/model/non_linear_fp32.safetensors  120 tensor FP32 còn lại
package/model/runtime_buffers.safetensors  rotary/runtime buffer
package/model/hf_smolvlm_config/           config và tokenizer offline
package/inputs/                            fixture, noise và W0 golden
package/reference/                         output W1 PC INT8
package/source/                            mã LeRobot và w0_common
run_w2_ps_int8.py                          runner + profiler
compare_w1_w2.py                           so sánh output
analyze_w2_profile.py                      xếp hạng bottleneck
check_ps_environment.sh                    kiểm tra KV260
verify_package.py                          kiểm tra checksum sau khi chép
```

Tổng package dữ liệu khoảng 581,16 MiB. Smoke test đóng gói trên PC đã nạp trực tiếp thành công và khớp W1 PC INT8 tuyệt đối (`max_abs=0`, `mean_abs=0`). Peak RSS trên PC khoảng 1770 MiB. KV260 hiện chỉ có khoảng 1,3 GiB available trong ảnh kiểm tra trước, vì vậy cần swap; chạy có thể rất chậm hoặc OOM tùy PyTorch ARM.

## 2. Chép sang KV260

Từ PowerShell trên PC, đang ở thư mục gốc dự án:

```powershell
scp -r ".\quan_int8_PS" ubuntu@192.168.50.21:/home/ubuntu/
```

Cũng có thể kéo cả thư mục `quan_int8_PS` vào `/home/ubuntu/` bằng cửa sổ SFTP của MobaXterm. Không chỉ chép riêng file 405 MB vì runner còn cần phần FP32, tokenizer, source và golden vector.

Trên KV260:

```bash
cd /home/ubuntu/quan_int8_PS
bash check_ps_environment.sh | tee ps_environment.txt
python3 verify_package.py
free -h
swapon --show
```

Không chạy full model nếu checksum FAIL. Nên có ít nhất 2 GiB swap cho smoke test; swap chỉ giúp tránh OOM, không làm inference nhanh.

## 3. Môi trường Python

LeRobot snapshot này yêu cầu Python 3.12. Kiểm tra `python3 --version` trước. Không cài đè Python hệ thống của image KV260. Tạo virtual environment bằng Python 3.12 hoặc Conda/Micromamba nếu máy đã có:

```bash
python3.12 -m venv .venv_ps
source .venv_ps/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-ps.txt
```

PyTorch ARM phải báo một backend INT8 khả dụng, ưu tiên `qnnpack`:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.backends.quantized.supported_engines)
PY
```

Nếu không có `qnnpack`, dừng lại và gửi `ps_environment.txt`; không tự coi kernel FP32 là INT8.

## 4. Smoke test đầu tiên

Lượt đầu không warm-up, một iteration và không gắn hook để giảm rủi ro RAM:

```bash
source .venv_ps/bin/activate
python run_w2_ps_int8.py \
  --threads 4 \
  --warmup 0 \
  --iterations 1 \
  --skip-layer-profile
```

Kết quả:

```text
results/w2_ps_int8_action_chunk_raw.npy
results/w2_ps_int8_benchmark.json
```

So sánh riêng nếu cần:

```bash
python compare_w1_w2.py results/w2_ps_int8_action_chunk_raw.npy
```

`w2_vs_w1_pc_int8` kiểm tra khác biệt runtime x86/ARM. `w2_vs_w0_original` là tổng sai khác do lượng tử và đổi nền tảng.

## 5. Benchmark thời gian tổng sạch

Không gắn hook lớp khi công bố latency end-to-end:

```bash
python run_w2_ps_int8.py \
  --threads 4 \
  --warmup 2 \
  --iterations 10 \
  --skip-layer-profile
```

Báo cáo tách riêng:

```text
meta architecture load
non-Linear load
INT8 Linear load
processor load
build_frame
preprocess
VLA inference
postprocess
end-to-end
peak RSS
```

## 6. Profile theo nhóm và từng Linear

Chạy ít iteration hơn vì 303 hook tạo overhead:

```bash
python run_w2_ps_int8.py \
  --threads 4 \
  --warmup 1 \
  --iterations 3 \
  --top-layers 30

python analyze_w2_profile.py
```

Profiler cộng số lần gọi, thời gian và MAC theo:

```text
Vision
VLM
Action Expert
```

Nó cũng liệt kê từng Linear có thời gian lớn nhất và ước lượng Amdahl. Thời gian hook chỉ dùng xếp hạng; quyết định accelerator cuối cùng phải đo A/B cùng runtime:

```text
W2: PS software GEMM
W3: PS → DMA → PL GEMM → DMA → PS
```

Chỉ phần backend GEMM được thay đổi giữa W2 và W3.

## 7. Trạng thái hiện tại

- Bundle và direct loader: PASS trên PC.
- Output direct loader so với W1 PC INT8: khớp tuyệt đối trên oneDNN x86.
- Kết quả smoke test PC nằm trong `validation_pc/`, không được coi là số đo W2.
- KV260 ARM: chưa chạy; cần kiểm tra Python/PyTorch/QNNPACK và RAM thực tế.
- Artifact vẫn là dynamic activation INT8. Static scale chỉ cần khóa trước khi thiết kế giao diện PL bit-exact.
