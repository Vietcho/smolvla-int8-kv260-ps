# SmolVLA INT8 trên KV260 PS

Gói này nạp trực tiếp 303 lớp Linear dynamic INT8. Các phép Conv, Embedding,
Norm, Softmax, SiLU và flow integration vẫn chạy FP32 trên CPU Cortex-A53.
Runner hỗ trợ chọn rõ `qnnpack` hoặc `onednn`, đo từng khối SmolVLA và xuất CSV.

## 1. Kích hoạt môi trường sau mỗi lần SSH

Môi trường đã cài trên KV260 là Conda `smolvla_ps`. Mỗi lần đăng nhập chỉ cần:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate smolvla_ps
cd ~/Hoangviet/quan_int8_PS
```

Kiểm tra nhanh:

```bash
python -c "import torch; print(torch.__version__); print(torch.backends.quantized.supported_engines)"
```

Kết quả mong đợi là PyTorch `2.9.1+cpu`, có `qnnpack` và `onednn`.
PyTorch 2.10 ARM64 không dùng được trên Cortex-A53 vì gây `Illegal instruction`.

## 2. Cài dependency lần đầu

Không chạy phần này sau mỗi lần SSH. Miniforge và môi trường chỉ tạo một lần:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate smolvla_ps
cd ~/Hoangviet/quan_int8_PS
export TMPDIR=~/tmp
export PIP_CACHE_DIR=~/.cache/pip
bash install_ps_dependencies.sh
```

Script khóa `torch==2.9.1` và `torchvision==0.24.1`, dùng wheel CPU ARM64 đã
biên dịch sẵn và kiểm tra QNNPACK.

## 3. Kiểm tra gói

```bash
python verify_package.py
free -h
swapon --show
```

`verify_package.py` phải báo `PASS`. Nên có swap trước khi nạp model.

## 4. Khảo sát QNNPACK và oneDNN, xuất CSV

Một lệnh chạy lần lượt hai backend, bật timer theo khối và tự xuất CSV:

```bash
python -u survey_backends.py \
  --engines qnnpack onednn \
  --threads 4 \
  --warmup 0 \
  --iterations 1 \
  --prefix kv260_02_backend_survey \
  2>&1 | tee results/kv260_02_backend_survey_console.log
```

oneDNN từng mất khoảng 11 phút cho một inference trên KV260-02, nên lượt khảo
sát đầu dùng `warmup=0`, `iterations=1`. Đây là số khảo sát; benchmark công bố
cần warm-up và nhiều iteration hơn.

Các file chính:

```text
results/kv260_02_backend_survey_summary.csv
results/kv260_02_backend_survey_phases.csv
results/kv260_02_backend_survey_blocks.csv
```

Mở `_blocks.csv` bằng Excel và sắp xếp `engine`, `rank_within_engine`. Cột
`mean_ms_per_inference` cho thời gian khối; `share_of_vla_pct` cho tỷ lệ so với
toàn bộ suy luận VLA.

Timer khối là inclusive: khối cha chứa thời gian khối con, vì vậy không cộng tất
cả các dòng với nhau.

## 5. Chạy riêng một backend

QNNPACK:

```bash
python -u run_w2_ps_int8.py \
  --engine qnnpack --threads 4 --warmup 0 --iterations 1 \
  --skip-layer-profile --profile-blocks \
  --result-prefix kv260_02_qnnpack_blocks

python export_profile_csv.py results/kv260_02_qnnpack_blocks_profile.json
```

oneDNN:

```bash
python -u run_w2_ps_int8.py \
  --engine onednn --threads 4 --warmup 0 --iterations 1 \
  --skip-layer-profile --profile-blocks \
  --result-prefix kv260_02_onednn_blocks

python export_profile_csv.py results/kv260_02_onednn_blocks_profile.json
```

`--engine auto` ưu tiên QNNPACK trên ARM. Backend xuất hiện trong
`supported_engines` chỉ cho biết PyTorch đã đăng ký backend; tốc độ phải đo trên
đúng bo.

## 6. Profile từng Linear

Chỉ chạy khi cần xếp hạng 303 Linear vì hook làm tăng overhead:

```bash
python -u run_w2_ps_int8.py \
  --engine qnnpack --threads 4 --warmup 0 --iterations 1 \
  --profile-blocks --top-layers 30 \
  --result-prefix kv260_02_qnnpack_layers

python export_profile_csv.py results/kv260_02_qnnpack_layers_profile.json
python analyze_w2_profile.py results/kv260_02_qnnpack_layers_profile.json
```

File `_linear_groups.csv` tổng hợp Vision, VLM và Action Expert. File
`_layers.csv` chứa từng Linear. Số đo hook dùng để xếp hạng ứng viên tăng tốc;
latency chuẩn lấy từ lượt chạy có `--skip-layer-profile`.

## 7. Ý nghĩa kết quả

JSON và CSV chứa:

- `phase_timings`: build frame, preprocess, VLA, postprocess và end-to-end.
- `block_profile`: vision encoder, language embedding, VLM prefill, Action
  Expert và 10 bước denoise.
- `linear_profile`: nhóm và từng Linear nếu bật hook.
- `data_movement`: kích thước tensor logic trên PS; chưa phải lưu lượng AXI DMA.
- `accuracy`: so sánh output với W1 PC INT8 và W0 original.
- `memory`: peak RSS của tiến trình.

`REVIEW` nghĩa là output hữu hạn nhưng sai khác W1 vượt tolerance; không có nghĩa
chương trình bị lỗi.

## 8. Cập nhật code trên KV260

Sau khi PC đã push:

```bash
cd ~/Hoangviet/quan_int8_PS
git pull --ff-only
git lfs pull
python verify_package.py
```

Các file trong `results/` bị Git bỏ qua và không bị mất khi pull.
