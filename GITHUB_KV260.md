# Cập nhật code qua GitHub

Repository đã có:

```text
https://github.com/Vietcho/smolvla-int8-kv260-ps.git
```

Hai file trọng số lớn dùng Git LFS. Không thêm file trong `results/` vào commit;
đó là kết quả riêng của từng bo.

## Push từ PC Windows

Mở PowerShell:

```powershell
Set-Location "D:\TAI_LIEU\DU_AN_XU_LY_TINHIEU\XU_LY_TIN_HIEU_FPGA\IP_FPGA\VLA\quan_int8_PS"

git status
git diff
git add run_w2_ps_int8.py survey_backends.py compare_backend_profiles.py export_profile_csv.py install_ps_dependencies.sh requirements-ps.txt README.md GITHUB_KV260.md
git diff --cached
git commit -m "Add KV260 backend survey and CSV profiling"
git push origin main
```

Hai file `ps_environment.txt` và `ps_environment_venv.txt` đang là file cục bộ
không theo dõi; lệnh `git add` ở trên không đưa chúng lên GitHub.

## Pull trên KV260

Mỗi bo chạy:

```bash
cd ~/Hoangviet/quan_int8_PS
git status --short
git pull --ff-only
git lfs pull
python verify_package.py
```

Nếu `git status --short` chỉ hiện file log hoặc kết quả bị ignore thì pull bình
thường. Nếu có sửa code trực tiếp trên KV260, lưu phần sửa trước khi pull để
tránh xung đột.

## Kích hoạt môi trường

Sau mỗi lần SSH:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate smolvla_ps
cd ~/Hoangviet/quan_int8_PS
```

Không tạo thêm `.venv` hoặc `.venv_ps`. Chỉ dùng Conda `smolvla_ps` đã cài.

## Chạy khảo sát và xuất CSV

```bash
python -u survey_backends.py \
  --engines qnnpack onednn \
  --threads 4 \
  --warmup 0 \
  --iterations 1 \
  --prefix kv260_02_backend_survey \
  2>&1 | tee results/kv260_02_backend_survey_console.log
```

Kết quả tổng hợp:

```text
results/kv260_02_backend_survey_summary.csv
results/kv260_02_backend_survey_phases.csv
results/kv260_02_backend_survey_blocks.csv
```

## Tải CSV về PC

Từ PowerShell trên PC:

```powershell
Set-Location "D:\TAI_LIEU\DU_AN_XU_LY_TINHIEU\XU_LY_TIN_HIEU_FPGA\IP_FPGA\VLA\quan_int8_PS"
scp "ubuntu@192.168.50.21:~/Hoangviet/quan_int8_PS/results/kv260_02_backend_survey*.csv" ".\results\"
```

Đổi IP và prefix nếu chạy trên bo khác.
