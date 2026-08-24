# Đưa gói SmolVLA INT8 PS lên GitHub và clone trên KV260

Hai file trọng số trong `package/model/` lớn hơn giới hạn file Git thông thường
của GitHub, vì vậy repository này dùng **Git LFS**. Không bỏ qua bước cài
`git-lfs`; nếu không, KV260 chỉ nhận file con trỏ vài trăm byte thay vì trọng số.

## A. Tạo repository trên GitHub

1. Đăng nhập GitHub và chọn **New repository**.
2. Đặt tên, ví dụ `smolvla-int8-kv260-ps`.
3. Nên chọn **Private** trong giai đoạn thử nghiệm.
4. Không tạo sẵn README, `.gitignore` hoặc license.
5. Sau khi tạo, sao chép URL HTTPS, ví dụ:

   ```text
   https://github.com/Vietcho/smolvla-int8-kv260-ps.git
   ```

> Các lệnh bên dưới giả sử tài khoản là `Vietcho` và repository đã được tạo với
> đúng tên `smolvla-int8-kv260-ps`. Nếu tên thực tế khác, phải thay URL trước khi
> chạy. Không dùng nguyên chữ giữ chỗ như `TEN_GITHUB`.

## B. Push từ PC Windows

Mở PowerShell tại thư mục dự án VLA và thay URL ví dụ bằng URL thật:

```powershell
Set-Location "D:\TAI_LIEU\DU_AN_XU_LY_TINHIEU\XU_LY_TIN_HIEU_FPGA\IP_FPGA\VLA\quan_int8_PS"

# Chỉ chạy dòng này nếu Git báo "detected dubious ownership".
git config --global --add safe.directory "D:/TAI_LIEU/DU_AN_XU_LY_TINHIEU/XU_LY_TIN_HIEU_FPGA/IP_FPGA/VLA/quan_int8_PS"

git lfs install --local
git status
git add .
git commit -m "Add SmolVLA INT8 KV260 PS deployment bundle"
git remote add origin "https://github.com/Vietcho/smolvla-int8-kv260-ps.git"
git push -u origin main
```

Nếu `git remote add origin` báo `remote origin already exists`, không chạy lại
`remote add`. Sửa URL đã lưu rồi push:

```powershell
git remote set-url origin "https://github.com/Vietcho/smolvla-int8-kv260-ps.git"
git remote -v
git push -u origin main
```

Nếu push báo `Repository not found`, kiểm tra cả hai điều kiện:

1. Repository `smolvla-int8-kv260-ps` đã thực sự được tạo trên tài khoản
   `Vietcho`.
2. `git remote -v` không còn URL mẫu hoặc sai tên tài khoản/repository.

Khi GitHub yêu cầu xác thực, đăng nhập qua Git Credential Manager hoặc dùng
Personal Access Token làm mật khẩu. Không ghi token vào file và không gửi token
qua tin nhắn.

Kiểm tra trên PC sau khi push:

```powershell
git lfs ls-files
git remote -v
```

Kết quả phải liệt kê ít nhất:

```text
package/model/linear_int8.safetensors
package/model/non_linear_fp32.safetensors
```

## C. Clone trên KV260

SSH vào KV260 rồi chạy:

```bash
sudo apt update
sudo apt install -y git git-lfs
git lfs install

cd /home/ubuntu
git clone https://github.com/Vietcho/smolvla-int8-kv260-ps.git quan_int8_PS
cd /home/ubuntu/quan_int8_PS
git lfs pull
```

Nếu repository là private, dùng URL SSH sau khi đã thêm SSH key của KV260 vào
GitHub, hoặc xác thực HTTPS bằng tài khoản/token. Không đặt token trực tiếp trong
URL vì nó sẽ bị lưu trong lịch sử shell.

Cách dùng SSH cho repository private:

```bash
ssh-keygen -t ed25519 -C "kv260"
cat ~/.ssh/id_ed25519.pub
```

Sao chép dòng public key vừa in, thêm nó tại **GitHub > Settings > SSH and GPG
keys**, rồi kiểm tra và clone:

```bash
ssh -T git@github.com
cd /home/ubuntu
git clone git@github.com:Vietcho/smolvla-int8-kv260-ps.git quan_int8_PS
cd /home/ubuntu/quan_int8_PS
git lfs pull
```

## D. Xác nhận tải đủ trọng số

```bash
cd /home/ubuntu/quan_int8_PS
git lfs ls-files
ls -lh package/model/*.safetensors
python3 verify_package.py
```

Kích thước mong đợi xấp xỉ:

- `linear_int8.safetensors`: 386 MiB
- `non_linear_fp32.safetensors`: 186 MiB

`verify_package.py` phải báo gói hợp lệ trước khi cài môi trường hoặc benchmark.

## E. Cài môi trường và chạy trên PS

Thực hiện tiếp theo đúng `README.md` trong repository. Tóm tắt:

```bash
cd /home/ubuntu/quan_int8_PS
chmod +x check_ps_environment.sh
./check_ps_environment.sh

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-ps.txt

python run_w2_ps_int8.py --help
```

## F. Cập nhật code sau này

Trên PC:

```powershell
git add .
git commit -m "Mô tả thay đổi"
git push
```

Trên KV260:

```bash
cd /home/ubuntu/quan_int8_PS
git pull
git lfs pull
```
