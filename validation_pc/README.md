# PC packaging smoke test

Các file trong thư mục này chỉ chứng minh direct loader và profiler chạy đúng trên PC x86. Chúng không phải kết quả W2 KV260. Khi chạy trên board, runner tạo kết quả thật trong `../results/`; luôn kiểm tra trường `environment.machine` phải là `aarch64` hoặc `arm64` trước khi dùng số đo.
