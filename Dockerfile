# 1. GIAI ĐOẠN XÂY DỰNG (Build Stage)
# Sử dụng Python base image chính thức của Docker
FROM python:3.11-slim

# Thiết lập biến môi trường để Python không ghi file .pyc
ENV PYTHONDONTWRITEBYTECODE 1 
ENV PYTHONUNBUFFERED 1 

# 2. THIẾT LẬP MÔI TRƯỜNG LÀM VIỆC
# Tạo thư mục làm việc trong container
WORKDIR /app 

# 3. CÀI ĐẶT DEPENDENCIES
# Sao chép file yêu cầu 
COPY requirements.txt .

# Cài đặt tất cả các thư viện cần thiết
RUN pip install --upgrade pip 
RUN pip install -r requirements.txt 

# 4. SAO CHÉP CODE VÀ ASSETS
COPY . /app 

# 5. CẤU HÌNH CỔNG
# Khai báo cổng mà ứng dụng FastAPI sẽ chạy
EXPOSE 8000 

# 6. KHỞI CHẠY ỨNG DỤNG
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
