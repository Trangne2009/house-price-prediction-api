import joblib 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

DATA_FILE = 'cleaned_data.xlsx'
RANDOM_STATE = 42
TEST_SIZE = 0.2

# --- KHỞI ĐẦU: TẢI DỮ LIỆU VÀ MÔ HÌNH ---
try: 
    # 1. Tải Mô hình Linear Regression
    linear_model = joblib.load('final_linear_model.pkl')

    # 2. Tải Đối tượng Scaler đã fit
    scaler_loaded = joblib.load('standard_scaler_final.pkl')

    # 3. Tải Dữ liệu GỐC để tái tạo X_test
    df = pd.read_excel(DATA_FILE)

    # 4. Tải Y_test đã được log-transform
    Y_test = joblib.load('Y_test_ensemble.pkl')

    print("-> Đã tải thành công các file mô hình, scaler và dữ liệu.")
except FileNotFoundError as e:
    print(f"Lỗi: Không tìm thấy file cần thiết. Đảm bảo bạn đã chạy lại script huấn luyện và lưu các file: {e}")
    exit() 

# --- TÁI TẠO X_TEST ĐÃ ĐƯỢC SCALED ---
# Lấy X và loại bỏ cột mục tiêu đã log
X = df.drop(columns=['price', 'price_log', 'date'])

# Chia tách lại X_test
X_train_dummy, X_test_raw, Y_train_dummy, Y_test_dummy = train_test_split(
    X, 
    df['price_log'], # Dùng price_log để đảm bảo chia tách chính xác
    test_size = TEST_SIZE,
    random_state =RANDOM_STATE
)

# Áp dụng Scaler đã fit trên X_test_raw (Sử dụng toàn bộ cột như trong code gốc)
numerical_cols_to_scale = X_test_raw.columns
X_test_scaled_array = scaler_loaded.transform(X_test_raw)
X_test_processed = pd.DataFrame(X_test_scaled_array, columns=X_test_raw.columns, index=X_test_raw.index)

# --- PHÂN TÍCH PHẦN DƯ ---
# 1. Dự đoán giá trị bằng mô hình Linear Regression (trên giá trị LOG)
Y_pred_linear_log = linear_model.predict(X_test_processed)

# 2. Tính toán Phần dư (Residuals)
# Phần dư = Giá trị Thực tế (log) - Giá trị Dự đoán (log)
residuals = Y_test - Y_pred_linear_log

# --- TRỰC QUAN HÓA BIỂU ĐỒ PHẦN DƯ ---
plt.figure(figsize=(10, 6))
# Trục X là Giá trị Dự đoán (log)
sns.scatterplot(x=Y_pred_linear_log, y=residuals, alpha=0.6)

# Vẽ đường tham chiếu y=0
plt.axhline(y=0, color='red', linestyle='--')

plt.xlabel('Giá trị Dự đoán (Log Price)', fontsize=12)
plt.ylabel('Phần dư (Residuals = Log Actual - Log Predicted)', fontsize=14) 
plt.grid(True, linestyle='--')

# Lưu biểu đồ thành PNG
plt.savefig('linear_residual_plot.png')
print("\n--- Phân tích Phần dư Hoàn tất --")
print("Biểu đồ Phần dư đã được lưu tại 'linear_residual_plot.png'")

    