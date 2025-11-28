import numpy as np 
from sklearn.metrics import mean_squared_error
import joblib 

def rmsle(y_true, y_pred):
    # Chuyển đổi ngược Log1P
    y_true_exp = np.expm1(y_true)
    y_pred_exp = np.expm1(y_pred) 
    
    # Tính toán RMSLE bằng cách sử dụng np.log1p cho tính ổn định
    return np.sqrt(mean_squared_error(np.log1p(y_true_exp), np.log1p(y_pred_exp)))


print("--- BẮT ĐẦU KẾT HỢP MÔ HÌNH (ENSEMBLING) ---")
# --- 1. Tải và Chuẩn bị Dữ liệu ---
try: 
    Y_pred_linear = joblib.load('Y_pred_linear.pkl')
    Y_pred_xgb = joblib.load('Y_pred_xgb.pkl')
    Y_test = joblib.load('Y_test_ensemble.pkl')
    print("-> Tải thành công các dự đoán đã lưu.")
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file dự đoán (.pkl). Vui lòng chạy train_model_tuned.py trước.")
    exit() 
    
# Tính RMSLE của từng mô hình (để so sánh)
linear_rmsle = rmsle(Y_test, Y_pred_linear)
xgb_rmsle = rmsle(Y_test, Y_pred_xgb)

print("\n--- KẾT QUẢ MÔ HÌNH KẾT HỢP (XGBoost + Linear Regression) ---")
# Chúng ta sử dụng Averaging 
Y_pred_ensemble = (Y_pred_xgb * 0.65) + (Y_pred_linear * 0.35) 

# Đánh giá mô hình kết hợp
ensemble_rmsle = rmsle(Y_test, Y_pred_ensemble)

print("\n=======================================================")
print("             KẾT QUẢ SO SÁNH CUỐI CÙNG")
print("=======================================================")
print(f"| XGBoost (Tuned)                     | RMSLE: {xgb_rmsle:.4f} |")
print(f"| Linear Regression (Baseline)        | RMSLE: {linear_rmsle:.4f} |")
print(f"| ENSEMBLE (XGB + Linear)             | RMSLE: {ensemble_rmsle:.4f} |")
print("=======================================================")

if ensemble_rmsle < xgb_rmsle:
    print(f"\nCHÚC MỪNG! Mô hình Kết hợp đã cải thiện điểm số thêm {(xgb_rmsle - ensemble_rmsle) * 100:.3f} điểm phần trăm.")
else: 
    print("\nKết quả: Mô hình Kết hợp không cải thiện được điểm số so với XGBoost đơn lẻ.")  
    
print("\n*RMSLE càng nhỏ càng tốt. Mô hình tốt nhất là mô hình có RMSLE nhỏ nhất.")
    