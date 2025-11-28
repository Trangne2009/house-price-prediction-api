from fastapi import FastAPI
from pydantic import BaseModel 
import joblib
import pandas as pd 
import numpy as np 
import uvicorn
from typing import Optional # Dùng cho các trường có thể tùy chọn
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline # Dùng để tải LR Pipeline

# Định nghĩa cấu trúc dữ liệu đầu vào (12 đặc trưng gốc + Năm hiện tại)
class HouseFeatures(BaseModel):
    # Các feature số học và thứ tự:
    bedrooms: int 
    bathrooms: float
    sqft_living: int 
    sqft_lot: int 
    floors: float
    waterfront: int
    view: int 
    condition: int 
    sqft_above: int 
    sqft_basement: int 
    yr_built: int 
    yr_renovated: int 
    
    # Feature phân loại đã được mã hóa trong mô hình
     
    city: str
    
    # Giá trị cần thiết cho Feature Engineering
    current_year: int = 2025 # Năm hiện tại dùng để tính tuổi nhà
    
app = FastAPI(
    title="House Price Prediction",
    description="API dự đoán giá nhà sử dụng mô hình Ensemble (XGBoost + Linear Regression)."
)

# --- 1. TẢI MÔ HÌNH VÀ DỮ LIỆU CẦN THIẾT ---
try: 
    xgb_model = joblib.load('final_xgb_model.pkl')
    # Tải Linear Regression Pipeline (chứa cả Scaler và LR model)
    linear_pipe = joblib.load('final_linear_model.pkl')
    print("-> Đã tải thành công các mô hình Ensemble.")
except FileNotFoundError as e:
    print(f"Lỗi: Không tìm thấy file mô hình: {e}. Đảm bảo đã chạy train_model_and_tuned.py.")
    
# Tải dữ liệu đã xử lý để tạo lại LabelEncoder cho 'city'
try: 
    df_cleaned = pd.read_excel('cleaned_data.xlsx')
    
    X_train_cols = df_cleaned.drop(columns=['price', 'price_log', 'date']).columns
    
    # Lọc ra các cột OHE của city (có tiền tố 'city_')
    city_ohe_cols = [col for col in X_train_cols if col.startswith('city_')]
    print(f"-> Đã tải danh sách {len(city_ohe_cols)} cột OHE của City.")
    
except FileNotFoundError:
    print("Lỗi: Không tìm thấy cleaned_data.xlsx. Không thể lấy danh sách cột.")
    
# --- 2. LOGIC TIỀN XỬ LÝ TRONG API ---
def preprocess_input(data: dict, X_train_cols: pd.Index, city_ohe_cols: list) -> pd.DataFrame:
    """Biến đổi dữ liệu thô (JSON) thành DataFrame đã xử lý (X_train format)."""
    
    df = pd.DataFrame([data])
    
    #1. Feature Engineering
    df['date_year'] = df['current_year']
    df['date_month'] = 1 # Giả định là tháng 1 cho dữ liệu mới
    df['age'] = df['current_year'] - df['yr_built']
    df['is_renovated'] = df['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)
    df['total_sqft'] = df['sqft_living'] + df['sqft_lot']
    
    # 2. ONE-HOT ENCODING CHO CITY
    df_ohe = pd.get_dummies(df['city'], prefix='city')
    df = pd.concat([df, df_ohe], axis=1)
    
    #3. Loại bỏ Cột Thừa (Giống như trong pre_processing.py)
    cols_to_drop = [
        'city', 'current_year', 'statezip', 'street', 'country', 'sqft_lot'
    ]
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # Loại bỏ các cột không dùng (tránh lỗi khi predict)
    df = df.drop(columns=['date', 'price', 'price_log', 'street', 'country', 'sqft_lot'], errors='ignore')
    
    # 4. CĂN CHỈNH THỨ TỰ VÀ SỐ LƯỢNG CỘT
    final_df = pd.DataFrame(columns=X_train_cols, index=[0])
    
    # Đổ dữ liệu từ df đã xử lý vào final_df
    for col in df.columns:
        if col in final_df.columns:
            final_df[col] = df[col]
            
    # Điền 0 cho tất cả các giá trị NaN (Đây là các cột OHE của City không xuất hiện trong input)
    final_df = final_df.fillna(0) 
    
    return final_df

# --- 3. ĐỊNH NGHĨA ENDPOINT ---
@app.post("/predict")
async def predict_price(features: HouseFeatures):
    # 1. TIỀN XỬ LÝ DỮ LIỆU ĐẦU VÀO
    if 'X_train_cols' not in globals():
        return {"error": "Lỗi khởi tạo: Không thể tải danh sách cột từ cleaned_data.xlsx."}
    
    try: 
        input_data_dict = features.model_dump() 
        input_df = preprocess_input(input_data_dict, X_train_cols, city_ohe_cols)
    except Exception as e:
        return {"error": f"Lỗi trong quá trình tiền xử lý: {e}. Vui lòng kiểm tra dữ liệu đầu vào."}
    
    # 2. DỰ ĐOÁN BẰNG MÔ HÌNH THÀNH PHẦN
    try: 
        # Dự đoán bằng XGBoost
        pred_xgb_log = xgb_model.predict(input_df)[0] 
        
        # Dự đoán bằng Linear Regression Pipeline (Tự động Scaling)
        pred_linear_log = linear_pipe.predict(input_df) [0]
        
        # 3. KẾT HỢP DỰ ĐOÁN (ENSEMBLE)
        pred_ensemble_log = (pred_xgb_log * 0.65) + (pred_linear_log * 0.35)
     
        # 4. CHUYỂN ĐỔI NGƯỢC (Inverse Transform)
        final_price = np.expm1(pred_ensemble_log)
        
        return {
            "predicted_price_log": float(f"{pred_ensemble_log:.4f}"),
            "predicted_price": float(f"{final_price:.2f}"),
            "note": "Kết quả là giá trị đã được chuyển đổi ngược từ Log (Đơn vị tiền tệ gốc)."
        }
    except Exception as e: 
        return {"error": f"Lỗi trong quá trình dự đoán mô hình: {e}"}
    
    
    
    