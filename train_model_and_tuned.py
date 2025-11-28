import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, make_scorer
import xgboost as xgb 
import joblib 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def rmsle(y_true, y_pred): 
    #Tính chỉ số Root Mean Squared Log Error (RMSLE).
    # Chuyển đổi ngược từ log(x+1) về x (giá trị price ban đầu)
    y_true_exp = np.expm1(y_true) 
    y_pred_exp = np.expm1(y_pred) 
 
    # Áp dụng công thức RMSLE: sqrt(MSE(log(y+1) - log(y_pred+1)))
    return np.sqrt(mean_squared_error(np.log1p(y_true_exp), np.log1p(y_pred_exp)))

# Tạo RMSLE Scorer cho GridSearchCV (Lưu ý: GridSearchCV tìm kiếm MINIMUM score, nên phải đảo ngược dấu)
rmsle_scorer = make_scorer(rmsle, greater_is_better=False) 

print("--- BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH DỰ ĐOÁN GIÁ NHÀ ---")
# --- 1. Tải và Chuẩn bị Dữ liệu ---
try: 
    df = pd.read_excel('cleaned_data.xlsx')
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file 'cleaned_data.xlsx'. Vui lòng chạy pre_processing.py trước.")
    exit() 

print(f"-> Đã tải dữ liệu với {len(df)} dòng và {len(df.columns)} cột.")

# Định nghĩa Biến độc lập (X) và Biến phụ thuộc (Y)
X = df.drop(columns=['price', 'price_log', 'date'])
Y = df['price_log'] # Sử dụng biến mục tiêu đã được log-transform

numerical_features = X.select_dtypes(include=np.number).columns.tolist() 

scaler = StandardScaler() 

numerical_cols_to_scale = X.columns

# Chia tập dữ liệu
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) 
print(f"-> Tập huấn luyện (Training set): {len(X_train)} mẫu.")
print(f"-> Tập kiểm tra (Testing set): {len(X_test)} mẫu.")

# Cấu hình K-Fold (Dùng 5 folds cho tốc độ)
kf = KFold(n_splits=5, shuffle=True, random_state=42) 
all_results = {} 

# Đối tượng Scaler cần lưu
scaler_model = StandardScaler()
scaler_model.fit(X_train[numerical_cols_to_scale])
joblib.dump(scaler_model, 'standard_scaler_final.pkl')
print("-> Đã lưu standard_scaler_final.pkl.")

# --- 2. Huấn luyện Mô hình ---
models = {
    "Linear Regression (Baseline)": LinearRegression(),
    "Ridge Regression": Ridge(alpha=10.0),
    "Lasso Regression": Lasso(alpha=0.001, max_iter=10000),
    "Elastic Net Regression": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000) # Thêm ElasticNet
}

results = {} 

print("\n--- 3. BẮT ĐẦU HUẤN LUYỆN VÀ ĐÁNH GIÁ ---")
for name, model in models.items():
    print(f"Đang huấn luyện: {name}...")
    
    if "Regression" in name:
        full_pipeline = Pipeline(steps=[('scaler', scaler_model),
                                 ('regressor', model)])
    else:
        full_pipeline = model 
 
    # Huấn luyện
    full_pipeline.fit(X_train, Y_train)
    # Dự đoán
    Y_pred = full_pipeline.predict(X_test) 
    # Đánh giá bằng RMSLE
    score = rmsle(Y_test, Y_pred)
    results[name] = score
    print(f"-> {name} - RMSLE: {score:.4f}")

    if name == "Linear Regression (Baseline)":
        # 1. Lưu mô hình Pipeline (final_linear_model.pkl)
        joblib.dump(full_pipeline, 'final_linear_model.pkl')
        print(f"-> ĐÃ LƯU: Mô hình Pipeline '{name}' vào file 'final_linear_model.pkl'.")
        
        # 2. LƯU DỰ ĐOÁN (Y_pred) CỦA LINEAR REGRESSION
        # Lưu dưới dạng mảng numpy để dễ dàng tải lại và sử dụng
        Y_pred_linear = Y_pred
        joblib.dump(Y_pred_linear, 'Y_pred_linear.pkl')
        print(f"-> ĐÃ LƯU: Dự đoán Y_pred của LR vào file 'Y_pred_linear.pkl'.")

print("\n--- A. TINH CHỈNH SIÊU THAM SỐ CHO RIDGE ---")
ridge_pipe = Pipeline(steps=[('scaler', StandardScaler()), ('ridge', Ridge(random_state=42))])

ridge_params = {
    'alpha': [1.0, 5.0, 10.0, 20.0, 50.0, 100.0] # Tham số Regularization L2
}

ridge_grid = GridSearchCV(
    estimator=Ridge(random_state=42),
    param_grid=ridge_params,
    scoring=rmsle_scorer,
    cv=kf,
    verbose=1,
    n_jobs=-1
)

ridge_grid.fit(X_train, Y_train) 
ridge_best = ridge_grid.best_estimator_
Y_pred_ridge = ridge_best.predict(X_test) 
ridge_rmsle = rmsle(Y_test, Y_pred_ridge)

all_results["Ridge (Tuned)"] = ridge_rmsle
print(f"-> Ridge Best Params: {ridge_grid.best_params_}")
print(f"-> Ridge (Tuned) RMSLE trên Test Set: {ridge_rmsle:.4f}")

print("\n--- B. TINH CHỈNH SIÊU THAM SỐ CHO LASSO ---")
lasso_pipe = Pipeline(steps=[('scaler', StandardScaler()), ('lasso', Lasso(random_state=42))])

lasso_params = {
    'alpha': [0.0005, 0.001, 0.002, 0.005], # Tham số Regularization L1
    'max_iter': [20000]
}

lasso_grid = GridSearchCV(
    estimator=Lasso(random_state=42),
    param_grid=lasso_params,
    scoring=rmsle_scorer,
    cv=kf,
    verbose=1,
    n_jobs=-1
)

lasso_grid.fit(X_train, Y_train)
lasso_best = lasso_grid.best_estimator_
Y_pred_lasso = lasso_best.predict(X_test) 
lasso_rmsle = rmsle(Y_test, Y_pred_lasso) 

all_results["Lasso (Tuned)"] = lasso_rmsle
print(f"-> Lasso Best Params: {lasso_grid.best_params_}")
print(f"-> Lasso (Tuned) RMSLE trên Test Set: {lasso_rmsle:.4f}")

print("\n--- C. TINH CHỈNH SIÊU THAM SỐ CHO XGBOOST ---")
# Phạm vi tham số nhỏ cho demo (có thể mở rộng sau)
xgb_params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
    'min_child_weight': [1, 5]
}

xgb_grid = GridSearchCV(
    estimator=xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    ),
    param_grid=xgb_params,
    scoring=rmsle_scorer,
    cv=kf,
    verbose=1,
    n_jobs=-1
)

xgb_grid.fit(X_train, Y_train) 
xgb_best = xgb_grid.best_estimator_
Y_pred_xgb = xgb_best.predict(X_test)
xgb_rmsle = rmsle(Y_test, Y_pred_xgb)

all_results["XGBoost (Tuned)"] = xgb_rmsle
print(f"-> XGBoost Best Params: {xgb_grid.best_params_}")
print(f"-> XGBoost (Tuned) RMSLE trên Test Set: {xgb_rmsle:.4f}")

joblib.dump(xgb_best, 'final_xgb_model.pkl')

print("\n--- LƯU KẾT QUẢ DỰ ĐOÁN ĐỂ ENSEMBLING ---")
# Lưu các mảng dự đoán cần thiết (Y_pred_xgb, Y_pred_linear)
joblib.dump(Y_pred_xgb, 'Y_pred_xgb.pkl')
# Lưu biến mục tiêu Y_test
joblib.dump(Y_test, 'Y_test_ensemble.pkl')

print("-> Đã lưu các file Y_pred_linear.pkl, Y_pred_xgb.pkl, Y_test_ensemble.pkl.")

# Thêm kết quả baseline để so sánh
all_results["Linear Regression (Baseline)"] = 0.2640
all_results["Ridge Regression (Untuned)"] = 0.2640
all_results["Lasso Regression (Untuned)"] = 0.2641
all_results["Elastic Net (Untuned)"] = 0.2654

print("\n=======================================================")
print("          KẾT QUẢ HUẤN LUYỆN VÀ TINH CHỈNH")
print("=======================================================")

for name, score in sorted(all_results.items(), key=lambda item: item[1]):
    print(f"| {name:<35} | RMSLE: {score:.4f} |")
    
print("\n*RMSLE càng nhỏ càng tốt. Mô hình tốt nhất là mô hình có RMSLE nhỏ nhất.")