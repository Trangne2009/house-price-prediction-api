import joblib
import matplotlib.pyplot as plt
import xgboost as xgb

# Tải mô hình XGBoost
try: 
    xgb_model = joblib.load('final_xgb_model.pkl')
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file 'final_xgb_model.pkl'.")
    exit() 

# Lấy các giá trị quan trọng theo 'gain'
importance = xgb_model.get_booster().get_score(importance_type='gain')

# Sắp xếp và chọn 5 đặc trưng quan trọng nhất
sorted_importance = dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))
top_5_features = dict(list(sorted_importance.items())[:5])

# Trực quan hóa Top 5
plt.figure(figsize=(10, 6))
xgb.plot_importance(xgb_model,
                    importance_type='gain',
                    max_num_features=5,
                    title='Top 5 Feature Importance (Gain) - XGBoost Component')
plt.savefig('top_5_feature_importance.png') 
plt.close()

# In ra 5 đặc trưng quan trọng nhất
print("\n--- Top 5 Đặc Trưng Quan Trọng Nhất (theo Gain) ---")
for feature, gain in top_5_features.items():
    print(f"{feature}: {gain:,.2f}")