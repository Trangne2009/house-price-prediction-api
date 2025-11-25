import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy.stats import skew 
import os 

# --- 1. HÀM TẢI DỮ LIỆU THỰC TẾ ---
def load_raw_data():
    try:
        FILE_PATH = 'data.xlsx' 
        
        df = pd.read_excel(FILE_PATH) 
        print(f"INFO: Đã tải thành công dữ liệu từ file '{FILE_PATH}'.")
        return df 
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file dữ liệu '{FILE_PATH}'. Vui lòng kiểm tra lại đường dẫn/tên file.")
        exit()
    except Exception as e:
        print(f"LỖI KHÔNG XÁC ĐỊNH khi đọc file Excel: {e}")
        exit() 
        
def run_eda(df):
    print("\n--- KHÁM PHÁ DỮ LIỆU (EDA) ---")
    
    print("\nCấu trúc Dữ liệu")
    print(f"-> Kích thước: {df.shape[0]} dòng, {df.shape[1]} cột.")
    df.info() # Hiển thị tóm tắt kiểu dữ liệu và NaN
    
    print("\nPhân tích Biến Mục tiêu (price)")
    # Xử lý các giá trị không phải số trong cột 'price'
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    price_data = df['price'].dropna() 
    initial_skew = skew(price_data) 
    
    print(f"-> Độ lệch (Skewness) ban đầu của Price: {initial_skew:.4f}")
    
    # --- TẠO VÀ LƯU BIỂU ĐỒ PHÂN PHỐI PRICE ---
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1) 
    sns.histplot(price_data, kde=True, bins=50, color='skyblue')
    plt.title('Phân phối của Giá (Price) - Thô')
    plt.xlabel('Price (USD)')
    
    plt.subplot(1, 2, 2) 
    sns.histplot(np.log1p(price_data), kde=True, bins=50, color='lightcoral')
    plt.title('Phân phối của Log(Price + 1) - Minh họa')
    plt.xlabel('Log(Price + 1)')
    
    plt.tight_layout() 
    plt.savefig('eda_price_distribution.png')
    print("-> Đã lưu biểu đồ phân phối Price (Thô và Log) vào eda_price_distribution.png")
    
    print("\nPhân tích Tương quan")
    
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist() 
    correlation_matrix = df[numerical_cols].corr() 
    
    # --- TẠO VÀ LƯU BIỂU ĐỒ HEATMAP ---
    plt.figure(figsize=(10, 8)) 
    sns.heatmap(
        correlation_matrix[['price']].sort_values(by='price', ascending=False),
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
    )
    plt.title('Tương quan của các biến với Price') 
    plt.tight_layout() 
    plt.savefig('eda_correlation_heatmap.png')
    print("-> Đã lưu Ma trận Tương quan (Heatmap) vào eda_correlation_heatmap.png")
    
    top_correlations = correlation_matrix['price'].abs().sort_values(ascending=False).head(5) 
    print("\n--- Top 5 Đặc trưng tương quan mạnh với Price (thô) ---")
    print(top_correlations) 
    
if __name__ == '__main__':
    raw_df = load_raw_data()
    run_eda(raw_df)
    
    
    
    
    