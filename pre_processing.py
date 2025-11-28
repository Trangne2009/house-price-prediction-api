import pandas as pd
import numpy as np 
from scipy.stats import skew 

# =================================================================
#           HÀM XỬ LÝ 1 & 2 (Giữ nguyên logic 4%)
# =================================================================

def handle_categorical_imputation(df, col, threshold=0.04):
    """
    Xử lý giá trị thiếu cho cột rời rạc dựa trên tỷ lệ thiếu (Ngưỡng: 4%).
    """
    missing_rate = df[col].isnull().sum() / len(df)
    
    if missing_rate == 0:
        return df # Không làm gì nếu không thiếu
    
    if missing_rate < threshold:
        print(f"[{col}] - RỜI RẠC - Tỷ lệ thiếu < {threshold:.0%} ({missing_rate:.2%}). Đang thực hiện XÓA dòng.")
        # Trả về DataFrame đã xóa các dòng NaN của cột này
        return df.dropna(subset=[col]) 
    else:
        print(f"[{col}] - RỜI RẠC - Tỷ lệ thiếu >= {threshold:.0%} ({missing_rate:.2%}). Đang điền bằng 'UNKNOWN'.")
        # Điền giá trị thiếu
        df[col].fillna('UNKNOWN', inplace=True) 
        return df # Trả về DataFrame đã được điền

def handle_continuous_imputation(df, col, threshold=0.04):
    """
    Xử lý giá trị thiếu cho cột liên tục: <4% xóa, >4% kiểm tra ngoại lai.
    """
    missing_rate = df[col].isnull().sum() / len(df)

    if missing_rate == 0:
        return df # Không làm gì nếu không thiếu

    if missing_rate < threshold:
        print(f"[{col}] - LIÊN TỤC - Tỷ lệ thiếu < {threshold:.0%} ({missing_rate:.2%}). Đang thực hiện XÓA dòng.")
        # Trả về DataFrame đã xóa các dòng NaN của cột này
        return df.dropna(subset=[col])
    else:
        Q1 = df[col].quantile(0.25) 
        Q3 = df[col].quantile(0.75) 
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Chỉ đếm ngoại lai trên dữ liệu KHÔNG thiếu
        outliers_count = df[col].loc[~df[col].isnull()][
            (df[col] < lower_bound) | (df[col] > upper_bound)
        ].count()
        
        if outliers_count > 0: 
            median_value = df[col].median() 
            print(f"[{col}] - LIÊN TỤC & Có {outliers_count} ngoại lai. Đã điền bằng Trung Vị ({median_value:.2f}).")
            df[col].fillna(median_value, inplace=True)
            return df
        else: 
            mean_value = df[col].mean() 
            print(f"[{col}] - LIÊN TỤC & Không ngoại lai. Đã điền bằng Trung Bình ({mean_value:.2f}).") 
            df[col].fillna(mean_value, inplace=True)
            return df
# =================================================================
#                       PHẦN THỰC THI CHÍNH
# =================================================================

# --- 0. Phân loại Cột thủ công ---
CONTINUOUS_COLS = ['price', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_above', 'sqft_basement', 'bedrooms', 'view', 'condition', 'waterfront']
DATE_COL = 'date'
ID_COL = ['street', 'statezip', 'city', 'country']
TIME_COLS = ['yr_built', 'yr_renovated']
# Sửa lỗi logic list lồng nhau
ALL_COLS = CONTINUOUS_COLS + [DATE_COL] + ID_COL 

# --- Tải dữ liệu ---
try:
    df = pd.read_excel('data.xlsx')
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file 'data.xlsx'. Đang thoát.")
    exit()

print("--- 1. Bắt đầu Tiền xử lý Bắt buộc ---")
print(f"Dòng ban đầu: {len(df)}") 

# --- A. XỬ LÝ ID (Luôn xóa NaN và Trùng lặp) ---
df.dropna(subset=ID_COL, inplace=True)
df.drop_duplicates(subset=ID_COL, keep='first', inplace=True)
print(f"Dòng sau khi xóa NaN & Trùng lặp ID: {len(df)}")

# --- B. XỬ LÝ LỖI DỮ LIỆU & CHUYỂN ĐỔI KIỂU DỮ LIỆU (Tạo NaN) ---
# 1. Quét lỗi text phổ biến 
for col in df.columns:
    if df[col].dtype == 'object':
        df.loc[df[col].astype(str).str.contains('ERROR|UNKNOWN|^$', na=False, regex=True, case=False), col] = np.nan

# 2. Xử lý các cột số (CONTINUOUS)
for col in CONTINUOUS_COLS + TIME_COLS:
    df[col] = pd.to_numeric(df[col], errors='coerce') 
    
# 3. Xử lý Cột Date (Chỉ nên xóa NaN cho cột date)
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')
df.dropna(subset=[DATE_COL], inplace=True)
print(f"Dòng sau khi xóa NaN cho DATE: {len(df)}")

# --- 2. Áp dụng Logic Imputation (Ngưỡng 4%) ---
print("\n--- 2. Áp dụng Logic Imputation theo Loại Dữ liệu (Ngưỡng 4%) ---")

# Áp dụng cho các cột Liên tục
for col in CONTINUOUS_COLS + TIME_COLS:
    df = handle_continuous_imputation(df, col) # Cần gán lại df vì có thể bị xóa dòng   

# --- Xử lý Skewness và Log Transformation cho PRICE ---
print("\n--- CHUẨN HÓA BIẾN MỤC TIÊU (PRICE) ---")

# 1. Áp dụng Log Transformation (np.log1p)
df['price_log'] = np.log1p(df['price'])
print(f"-> Đã tạo cột price_log (log(price + 1)).")

# Áp dụng IQR để tìm ngoại lai trên dữ liệu đã được log-transform
Q1 = df['price_log'].quantile(0.25)
Q3 = df['price_log'].quantile(0.75)
IQR = Q3 - Q1 
lower_bound = Q1 - 1.5 * IQR 
upper_bound = Q3 + 1.5 * IQR 

# Lọc và xóa các hàng có price_log nằm ngoài giới hạn IQR
df_filtered = df[
    (df['price_log'] >= lower_bound) &
    (df['price_log'] <= upper_bound)
].copy() # Sử dụng .copy() để tránh SettingWithCopyWarning

num_removed = len(df) - len(df_filtered) 
df = df_filtered
print(f"-> Đã xóa {num_removed} hàng ngoại lai dựa trên price_log (Sử dụng IQR).")
print(f"-> Số dòng còn lại sau khi xóa ngoại lai: {len(df)}")

# 3. Tính lại Skewness (Kiểm tra kết quả)
current_skew = skew(df['price_log'])
print(f"-> Độ lệch (Skewness) của Price sau khi xử lý: {current_skew:.4f}")

# ------------------------------------------------------------------
# --- 3. TẠO BIẾN THỜI GIAN (Age & AgeRenovated) ---
# ------------------------------------------------------------------
print("\n--- 3. TẠO BIẾN THỜI GIAN (FEATURE ENGINEERING) ---")

# 1. Tách Năm Bán
df['yr_sold'] = df[DATE_COL].dt.year 

# 2. Tính Age (Tuổi nhà)
df['Age'] = df['yr_sold'] - df['yr_built']
# Xử lý lỗi nếu Age < 0
df['Age'] = df['Age'].apply(lambda x: 0 if x < 0 else x)

# 3. Tính AgeRenovated (Tuổi cải tạo)
# Nếu yr_renovated > 0, tính tuổi cải tạo; ngược lại, dùng tuổi nhà (Age)
df['AgeRenovated'] = np.where(
    df['yr_renovated'] > 0, 
    df['yr_sold'] - df['yr_renovated'],
    df['Age']
)

# Đảm bảo AgeRenovated không âm
df['AgeRenovated'] = df['AgeRenovated'].apply(lambda x: 0 if x < 0 else x)

# Loại bỏ các cột năm ban đầu và yr_sold
df.drop(columns=['yr_built', 'yr_renovated', 'yr_sold'], inplace=True) 
print(f"-> Đã tạo Age và AgeRenovated.")

# 4.1. Định nghĩa các cột cần xóa
COLS_TO_DROP_FINALLY = ['street', 'statezip', 'country']

# 4.2. Hiển thị phân phối TẤT CẢ Thành phố (City)
city_counts = df['city'].value_counts() 
city_percentages = df['city'].value_counts(normalize=True).mul(100).round(2)

city_distribution = pd.DataFrame({
    'Count': city_counts,   
    'Percentage': city_percentages.astype(str) + '%'
})

print("-> Phân phối TẤT CẢ Thành phố (City - Tần suất và Tỷ lệ):")
print(f"Tổng số thành phố duy nhất: {len(city_distribution)}")
print(city_distribution)

# Gộp nhóm CITY (Ngưỡng 1.0%)
print("\n-> Bắt đầu Gộp nhóm và Mã hóa City (Ngưỡng 1.0%)...")
# Tính tỷ lệ phần trăm của mỗi thành phố
city_counts_normalized = df['city'].value_counts(normalize=True)

# Lấy danh sách các thành phố lớn (tỷ lệ >= 1.0%)
large_cities = city_counts_normalized[city_counts_normalized >= 0.01].index.tolist() 

# Gán 'Other_City' cho các thành phố nhỏ hơn 1.0%
df['city_grouped'] = df['city'].apply(lambda x: x if x in large_cities else 'Other_City')

print(f"-> Đã gộp nhóm: {len(large_cities)} thành phố lớn được giữ lại.")
print(f"-> Số lượng nhóm city cuối cùng: {df['city_grouped'].nunique()}")

# Mã hóa One-Hot Encoding cho cột city_grouped
df = pd.get_dummies(df, columns=['city_grouped'], prefix='city', dtype=int)
print("-> Đã thực hiện One-Hot Encoding cho city_grouped.")

# Loại bỏ cột 'city' ban đầu
df.drop(columns=['city'], inplace=True)

# Loại bỏ các cột thừa
df.drop(columns=COLS_TO_DROP_FINALLY, inplace=True)
print(f"-> Đã xóa các cột thừa: {COLS_TO_DROP_FINALLY}")

# --- C. Xóa các hàng còn sót lại NaN (trường hợp hiếm) ---
df.dropna(inplace=True) 

# Xuất file sạch và kết quả cuối cùng
cleaned_file_name = 'cleaned_data.xlsx'
df.to_excel(cleaned_file_name, index=False)
print(f"\nFile đã được lưu vào: {cleaned_file_name}")

print(f"\n--- TIỀN XỬ LÝ HOÀN TẤT ---")
print(f"DataFrame cuối cùng có {df.shape[0]} dòng và {df.shape[1]} cột.")
