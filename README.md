# House Price Prediction - End-to-End MLOps Project

Dự án này là một quy trình hoàn chỉnh (End-to-End) trong lĩnh vực Machine Learning, từ khâu làm sạch và xử lý dữ liệu thô, xây dựng mô hình dự đoán giá nhà chính xác, cho đến việc triển khai mô hình thành một dịch vụ API công khai và có thể mở rộng (MLOps Foundation).

## DEMO & TƯƠNG TÁC (Interaction)

Bạn có thể tương tác trực tiếp với mô hình đã triển khai để kiểm chứng tính năng **real-time prediction**.

| Kênh Tương tác | Mục đích | Đường dẫn |
| :--- | :--- | :--- |
| **Deployed API (Swagger UI)** | Kiểm thử API trực tiếp (nhập dữ liệu mẫu và nhận kết quả dự đoán). **(Khuyến nghị cho nhà tuyển dụng)** | `https://house-price-prediction-api-7wxj.onrender.com/docs` |
| **Colab Notebook** | Xem toàn bộ mã nguồn, quy trình EDA, Tinh chỉnh mô hình, và các kết quả so sánh cuối cùng. | `https://colab.research.google.com/drive/1yf25-jL6mhG1xvpBr1OaDWWHfyu3BZcd#scrollTo=hjLxbCXAPCKs` |

## MỤC TIÊU DỰ ÁN

* Đạt độ chính xác dự đoán giá nhà tối ưu bằng cách sử dụng kỹ thuật **Ensemble Learning**.
* Hoàn thành chu trình MLOps bằng cách **Dockerize** và **Triển khai** mô hình lên nền tảng Cloud (Render).
* Sử dụng chỉ số **RMSLE** (Root Mean Squared Logarithmic Error) làm metric đánh giá chính, phù hợp với bài toán định giá.
* **Kết quả Cuối cùng (RMSLE):** **0.2505**

## CÁC CÔNG NGHỆ CHỦ YẾU

| Lĩnh vực | Công nghệ | Mục đích |
| :--- | :--- | :--- |
| **Modeling** | XGBoost, Linear Regression, Scikit-learn | Xây dựng và tối ưu hóa các mô hình cơ sở. |
| **Ensembling** | Weighted Averaging | Tăng cường độ chính xác dự đoán. |
| **MLOps** | **FastAPI, Docker, Render** | Xây dựng API, đóng gói môi trường, và triển khai lên Cloud. |
| **Data/Notebook** | Pandas, NumPy, Matplotlib, Colab | Xử lý dữ liệu, phân tích và trực quan hóa. |

***

## QUY TRÌNH PHÁT TRIỂN & KỸ THUẬT NỔI BẬT

### 1. Data Processing & Feature Engineering

* **Log Transformation:** Áp dụng cho biến mục tiêu (`price`) để chuyển đổi phân phối lệch, cải thiện hiệu suất mô hình tuyến tính.
* **Kỹ thuật Đặc trưng:** Tạo các biến thời gian có ý nghĩa kinh doanh (`age`, `agerenovated`).
* **Feature Importance:** Phân tích chỉ số **Gain** của XGBoost để xác định **Top 5** đặc trưng quyết định, khẳng định tầm quan trọng của `sqft_living` và các biến vị trí (`city`).

### 2. Mô hình Hợp thể (Ensemble Learning)

Em đã đạt hiệu suất tối ưu bằng cách kết hợp hai mô hình có các loại lỗi khác nhau:
* **Chiến lược:** **Weighted Averaging** với tỷ lệ tối ưu **65% (XGBoost) : 35% (Linear Regression)**.
* **Lợi ích:** Đạt RMSLE thấp nhất là **0.2505**, cao hơn bất kỳ mô hình đơn lẻ nào.

### 3. Nền tảng MLOps (Deployment)

#### A. API Development & Model Serving
* Xây dựng API bằng **FastAPI**, sử dụng **Pydantic Schema** để xác thực dữ liệu.
* Logic **tiền xử lý phức tạp** và **Ensemble** được tích hợp trực tiếp vào End-point `/predict`.
* Mô hình được tải ở cấp độ **Global Loading** để giảm thiểu độ trễ I/O.

#### B. Dockerization
* Sử dụng **Dockerfile** để đóng gói toàn bộ ứng dụng, đảm bảo **tính nhất quán môi trường** giữa môi trường phát triển cục bộ và Cloud.

#### C. Cloud Deployment
* Triển khai Docker Image lên dịch vụ **Render Web Service**, tạo ra một đường link công khai có thể truy cập 24/7.

***

## HƯỚNG PHÁT TRIỂN TƯƠNG LAI

1.  **Hệ thống Giám sát (Monitoring):** Triển khai **Prometheus & Grafana** để theo dõi hiệu suất mô hình và phát hiện **Data Drift** trong môi trường sản xuất.
2.  **Nâng cấp Ensemble:** Áp dụng kỹ thuật **Stacking** (Stacked Generalization) sử dụng mô hình **Meta-Learner** để đạt được sự kết hợp phi tuyến tính hiệu quả hơn.
>>>>>>> e55f586f7bbc2f2c031e2382d0d501ece63a2a69
3.  **Tối ưu Triển khai:** Chuyển sang **Kubernetes (K8s)** hoặc **Google Cloud Run** để hỗ trợ **Tự động mở rộng (Auto-Scaling)** và quản lý container hiệu quả hơn.