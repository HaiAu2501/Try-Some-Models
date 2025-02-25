# DataFlow-2025

## Mục lục

- [1. Giới thiệu](#1-giới-thiệu)
  - [1.1. Thành viên nhóm](#11-thành-viên-nhóm)
  - [1.2. Tổng quan dự án](#12-tổng-quan-dự-án)
- [2. Hệ thống mô hình](#2-hệ-thống-mô-hình)
  - [2.1. Mô hình cơ sở](#21-mô-hình-cơ-sở)
  - [2.2. Mô hình kết hợp](#22-mô-hình-kết-hợp)
- [3. Cài đặt](#3-cài-đặt)
- [4. Sử dụng](#4-sử-dụng)

## 1. Giới thiệu

### 1.1. Thành viên nhóm

- Nguyễn Viết Tuấn Kiệt[^1][^2][^4][^5]: Trưởng nhóm
- Nguyễn Công Hùng[^1][^3][^4][^5]: Thành viên
- Tăng Trần Mạnh Hưng[^1][^2][^4][^5]: Thành viên
- Mai Lê Phú Quang[^1][^2][^4][^5]: Thành viên

[^1]: Trường Công nghệ Thông tin và Truyền thông - Đại học Bách Khoa Hà Nội
[^2]: Chương trình tài năng - Khoa học máy tính
[^3]: Khoa học máy tính
[^4]: Phòng thí nghiệm Mô hình hóa, Mô phỏng và Tối ưu hóa
[^5]: Trung tâm nghiên cứu quốc tế về trí tuệ nhân tạo, BKAI

### 1.2. Tổng quan dự án

Dự án này trình bày hệ thống các mô hình học sâu tiên tiến cho xử lý dữ liệu chuỗi thời gian trong cuộc thi Data Flow 2025. Sau đó, tiến hành chiến lược học tập tập thể (ensemble learning) để kết hợp các mô hình đã xây dựng và tạo ra một mô hình vượt trội cuối cùng.

## 2. Hệ thống mô hình

### 2.1. Mô hình cơ sở

- **`TFT` (Temporal Fusion Transformer):**

  - Tận dụng sự độc đáo của cơ chế Positional Encoding trong kiến trúc tân tiến Transformer.
  - Bổ sung lớp Multiscale Feature Extractor nhằm trích xuất thông tin chuỗi thời gian ở quy mô khác nhau.
  - Sử dụng Variable Selection Network để chọn ra các biến quan trọng.

- **`TCN` (Temporal Convolutional Network)**

  - Sử dụng kiến trúc mạng CNN 1D để trích xuất thông tin chuỗi thời gian.
  - Tận dụng Residual Connection để tăng tính ổn định và tốc độ học của mô hình.
  - Kết hợp nhánh dựa trên Neural ODE để tăng cường khả năng mô hình hóa thông tin liên tục.

- **`HFM` (Hybrid Forecasting Model)**

  - Tổ hợp thông tin từ các nhánh mang kiến trúc khác nhau: N-HiTS, iTransformer, LSTM.
  - Kết hợp khả năng học của nhiều mô hình tiên tiến, để hiểu dữ liệu trên nhiều khía cạnh.

- **`VAE` (Conditional Variational AutoEncoder)**

  - Ánh xạ dữ liệu vào không gian tiềm ẩn, sau đó tái tạo lại dữ liệu từ không gian này.
  - Cho phép mô hình học được phân phối của dữ liệu, và lấy mẫu từ phân phối này để tạo ra dữ liệu mới.

- **`DCF` (Distributional Conditional Forecast)**

  - Cải tiến của VAE.
  - Cho phép trả về tham số của một phân phối xác suất, thay vì dự đoán giá trị cụ thể.

- **`PFT` (Probabilistic Forecasting Transformer)**

  - Cải tiến của VAE.
  - Sử dụng cơ chế Positional Encoding trong kiến trúc Transformer.
  - Cho phép trả về tham số của một phân phối xác suất, thay vì dự đoán giá trị cụ thể.

- **`CQV` (Conditional Quantile VAE)**
  - Cải tiến của VAE.
  - Cho phép trả về các phân vị của phân phối xác suất.

### 2.2. Mô hình kết hợp

- `META1`: Tổ hợp tuyến tính từ kết quả trả về của `TFT`, `TCN`, `HFM`, `DCF` theo phương pháp OLS.
- `META2`: Tổ hợp tuyến tính theo phương pháp Ridge từ:
  - Kết quả trả về của `TFT`, `TCN`, `HFM`.
  - Cận trên và cận dưới trong khoảng tin cậy 95% từ phân phối do `DCF` trả về.
  - Đặc trưng của dữ liệu.
- `META3`: Cải tiến của `META2`, có sử dụng thêm `PFT`.
- `META4`: Cải tiến từ `META3`, có sử dụng thêm `CQV`.
- `META5` **(best)**: Tổ hợp tuyến tính theo phương pháp Lasso, thực hiện Grid Search trên các siêu tham số.

## 3. Cài đặt

**Điều kiện tiên quyết**

- Python 3.11 hoặc mới hơn.

**Bước 1. Tạo bản sao của dự án từ GitHub**

```bash
git clone https://github.com/HaiAu2501/DataFlow-2025.git
```

- Tiếp theo, di chuyển vào thư mục dự án:

```bash
cd DataFlow-2025
```

- Tại đây, bạn cần tạo thư mục `data` cùng cấp với thư mục `source`, chứa dữ liệu do BTC cung cấp.

**Bước 2. Cài đặt môi trường ảo**

```bash
python -m venv env
```

**Bước 3. Kích hoạt môi trường ảo**

- Windows:

```bash
env\Scripts\activate
```

- MacOS và Linux:

```bash
source env/bin/activate
```

**Bước 4. Cài đặt các thư viện cần thiết**

```bash
pip install -r requirements.txt
```

## 4. Sử dụng

> [!NOTE]  
> Việc huấn luyện lại các mô hình là không cần thiết và có thể tốn kém. Yêu cầu CUDA để huấn luyện được nhanh chóng.

- Đối với các file `.ipynb`, bạn cần sử dụng Jupyter Notebook để chạy chúng bằng việc chọn đúng môi trường ảo đã tạo ở trên; hoặc sử dụng Google Colab.
- Đối với các mô hình cơ sở, bạn có thể sử dụng các checkpoint với đuôi `.pth` trong thư mục `source/checkpoints` để tải mô hình đã được huấn luyện. Các mô hình có sẵn bao gồm: `TFT`, `TCN`, `HFM`, `VAE`, `DCF`, `PFT`, `CQV`, `PDCF_Central`, `PDCF_East`, `PDCF_West`.
- Đối với các mô hình kết hợp, được huấn luyện trong các file `learner.ipynb`, `learner_v2.ipynb`, `learner_v3.ipynb`.
