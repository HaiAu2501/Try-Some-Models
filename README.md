# DataFlow-2025

## 📖 Mục lục

- [1. Giới thiệu](#1-giới-thiệu)
  - [1.1. Thành viên nhóm](#11-thành-viên-nhóm)
  - [1.2. Tổng quan dự án](#12-tổng-quan-dự-án)
- [2. Mô tả dự án](#2-mô-tả-dự-án)
  - [2.1. Các công nghệ sử dụng](#21-các-công-nghệ-sử-dụng)
  - [2.2. Luồng thuật toán](#22-luồng-thuật-toán)
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

## 2. Mô tả dự án

### 2.1. Các công nghệ sử dụng

### 2.2. Luồng thuật toán

## 3. Cài đặt

**Bước 1. Tạo bản sao của dự án từ GitHub**

```bash
git clone https://github.com/HaiAu2501/DataFlow-2025.git
```

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

- Đối với các file `.ipynb`, bạn cần sử dụng Jupyter Notebook để chạy chúng bằng việc chọn đúng môi trường ảo đã tạo ở trên; hoặc sử dụng Google Colab.
- Đối với các mô hình, bạn có thể sử dụng các checkpoint với đuôi `.pth` trong thư mục `source/checkpoints` để tải mô hình đã được huấn luyện. Các mô hình có sẵn bao gồm: `TFT`, `TCN`, `HFM`, `VAE`, `DCF`, `PDCF_Central`, `PDCF_East`, `PDCF_West`.

> [!NOTE]  
> Việc huấn luyện lại các mô hình là không cần thiết và có thể tốn kém. Yêu cầu CUDA để huấn luyện được nhanh chóng.
