# 🚀 DataFlow-2025

## 📑 Mục lục

- [1. Giới thiệu](#1-giới-thiệu)
  - [1.1. Thành viên nhóm](#11-thành-viên-nhóm)
  - [1.2. Tổng quan dự án](#12-tổng-quan-dự-án)
- [2. Cài đặt](#2-cài-đặt)

## 1. 🚀 Giới thiệu

### 1.1. 👥 Thành viên nhóm

- **Nguyễn Viết Tuấn Kiệt**[^1][^2][^4][^5]: Trưởng nhóm
- **Nguyễn Công Hùng**[^1][^3][^4][^5]: Thành viên
- **Tăng Trần Mạnh Hưng**[^1][^2][^4][^5]: Thành viên
- **Mai Lê Phú Quang**[^1][^2][^4][^5]: Thành viên

[^1]: 🏫 Trường Công nghệ Thông tin và Truyền thông - Đại học Bách Khoa Hà Nội
[^2]: 🎓 Chương trình tài năng - Khoa học máy tính
[^3]: 💻 Khoa học máy tính
[^4]: 🧪 Phòng thí nghiệm Mô hình hóa, Mô phỏng và Tối ưu hóa
[^5]: 🤖 Trung tâm nghiên cứu quốc tế về trí tuệ nhân tạo, BKAI

### 1.2. 🌟 Tổng quan dự án

Dự án này là bài dự thi Vòng Chung kết cuộc thi Data Flow 2025. Với đề bài **Tối ưu hóa chiến lược đầu tư trên thị trường chứng khoán Việt Nam**, nhóm chúng tôi đề xuất 3 giải pháp sau:

1. **Hệ đa chuyên gia** (multi-expert system) gồm các tác tử cộng tác dưới một quy trình làm việc thống nhất. Mỗi tác tử có thể khảo sát thông tin về thị trường theo thời gian thực, đồng thời đóng vai trò chuyên gia của lĩnh vực đặc trưng để phân tích và giải quyết vấn đề dưới góc độ định tính.

2. **Một mô hình cho tất cả** (one-model for-all) là khung làm việc với sự giúp đỡ của 2 mô hình tiên tiến: (B-1) Mô hình dự đoán khoảng tin cậy của giá cổ phiếu của một công ty bất kỳ trong tương lai ngắn hạn. (B-2) Mô hình phân loại các điểm chuyển tiếp trên biến đổi của giá cổ phiếu. Hai khía cạnh này bổ sung cho nhau để hỗ trợ nhà đầu tư phân tích định lượng về thị trường.

3. **Mô phỏng Monte-Carlo** (Monte-Carlo simulation) nhằm mô hình hóa hành vi của thị trường. Đây cũng là phương án giải quyết cuối cùng của bài toán, dựa trên hàng nghìn kịch bản khác nhau mà thống nhất được trọng số phân bổ tài sản cho các danh mục đầu tư theo cách hiệu quả nhất.

## 2. ⚙️ Cài đặt

1. Tạo bản sao dự án:

```bash
git clone https://github.com/HaiAu2501/DataFlow-2025.git
```

2. Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

3. Chạy dự án:

- Bạn có thể chạy các file Jupyter Notebook trong thư mục `ROUND_2/models/` và `ROUND_2/tasks/` để xem kết quả của mô hình.

- Để sủ dụng được hệ đa tác tử, bạn cần tạo một file `.env` trong thư mục `agents` với nội dung như sau:

```bash
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="<your_langsmith_api_key>"
LANGSMITH_PROJECT="<your_project_name>"
OPENAI_API_KEY="<your_openai_api_key>"
GEMINI_API_KEY="<your_gemini_api_key>"
GEMINI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
```

- Sau đó, dùng lệnh `langgraph dev` để khởi động UI của hệ đa chuyên gia. Hoặc chạy file `ROUND_2/agents/main.py` để chạy trực tiếp trên terminal.
