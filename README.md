# Legal Information Retrieval using Sentence-BERT

Dự án áp dụng mô hình **Sentence-BERT (SBERT)** kết hợp với thư viện **FAISS (Facebook AI Similarity Search)** nhằm xây dựng hệ thống tìm kiếm và truy vấn thông tin văn bản luật Tiếng Việt tối ưu về mặt ngữ nghĩa (Semantic Search).

---

## 🚀 Tính năng cốt lõi

- **Tiền xử lý văn bản luật:** Chuẩn hóa Unicode (NFC), làm sạch ký tự đặc biệt và tối ưu hóa cấu trúc câu đặc thù của văn bản pháp luật.
- **Bi-Encoder Embedding:** Sử dụng mô hình Pre-trained SBERT chất lượng cao cho tiếng Việt (`bkai-foundation-models/vietnamese-bi-encoder`) để nhúng ngữ nghĩa câu.
- **Vector Database siêu tốc:** Sử dụng FAISS để lưu trữ và truy vấn vector tương đồng bằng khoảng cách L2 với độ trễ cực thấp.
- **Giao diện trực quan:** Tích hợp giao diện Web Demo giúp người dùng dễ dàng thử nghiệm truy vấn và xem kết quả trực quan.

---

## 📁 Cấu trúc thư mục

```text
legal-sbert-search/
│
├── data/
│   ├── raw/                  # Dữ liệu luật thô do Researchers cung cấp
│   └── processed/            # Dữ liệu luật sau khi được làm sạch
├── models/                   # Nơi lưu trữ FAISS index và mảng document (.npy)
├── src/                      # Mã nguồn hệ thống
│   ├── config.py             # Cấu hình tập trung (Model, Paths, Hardware)
│   ├── logger.py             # Quản lý log hệ thống
│   ├── text_processor.py     # Tiền xử lý văn bản
│   ├── vector_db.py          # Quản lý FAISS (Create, Save, Load, Search)
│   └── search_engine.py      # Core logic kết hợp pipeline tìm kiếm
├── app.py                    # Giao diện Web Demo
└── requirements.txt          # Danh sách thư viện phụ thuộc
```

---

## 🛠️ Hướng dẫn cài đặt

### 1. Chuẩn bị môi trường

Yêu cầu máy cài đặt sẵn **Python 3.9+**. Mở terminal tại thư mục dự án và chạy các lệnh sau:

```bash
# Khởi tạo môi trường ảo (Dành cho Windows PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Cập nhật pip và cài đặt thư viện
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu

Đặt file dữ liệu luật của bạn vào đường dẫn `data/raw/legal_data.csv`. File cần có định dạng tối thiểu gồm cột chứa nội dung văn bản pháp luật (Ví dụ: `noi_dung`).

---

## 💻 Hướng dẫn sử dụng

### Bước 1: Trích xuất Vector và Khởi tạo Database

Chạy script để làm sạch dữ liệu và tạo file Index Vector (chỉ cần chạy 1 lần duy nhất khi có dữ liệu mới):

```bash
python main.py --mode index
```

### Bước 2: Khởi chạy Giao diện tìm kiếm (Demo)

Khởi động giao diện Web để tương tác trực quan:

```bash
streamlit run app.py
```

---

## 👥 Thành viên tham gia

- **Coder:** 2dSyh – Phát triển hệ thống, Pipeline & Giao diện
- **Researchers:** [N/A] – Thu thập, gán nhãn dữ liệu & Nghiên cứu mô hình