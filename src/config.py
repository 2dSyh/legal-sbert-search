import os
from pathlib import Path

class Config:
    # Đường dẫn thư mục gốc
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    # Đường dẫn dữ liệu và artifact
    RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "legal_data.csv"
    PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "legal_data_cleaned.csv"
    INDEX_PATH = BASE_DIR / "models" / "faiss_index.bin"
    
    # Cấu hình Mô hình SBERT
    # Sử dụng mô hình xuất sắc cho tiếng Việt của BKAI
    MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"
    TOP_K = 5
    
    # Tự động cấu hình phần cứng (Ưu tiên GPU nếu có)
    try:
        import torch
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        DEVICE = "cpu"