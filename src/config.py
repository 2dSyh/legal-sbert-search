import os
from pathlib import Path

class Config:
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "legal_data.csv"
    PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "legal_data_cleaned.csv"
    INDEX_PATH = BASE_DIR / "models" / "faiss_index.bin"
    DOCS_PATH = BASE_DIR / "models" / "documents.npy"
    
    # Mô hình Bi-Encoder (Dùng để quét nhanh qua FAISS)
    MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"
    
    # NÂNG CẤP: Mô hình Cross-Encoder (Dùng để Rerank/Tái xếp hạng chính xác cao)
    RERANK_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    
    TOP_K_RETRIEVAL = 10  # FAISS sẽ lấy ra 10 kết quả thô nhanh nhất
    TOP_K = 3             # Sau khi Rerank, chỉ trả về Top 3 kết quả tốt nhất lên màn hình
    
    try:
        import torch
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        DEVICE = "cpu"