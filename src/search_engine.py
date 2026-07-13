import pandas as pd
import numpy as np
from sentence_transformers import CrossEncoder
from src.config import Config
from src.logger import logger
from src.text_processor import TextProcessor
from src.vector_db import VectorDB

class LegalSearchEngine:
    def __init__(self):
        self.processor = TextProcessor()
        self.db = VectorDB()
        self.reranker = None
        
    def build_knowledge_base(self):
        """Đọc dữ liệu luật thô, tiền xử lý tách từ và đưa vào FAISS index"""
        logger.info(f"Đang đọc dữ liệu luật từ: {Config.RAW_DATA_PATH}")
        try:
            df = pd.read_csv(Config.RAW_DATA_PATH)
        except Exception as e:
            logger.error(f"Không thể đọc file dữ liệu. Lỗi: {e}")
            return False

        if "noi_dung" not in df.columns:
            logger.error("File CSV phải chứa cột 'noi_dung'!")
            return False

        # Tiền xử lý văn bản luật (có kèm tách từ underthesea)
        df = self.processor.process_dataframe(df, text_column="noi_dung", output_column="cleaned_text")
        df.to_csv(Config.PROCESSED_DATA_PATH, index=False, encoding="utf-8")
        
        records = []
        for _, row in df.iterrows():
            id_dieu = row.get("id_dieu", "Không rõ ID")
            cleaned_text = row["cleaned_text"]
            records.append(f"[{id_dieu}] {cleaned_text}")
            
        self.db.create_index(records)
        self.db.save(str(Config.INDEX_PATH), str(Config.DOCS_PATH))
        return True

    def load_search_engine(self):
        """Tải cả FAISS Index (Bi-Encoder) và Mô hình Cross-Encoder lên RAM"""
        # Load FAISS
        self.db.load(str(Config.INDEX_PATH), str(Config.DOCS_PATH))
        
        # Load Reranker Cross-Encoder
        logger.info(f"Đang tải mô hình Reranker Cross-Encoder: {Config.RERANK_MODEL_NAME}")
        self.reranker = CrossEncoder(Config.RERANK_MODEL_NAME, device=Config.DEVICE)

    def query(self, user_query: str, top_k: int = None):
        """Pipeline tìm kiếm nâng cao: Retrieve qua FAISS -> Re-rank qua Cross-Encoder"""
        if top_k is None:
            top_k = Config.TOP_K
            
        # 1. Làm sạch và tách từ câu hỏi đầu vào
        cleaned_query = self.processor.clean_text(user_query)
        logger.info(f"Truy vấn gốc: '{user_query}' -> Tách từ: '{cleaned_query}'")
        
        # 2. GIAI ĐOẠN RETRIEVAL: Lấy nhanh danh sách kết quả ứng viên từ FAISS
        # Lấy số lượng ứng viên nhiều hơn (Config.TOP_K_RETRIEVAL) để không bị sót
        retrieved_results = self.db.search(cleaned_query, top_k=Config.TOP_K_RETRIEVAL)
        
        if not retrieved_results or self.reranker is None:
            return retrieved_results[:top_k]
            
        # 3. GIAI ĐOẠN RE-RANK: Chấm điểm lại các ứng viên bằng Cross-Encoder
        logger.info(f"Bắt đầu giai đoạn Re-rank cho {len(retrieved_results)} ứng viên...")
        
        # Tạo các cặp định dạng đầu vào cho Cross-Encoder: [(câu_hỏi, văn_bản_1), (câu_hỏi, văn_bản_2), ...]
        pairs = [[cleaned_query, res["text"]] for res in retrieved_results]
        
        # Mô hình dự đoán điểm số tương quan (điểm càng cao càng chính xác)
        rerank_scores = self.reranker.predict(pairs)
        
        # Cập nhật lại điểm số mới cho các kết quả
        for i, score in enumerate(rerank_scores):
            retrieved_results[i]["rerank_score"] = float(score)
            
        # Sắp xếp lại danh sách kết quả dựa trên điểm rerank_score giảm dần
        sorted_results = sorted(retrieved_results, key=lambda x: x["rerank_score"], reverse=True)
        
        logger.info("Re-rank hoàn tất.")
        # Trả về số lượng kết quả tinh gọn nhất theo yêu cầu
        return sorted_results[:top_k]