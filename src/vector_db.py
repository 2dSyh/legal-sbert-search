import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import Config
from src.logger import logger

class VectorDB:
    def __init__(self):
        logger.info(f"Đang tải mô hình SBERT: {Config.MODEL_NAME} trên thiết bị: {Config.DEVICE}")
        # Tải mô hình mã hóa câu hỏi/văn bản
        self.model = SentenceTransformer(Config.MODEL_NAME, device=Config.DEVICE)
        self.index = None
        self.documents = []  # Lưu text gốc tương ứng với từng dòng trong vector DB

    def create_index(self, texts: list):
        """
        Mã hóa danh sách văn bản thành vector và xây dựng FAISS Index.
        """
        logger.info(f"Đang tiến hành nhúng vector cho {len(texts)} văn bản luật...")
        self.documents = texts
        
        # 1. Chuyển đổi text thành embeddings (numpy array)
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        # 2. Lấy kích thước chiều của vector (Embedding Dimension)
        dimension = embeddings.shape[1]
        
        # 3. Khởi tạo Index FAISS sử dụng L2 Distance (hoặc Inner Product nếu thích)
        # IndexFlatL2 phù hợp cho tập dữ liệu vừa và nhỏ, tìm kiếm chính xác 100%
        self.index = faiss.IndexFlatL2(dimension)
        
        # 4. Thêm vector vào database
        # FAISS yêu cầu dữ liệu float32
        self.index.add(embeddings.astype('float32'))
        logger.info("Xây dựng FAISS Index thành công.")

    def save(self, index_path: str, docs_path: str):
        """
        Lưu file index và danh sách text gốc xuống ổ cứng.
        """
        if self.index is None:
            logger.error("Chưa có index nào được tạo để lưu!")
            return
            
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        # Lưu FAISS Index
        faiss.write_index(self.index, str(index_path))
        
        # Lưu danh sách text tương ứng (dùng numpy để lưu nhanh gọn)
        np.save(str(docs_path), np.array(self.documents, dtype=object))
        logger.info(f"Đã lưu các artifact thành công tại thư mục: {os.path.dirname(index_path)}")

    def load(self, index_path: str, docs_path: str):
        """
        Tải index và danh sách text gốc từ ổ cứng lên RAM khi khởi động ứng dụng tìm kiếm.
        """
        if not os.path.exists(index_path) or not os.path.exists(docs_path):
            raise FileNotFoundError("Không tìm thấy file index hoặc file lưu văn bản gốc. Hãy chạy script index trước!")
            
        self.index = faiss.read_index(str(index_path))
        self.documents = np.load(str(docs_path), allow_pickle=True).tolist()
        logger.info("Đã tải thành công FAISS Index và dữ liệu văn bản lên RAM.")

    def search(self, query: str, top_k: int = 5):
        """
        Tìm kiếm top_k văn bản có nội dung tương đồng nhất với câu truy vấn.
        """
        if self.index is None:
            logger.error("Hệ thống chưa load Index, không thể tìm kiếm.")
            return []
            
        # 1. Mã hóa câu hỏi của người dùng thành vector
        query_vector = self.model.encode([query], convert_to_numpy=True).astype('float32')
        
        # 2. Thực hiện tìm kiếm trong FAISS
        # D chứa khoảng cách (distance), I chứa vị trí index (indices) của kết quả
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:  # -1 nghĩa là không tìm đủ kết quả tương thích
                results.append({
                    "text": self.documents[idx],
                    "score": float(dist) # Khoảng cách L2 (càng nhỏ càng gần/giống)
                })
        return results