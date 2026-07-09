import pandas as pd
from src.config import Config
from src.logger import logger
from src.text_processor import TextProcessor
from src.vector_db import VectorDB

class LegalSearchEngine:
    def __init__(self):
        self.processor = TextProcessor()
        self.db = VectorDB()
        
    def build_knowledge_base(self):
        """
        Đọc dữ liệu thô, làm sạch và đưa vào FAISS để tạo cơ sở dữ liệu vector.
        """
        logger.info(f"Đang đọc dữ liệu luật từ: {Config.RAW_DATA_PATH}")
        try:
            # Giả định file CSV có cột 'id_dieu' và 'noi_dung'
            df = pd.read_csv(Config.RAW_DATA_PATH)
        except Exception as e:
            logger.error(f"Không thể đọc file dữ liệu. Lỗi: {e}")
            logger.warning("Hãy đảm bảo bạn đã đặt file dữ liệu vào đúng thư mục data/raw/")
            return False

        if "noi_dung" not in df.columns:
            logger.error("File CSV phải chứa cột 'noi_dung'!")
            return False

        # Tiền xử lý dữ liệu text văn bản luật
        df = self.processor.process_dataframe(df, text_column="noi_dung", output_column="cleaned_text")
        
        # Lưu file đã làm sạch xuống thư mục processed để theo dõi nếu cần
        df.to_csv(Config.PROCESSED_DATA_PATH, index=False, encoding="utf-8")
        
        # Đưa danh sách text đã làm sạch vào lập chỉ mục FAISS
        # Để khi trả về kết quả hiển thị đẹp, ta có thể lưu kết hợp cả ID và text gốc
        # Ở đây để tối giản, chúng ta lưu text gốc làm sạch hoặc tạo một định dạng hiển thị
        records = []
        for _, row in df.iterrows():
            id_dieu = row.get("id_dieu", "Không rõ ID")
            cleaned_text = row["cleaned_text"]
            # Tạo chuỗi định dạng để lưu vào DB: "[ID] Nội dung"
            records.append(f"[{id_dieu}] {cleaned_text}")
            
        self.db.create_index(records)
        
        # Lưu index và danh sách text xuống ổ cứng
        self.db.save(str(Config.INDEX_PATH), str(Config.BASE_DIR / "models" / "documents.npy"))
        return True

    def load_search_engine(self):
        """
        Tải dữ liệu index đã có sẵn lên RAM để chuẩn bị tìm kiếm.
        """
        self.db.load(str(Config.INDEX_PATH), str(Config.BASE_DIR / "models" / "documents.npy"))

    def query(self, user_query: str, top_k: int = None):
        """
        Nhận câu hỏi của người dùng, làm sạch và tìm kiếm trong DB.
        """
        if top_k is None:
            top_k = Config.TOP_K
            
        # Làm sạch câu hỏi đầu vào giống hệt cách làm sạch data luật
        cleaned_query = self.processor.clean_text(user_query)
        logger.info(f"Truy vấn từ người dùng: '{user_query}' -> Chuẩn hóa: '{cleaned_query}'")
        
        # Tìm kiếm
        return self.db.search(cleaned_query, top_k=top_k)