import re
import unicodedata
from underthesea import word_tokenize
from src.logger import logger

class TextProcessor:
    def __init__(self):
        logger.info("Khởi tạo TextProcessor với Underthesea thành công.")

    def clean_text(self, text: str) -> str:
        """
        Hàm làm sạch văn bản luật nâng cao: Chuẩn hóa unicode, xóa ký tự đặc biệt,
        xử lý khoảng trắng thừa, đưa về chữ thường và thực hiện tách từ tiếng Việt.
        """
        if not isinstance(text, str):
            return ""
        
        # 1. Chuẩn hóa Unicode (NFC)
        text = unicodedata.normalize("NFC", text)
        
        # 2. Chuyển thành chữ thường
        text = text.lower()
        
        # 3. Thay thế khoảng trắng đặc biệt thành khoảng trắng thường
        text = re.sub(r'[\r\n\t]', ' ', text)
        
        # 4. Loại bỏ các ký tự đặc biệt không cần thiết
        text = re.sub(r'[^\w\s\d,.\-_:;\(\)]', '', text)
        
        # 5. Xóa khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 6. NÂNG CẤP: Tách từ tiếng Việt (Word Segmentation)
        # format="text" sẽ biến "an toàn giao thông" thành "an_toàn giao_thông"
        try:
            text = word_tokenize(text, format="text")
        except Exception as e:
            logger.warning(f"Lỗi khi tách từ bằng underthesea: {e}. Sử dụng text gốc.")
        
        return text

    def process_dataframe(self, df, text_column: str, output_column: str = "cleaned_text"):
        logger.info(f"Bắt đầu tiền xử lý nâng cao cột '{text_column}'...")
        df[output_column] = df[text_column].apply(self.clean_text)
        logger.info("Tiền xử lý dữ liệu hoàn tất.")
        return df

if __name__ == "__main__":
    processor = TextProcessor()
    sample_text = "Luật giao thông đường bộ quy định về an toàn giao thông."
    print("Kết quả tiền xử lý & tách từ thử nghiệm:\n", processor.clean_text(sample_text))