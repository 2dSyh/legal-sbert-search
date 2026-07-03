import re
import unicodedata
from src.logger import logger

class TextProcessor:
    def __init__(self):
        logger.info("Khởi tạo TextProcessor thành công.")

    def clean_text(self, text: str) -> str:
        """
        Hàm làm sạch văn bản luật: Chuẩn hóa unicode, xóa ký tự đặc biệt, 
        xử lý khoảng trắng thừa và đưa về dạng chữ thường.
        """
        if not isinstance(text, str):
            return ""
        
        # 1. Chuẩn hóa Unicode (tránh lỗi cùng một chữ nhưng gõ bằng 2 kiểu bảng mã)
        text = unicodedata.normalize("NFC", text)
        
        # 2. Chuyển thành chữ thường
        text = text.lower()
        
        # 3. Thay thế các ký tự xuống dòng, tab thành khoảng trắng
        text = re.sub(r'[\r\n\t]', ' ', text)
        
        # 4. Loại bỏ các ký tự đặc biệt không cần thiết (giữ lại chữ cái, số và dấu câu cơ bản)
        # Bạn có thể điều chỉnh regex này tùy thuộc vào đặc thù dữ liệu luật của bạn
        text = re.sub(r'[^\w\s\d,.\-_:;\(\)]', '', text)
        
        # 5. Xóa khoảng trắng thừa (ví dụ: "luật   pháp" -> "luật pháp")
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def process_dataframe(self, df, text_column: str, output_column: str = "cleaned_text"):
        """
        Áp dụng hàm làm sạch lên toàn bộ một cột trong Pandas DataFrame.
        """
        logger.info(f"Bắt đầu tiền xử lý cột '{text_column}' trong DataFrame...")
        df[output_column] = df[text_column].apply(self.clean_text)
        logger.info("Tiền xử lý dữ liệu hoàn tất.")
        return df

# Test nhanh module khi chạy trực tiếp file này
if __name__ == "__main__":
    processor = TextProcessor()
    sample_text = "Điều 1.   Phạm vi điều chỉnh...\nLuật này quy định về trị an @2026!"
    print("Trước:", sample_text)
    print("Sau:", processor.clean_text(sample_text))