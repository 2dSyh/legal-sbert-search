import pandas as pd
import numpy as np
from datasets import load_dataset
from src.config import Config
from src.logger import logger

def download_and_prepare_data(limit: int = 5000):
    logger.info("Bắt đầu tải dữ liệu luật thực tế từ Hugging Face (duyet/vietnamese-legal-instruct)...")
    
    try:
        # Tải dataset từ cache
        ds = load_dataset("duyet/vietnamese-legal-instruct", split="train")
        df_raw = pd.DataFrame(ds)
        logger.info(f"Tải thành công! Tổng số dòng thô thu được: {len(df_raw)}")
        
        extracted_texts = []
        
        if "conversations" in df_raw.columns:
            logger.info("Đang bóc tách dữ liệu từ cột 'conversations'...")
            for conv in df_raw["conversations"]:
                # Xử lý nếu conv là list hoặc numpy array
                if isinstance(conv, (list, tuple, np.ndarray)):
                    for msg in conv:
                        # Kiểm tra nếu element là dict
                        if isinstance(msg, dict):
                            val = str(msg.get("value", "") or msg.get("content", "")).strip()
                            if len(val) > 30:
                                extracted_texts.append(val)
                        elif isinstance(msg, str) and len(msg) > 30:
                            extracted_texts.append(msg.strip())

        # Cơ chế dự phòng (Fallback): Nếu cách trên không lấy được thì trích xuất trực tiếp
        if not extracted_texts:
            logger.warning("Thử cơ chế trích xuất dự phòng cho cấu trúc dữ liệu đặc thù...")
            for conv in df_raw["conversations"]:
                try:
                    # Ép phẳng cấu trúc nếu cần
                    for item in list(conv):
                        if isinstance(item, dict):
                            text = str(item.get("value", "")).strip()
                            if len(text) > 30:
                                extracted_texts.append(text)
                except Exception:
                    continue

        if not extracted_texts:
            logger.error("Không trích xuất được văn bản nào từ dataset!")
            return False

        logger.info(f"Tổng số đoạn văn bản luật trích xuất được: {len(extracted_texts)}")

        # Lọc dữ liệu & chuẩn hóa
        df_processed = pd.DataFrame({"noi_dung": extracted_texts})
        df_processed = df_processed.drop_duplicates(subset=["noi_dung"]).reset_index(drop=True)
        df_processed = df_processed.head(limit)
        
        # Tạo ID điều luật
        df_processed["id_dieu"] = [f"Văn bản/Điều {i+1}" for i in range(len(df_processed))]
        df_processed = df_processed[["id_dieu", "noi_dung"]]
        
        # Lưu file
        df_processed.to_csv(Config.RAW_DATA_PATH, index=False, encoding="utf-8")
        logger.info(f"Đã chuẩn hóa và lưu thành công {len(df_processed)} điều luật thật vào: {Config.RAW_DATA_PATH}")
        return True

    except Exception as e:
        logger.error(f"Có lỗi xảy ra trong quá trình tải dữ liệu: {e}")
        return False

if __name__ == "__main__":
    download_and_prepare_data(limit=5000)