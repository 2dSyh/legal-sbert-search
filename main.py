import argparse
import sys
from src.search_engine import LegalSearchEngine
from src.logger import logger

def main():
    parser = argparse.ArgumentParser(description="Legal SBERT Search Engine CLI Tool")
    parser.add_argument(
        "--mode", 
        type=str, 
        required=True, 
        choices=["index", "test"], 
        help="Chế độ chạy: 'index' để tạo cơ sở dữ liệu vector, 'test' để test nhanh trên terminal."
    )
    
    args = parser.parse_args()
    engine = LegalSearchEngine()

    if args.mode == "index":
        logger.info("Bắt đầu quá trình xây dựng Vector Database...")
        success = engine.build_knowledge_base()
        if success:
            logger.info("Quá trình Indexing hoàn tất thành công!")
        else:
            logger.error("Quá trình Indexing thất bại.")
            
    elif args.mode == "test":
        logger.info("Khởi động chế độ kiểm tra nhanh...")
        try:
            engine.load_search_engine()
            while True:
                query = input("\nNhập câu hỏi tìm kiếm luật (hoặc gõ 'exit' để thoát): ")
                if query.lower() == 'exit':
                    break
                results = engine.query(query, top_k=3)
                print("\n=== KẾT QUẢ TÌM KIẾM ===")
                for i, res in enumerate(results):
                    print(f"\nTop {i+1} (Khoảng cách L2: {res['score']:.4f}):")
                    print(res['text'])
        except Exception as e:
            logger.error(f"Lỗi khi chạy chế độ test: {e}")

if __name__ == "__main__":
    main()