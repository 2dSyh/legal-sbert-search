import streamlit as st
import os
from src.search_engine import LegalSearchEngine
from src.config import Config

# Cấu hình trang giao diện Streamlit
st.set_page_config(
    page_title="Legal SBERT Search Engine",
    page_icon="⚖️",
    layout="wide"
)

# Khởi tạo Search Engine và lưu vào cache của Streamlit để tránh việc mỗi lần bấm nút lại phải load lại model
@st.cache_resource
def get_search_engine():
    engine = LegalSearchEngine()
    engine.load_search_engine()
    return engine

# Kiểm tra xem file Index đã tồn tại chưa trước khi chạy app
if not os.path.exists(Config.INDEX_PATH):
    st.error("❌ Không tìm thấy dữ liệu Vector Database! Vui lòng chạy lệnh `python main.py --mode index` ở terminal trước.")
else:
    engine = get_search_engine()

    # Giao diện chính
    st.title("⚖️ Hệ Thống Tìm Kiếm Văn Bản Luật Bằng Sentence-BERT")
    st.markdown("Dự án NLP - Tra cứu thông tin pháp luật tối ưu ngữ nghĩa ứng dụng kiến trúc Bi-Encoder.")
    
    st.divider()

    # Thanh tìm kiếm và cấu hình số lượng kết quả trả về
    col1, col2 = st.columns([4, 1])
    with col1:
        user_query = st.text_input("🔍 Nhập câu hỏi hoặc nội dung cần tra cứu pháp luật:", placeholder="Ví dụ: uống rượu bia lái xe phạt bao nhiêu?")
    with col2:
        top_k = st.number_input("Số kết quả (Top K)", min_value=1, max_value=10, value=Config.TOP_K)

    # Xử lý khi người dùng nhấn Enter hoặc nhập câu hỏi
    if user_query:
        with st.spinner("🧠 Đang phân tích ngữ nghĩa và tìm kiếm điều luật liên quan..."):
            results = engine.query(user_query, top_k=int(top_k))
            
        if not results:
            st.warning("Không tìm thấy kết quả phù hợp.")
        else:
            st.subheader(f"📌 Top {len(results)} kết quả phù hợp nhất:")
            
            # Hiển thị từng kết quả dưới dạng thẻ (card) chỉn chu
            for i, res in enumerate(results):
                with st.container():
                    # Phân tách chuỗi "[ID] Nội dung" đã lưu trong DB để hiển thị đẹp hơn
                    text_raw = res["text"]
                    if text_raw.startswith("[") and "]" in text_raw:
                        id_part, content_part = text_raw.split("]", 1)
                        id_law = id_part.replace("[", "").strip()
                    else:
                        id_law = f"Kết quả {i+1}"
                        content_part = text_raw
                    
                    # Tính toán độ khớp tương đối (L2 distance càng nhỏ tức là vector càng gần nhau)
                    st.markdown(f"### 📑 {id_law} *(Khoảng cách vector: {res['score']:.4f})*")
                    st.info(content_part.strip())
                    st.markdown("---")