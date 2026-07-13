import streamlit as st
import os
from src.search_engine import LegalSearchEngine
from src.config import Config

st.set_page_config(
    page_title="Advanced Legal SBERT Search Engine",
    page_icon="⚖️",
    layout="wide"
)

@st.cache_resource
def get_search_engine():
    engine = LegalSearchEngine()
    engine.load_search_engine()
    return engine

if not os.path.exists(Config.INDEX_PATH):
    st.error("❌ Không tìm thấy dữ liệu Vector Database! Vui lòng chạy lệnh `python main.py --mode index` trước.")
else:
    engine = get_search_engine()

    st.title("⚖️ Hệ Thống Tìm Kiếm Luật Nâng Cao (Retrieve & Re-rank)")
    st.markdown("Ứng dụng kiến trúc kết hợp **Bi-Encoder (SBERT)** và **Cross-Encoder** tách từ chuyên sâu bằng **Underthesea**.")
    
    st.divider()

    col1, col2 = st.columns([4, 1])
    with col1:
        user_query = st.text_input("🔍 Nhập câu hỏi pháp luật cần tra cứu:", placeholder="Ví dụ: mức phạt nồng độ cồn khi lái xe?")
    with col2:
        top_k = st.number_input("Số kết quả hiển thị (Top K)", min_value=1, max_value=5, value=Config.TOP_K)

    if user_query:
        with st.spinner("🧠 Hệ thống đang tìm kiếm và tái xếp hạng ngữ nghĩa sâu..."):
            results = engine.query(user_query, top_k=int(top_k))
            
        if not results:
            st.warning("Không tìm thấy kết quả phù hợp.")
        else:
            st.subheader(f"📌 Top {len(results)} kết quả chính xác nhất sau khi Re-rank:")
            
            for i, res in enumerate(results):
                with st.container():
                    text_raw = res["text"]
                    if text_raw.startswith("[") and "]" in text_raw:
                        id_part, content_part = text_raw.split("]", 1)
                        id_law = id_part.replace("[", "").strip()
                    else:
                        id_law = f"Kết quả {i+1}"
                        content_part = text_raw
                    
                    # Điểm số Cross-Encoder (càng cao càng tốt)
                    score_display = f"Điểm tương quan hệ thống: {res['rerank_score']:.4f}"
                    
                    # Bỏ dấu gạch dưới của từ ghép khi hiển thị ra màn hình cho người dùng dễ đọc text tự nhiên
                    clean_display_text = content_part.replace("_", " ").strip()
                    
                    st.markdown(f"### 📑 {id_law} *({score_display})*")
                    st.info(clean_display_text)
                    st.markdown("---")