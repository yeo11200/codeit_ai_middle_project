import streamlit as st
import os
import sys

# [SQLite 호환성 패치]
try:
    import pysqlite3
    if not hasattr(pysqlite3, "sqlite_version_info"):
        pysqlite3.sqlite_version_info = (3, 35, 0)
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

# 설정 파일 로더 (파일 없어도 안전하게 빈 딕셔너리 반환)
try:
    from src.common.config import config
except:
    config = {}

from src.indexing.vector_store import VectorStoreWrapper
from src.generation.rag import RAGChain

st.set_page_config(page_title="RAG ChatBot", page_icon="🤖", layout="wide")

# --- 시스템 로딩 (캐싱) ---
@st.cache_resource
def load_system():
    vector_store = VectorStoreWrapper(config)
    vector_store.initialize()
    rag_chain = RAGChain(config, vector_store)
    return vector_store, rag_chain

try:
    vector_store_wrapper, rag_chain = load_system()
except Exception as e:
    st.error(f"시스템 로딩 실패: {e}")
    st.stop()

# --- 사이드바 UI ---
with st.sidebar:
    st.header("🔧 분석 설정")
    st.markdown("---")
    
    st.subheader("📂 문서 선택 (필터)")
    
    # 문서 목록 가져오기
    all_docs = vector_store_wrapper.get_all_documents()
    
    # 멀티 셀렉트 박스
    selected_docs = st.multiselect(
        "분석할 문서를 선택하세요 (비워두면 전체 검색)",
        options=all_docs,
        default=[] 
    )
    
    st.markdown("---")
    if st.button("🗑️ 대화 기록 지우기", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption("Developed by Joonyoung_Dev")

# --- 메인 화면 ---
st.title("🤖 AI RFP 분석기 (Final Ver.)")

if len(all_docs) > 0:
    st.caption(f"🚀 현재 {len(all_docs)}개의 RFP 문서가 연동되어 있습니다.")
else:
    st.warning("⚠️ 연동된 문서가 없습니다. DB Collection 이름을 확인해주세요.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 대화 기록 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 근거 문서 확인"):
                for src in msg["sources"]:
                    st.markdown(f"- **{src['source']}**: {src['content'][:100]}...")

# 질문 처리
if prompt := st.chat_input("질문을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 분석 중입니다..."):
            answer, docs = rag_chain.generate_answer(prompt, selected_docs)
            
            st.markdown(answer)
            
            sources = []
            if docs:
                sources = [{"source": os.path.basename(d.metadata.get('source', 'Unknown')), "content": d.page_content} for d in docs]
                with st.expander("📚 분석에 사용된 문서"):
                    for s in sources:
                        st.markdown(f"- **{s['source']}**: {s['content'][:200]}...")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })