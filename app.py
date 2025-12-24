import streamlit as st
import os
import sys
import time

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

# [Streamlit Cloud Fix] ChromaDB requires sqlite3 > 3.35. 
# On Streamlit Cloud, the default sqlite3 is old. We replace it with pysqlite3.
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from src.common.config import config
from src.generation.rag import RAGChain
from src.indexing.vector_store import VectorStoreWrapper

# 페이지 설정
st.set_page_config(
    page_title="RAG ChatBot",
    page_icon="🤖",
    layout="wide"
)

# 1. 시스템 초기화 (캐싱하여 리소스 절약)
# RAG 시스템은 무거운 객체(벡터 DB 등)를 로드해야 하므로, 매번 실행되지 않도록 캐싱합니다.
@st.cache_resource
def load_rag_system():
    print("RAG 시스템을 로딩 중입니다...")
    vector_store = VectorStoreWrapper(config)
    vector_store.initialize()
    # RAG 메인 체인 인스턴스 생성
    rag_chain = RAGChain(config=config, vector_store_wrapper=vector_store)
    return rag_chain

try:
    rag_chain = load_rag_system()
except Exception as e:
    st.error(f"시스템 초기화 중 오류가 발생했습니다: {e}")
    st.stop()

# 2. 사이드바: 문서 관리 (다중 선택 기능)
with st.sidebar:
    st.header("📄 문서 관리")
    
    # 세션 상태에 '선택된 문서 리스트'가 없으면 초기화
    if "selected_docs" not in st.session_state:
        st.session_state.selected_docs = set()

    # 원본 파일 경로 확인 (config에서 로드)
    files_dir = config['paths'].get('raw_data', 'data/files')
    if os.path.exists(files_dir):
        # 파일 목록 로드
        all_files = sorted(os.listdir(files_dir))
        
        # 2-1. 문서 검색 및 추가 UI
        st.subheader("문서 검색 & 추가")
        search_query = st.text_input("파일명 검색", placeholder="예: 용인시, 공고...")
        
        # 입력된 검색어로 파일 필터링
        filtered_files = [f for f in all_files if search_query.lower() in f.lower()]
        
        # 검색 결과가 있을 때만 선택 박스 표시
        if filtered_files:
            file_to_add = st.selectbox("추가할 문서 선택", filtered_files, key="sb_file_add")
            
            if st.button("➕ 목록에 추가"):
                if file_to_add:
                    st.session_state.selected_docs.add(file_to_add)
                    st.success(f"'{file_to_add}' 추가됨")
                    st.rerun() # UI 갱신을 위해 재실행
        else:
            st.caption("검색 결과가 없습니다.")
            
        st.markdown("---")

        # 2-2. 선택된 문서 목록 확인 및 삭제
        st.subheader(f"선택된 문서 ({len(st.session_state.selected_docs)})")
        
        if not st.session_state.selected_docs:
            st.info("🌐 선택된 문서가 없어 **전체 문서**를 대상으로 검색합니다.")
        else:
            # 집합(Set)을 리스트로 변환하여 순회 (반복 중 수정 방지)
            for doc in list(st.session_state.selected_docs):
                col1, col2 = st.columns([0.8, 0.2])
                with col1:
                    st.text(doc) # 긴 파일명 처리를 위해 text 위젯 사용
                with col2:
                    if st.button("❌", key=f"del_{doc}", help=f"{doc} 삭제"):
                        st.session_state.selected_docs.remove(doc)
                        st.rerun()
            
            if st.button("🗑️ 전체 삭제 (초기화)"):
                st.session_state.selected_docs.clear()
                st.rerun()
                
    # ... (에러 처리 및 기타 사이드바 설정)

    else:
        st.error(f"경로를 찾을 수 없습니다: {files_dir}")
        
    st.markdown("---")
    st.header("⚙️ 설정")
    st.info(f"모델: {config['model']['llm_name']}")
    
    # 답변 길이 조절 슬라이더
    response_level = st.select_slider(
        "답변 길이 조절",
        options=["상세", "보통", "요약", "초요약"],
        value="보통",
        help="상세: 자세한 설명 / 보통: 적절한 길이 / 요약: 핵심만 간단히 / 초요약: 1~2문장"
    )
    
    # 고속 모드 토글
    fast_mode = st.toggle("🚀 고속 모드 (리랭킹 끄기)", value=False, help="정확도는 조금 낮아지지만 속도가 빨라집니다.")
    
    if st.button("대화 내용 초기화"):
        st.session_state.messages = []
        st.rerun()

    if st.button("🔄 시스템 재시작 (캐시 초기화)"):
        st.cache_resource.clear()
        st.rerun()

# 3. 메인 인터페이스: 채팅
st.title("🤖 RAG ChatBot")
st.markdown("입찰 공고(RFP) 문서에 대해 무엇이든 물어보세요!")

# 세션 상태 초기화 (대화 기록)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 대화 내용 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # 저장된 소요 시간이 있으면 표시
        if "elapsed_time" in message:
            st.markdown(f"""
                <p style='color: gray; font-size: 0.8em; opacity: 0.6; margin-top: -10px;'>
                    ⚡ 소요 시간: {message['elapsed_time']:.2f}초
                </p>
            """, unsafe_allow_html=True)
            
        if "sources" in message:
            with st.expander("📚 참고 문서"):
                for src in message["sources"]:
                    st.markdown(f"- **{src.get('metadata', {}).get('source', 'Unknown')}**: {src.get('content', '')[:200]}...")

# 사용자 입력 처리
if prompt := st.chat_input("질문을 입력하세요..."):
    # 1. 사용자 메시지 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. AI 답변 생성
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            with st.spinner("문서를 검색하고 답변을 생성 중입니다..."):
                # 검색 단계 (고속 모드 반영)
                retriever = rag_chain.get_retriever(fast_mode=fast_mode)
                docs = retriever.invoke(prompt)
                
                # 스트리밍 호출 (선택된 레벨 전달)
                start_time = time.time() # 시간 측정 시작
                stream_generator = rag_chain.stream_answer(prompt, docs, level=response_level)
                
                for chunk in stream_generator:
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                
                end_time = time.time() # 시간 측정 종료
                elapsed_time = end_time - start_time
                
                message_placeholder.markdown(full_response)
                
                # 소요 시간 표시 (투명도 적용)
                st.markdown(f"""
                    <p style='color: gray; font-size: 0.8em; opacity: 0.6; margin-top: -10px;'>
                        ⚡ 소요 시간: {elapsed_time:.2f}초
                    </p>
                """, unsafe_allow_html=True)
                
                # 소스 메타데이터 정리
                sources = [{"content": d.page_content, "metadata": d.metadata} for d in docs]
                
                # 소스 표시
                with st.expander("📚 참고 문서"):
                    for d in docs:
                        st.markdown(f"- **{d.metadata.get('source', 'Unknown')}**: {d.page_content[:200]}...")

            # 3. 대화 기록 저장
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources,
                "elapsed_time": elapsed_time # 기록에도 저장
            })
            
        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
