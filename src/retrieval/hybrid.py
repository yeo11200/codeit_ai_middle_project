# [임시 수정] 에러 나는 앙상블 기능을 끄고, 기본 검색만 사용합니다.
# from langchain.retrievers import EnsembleRetriever (삭제)
# from langchain.retrievers.ensemble import EnsembleRetriever (삭제)

def get_hybrid_retriever(vector_store, k=5):
    # 말썽 부리는 앙상블 대신, 튼튼한 기본 벡터 검색기를 반환합니다.
    print("⚠️ 앙상블 검색 기능을 잠시 끄고, 벡터 검색 모드로 실행합니다.")
    return vector_store.as_retriever(search_kwargs={"k": k})