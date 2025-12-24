# [임시 수정] 에러 나는 리랭커 기능을 끄고, 그대로 통과시킵니다.

def get_reranker_retriever(base_retriever):
    print("⚠️ 리랭커(Reranker) 기능을 끄고, 기본 검색 결과를 그대로 사용합니다.")
    
    # 아무런 가공 없이 원래 검색기를 그대로 반환 (Pass-through)
    return base_retriever