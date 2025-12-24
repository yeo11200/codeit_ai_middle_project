from typing import List, Dict, Any
from src.indexing.vector_store import VectorStoreWrapper

from src.retrieval.hybrid import get_hybrid_retriever

class Retriever:
    def __init__(self, vector_store: VectorStoreWrapper, k: int = 4):
        self.vector_store = vector_store
        self.hybrid_retriever = get_hybrid_retriever(
            vector_store=vector_store.vector_store,
            k=k
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[Any]:
        """
        쿼리에 대해 관련성 있는 문서를 검색합니다.
        참고: EnsembleRetriever는 초기화 시 k값이 고정되므로, 여기서 전달되는 top_k는 
        내부적으로 이미 설정된 값을 따르거나 재설정해야 합니다. (현재 구조상 초기화 시점의 k 사용)
        """
        print(f"하이브리드 검색 수행: '{query}'")
        return self.hybrid_retriever.invoke(query)

    def retrieve_with_filter(self, query: str, filters: Dict[str, Any], top_k: int = 5) -> List[Any]:
        # TODO: 메타데이터 필터링 로직 구현 필요
        pass
