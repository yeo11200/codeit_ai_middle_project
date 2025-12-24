from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.vectorstores import VectorStore

def get_hybrid_retriever(
    vector_store: VectorStore, 
    documents: List[Document] = None, 
    k: int = 4,
    weights: List[float] = [0.5, 0.5]
) -> BaseRetriever:
    """
    BM25(키워드 검색)와 VectorStore(의미 검색)를 결합한 앙상블 검색기(EnsembleRetriever)를 생성합니다.
    
    Args:
        vector_store: Chroma 등의 벡터 스토어 인스턴스
        documents: BM25 초기화를 위한 문서 리스트. 
                   None일 경우 vector_store에서 문서를 가져오려고 시도합니다.
        k: 각 검색기가 반환할 문서 수 (top-k)
        weights: [BM25 가중치, Vector 검색 가중치] 비율 (기본값: 0.5:0.5)
        
    Returns:
        EnsembleRetriever: 결합된 검색기 객체
    """
    
    # 1. 벡터 검색기(Vector Retriever) 초기화
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": k})
    
    # 2. BM25 검색기(BM25 Retriever) 초기화
    if not documents:
        # DB에서 전체 문서를 가져와서 BM25 인덱스 생성
        try:
            # VectorStoreWrapper 내부의 실제 스토어 객체 혹은 Chroma 객체인지 확인
            if hasattr(vector_store, "get"): 
                # 메타데이터와 텍스트를 모두 가져옴 (데이터 양이 많으면 무거울 수 있음)
                result = vector_store.get() 
                texts = result['documents']
                metadatas = result['metadatas']
                documents = [
                    Document(page_content=t, metadata=m) 
                    for t, m in zip(texts, metadatas) if t # None 필터링
                ]
            else:
                print("경고: 벡터 스토어에서 문서를 가져올 수 없어 BM25 초기화에 실패했습니다. 하이브리드 검색 대신 벡터 검색만 사용합니다.")
                return vector_retriever
        except Exception as e:
            print(f"BM25 문서 로드 중 오류 발생: {e}")
            return vector_retriever

    if not documents:
        print("경고: BM25 초기화에 필요한 문서가 없습니다. 벡터 검색기만 반환합니다.")
        return vector_retriever

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k

    # 3. 앙상블(Ensemble) 구성
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=weights
    )
    
    return ensemble_retriever
