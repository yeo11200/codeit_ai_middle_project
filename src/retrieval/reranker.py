from typing import Sequence, List
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents.compressor import BaseDocumentCompressor
from flashrank import Ranker, RerankRequest

class FlashrankRerank(BaseDocumentCompressor):
    """
    LangChain BaseDocumentCompressor wrapper for FlashRank.
    """
    model: str = "ms-marco-TinyBERT-L-2-v2"
    ranker: Ranker = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, model: str = "ms-marco-TinyBERT-L-2-v2", **kwargs):
        super().__init__(model=model, **kwargs)
        # FlashRank Ranker 초기화 (경량화된 크로스 인코더 모델)
        self.ranker = Ranker(model_name=model)

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks = None,
    ) -> Sequence[Document]:
        """
        검색된 문서들을 FlashRank를 사용하여 재순위화(Re-rank)합니다.
        가장 관련성이 높은 문서 순서대로 정렬하여 반환합니다.
        """
        if not documents:
            return []

        # FlashRank 입력 형식에 맞춰 변환
        passages = [
            {"id": str(i), "text": doc.page_content, "meta": doc.metadata}
            for i, doc in enumerate(documents)
        ]

        # 리랭킹 수행
        rerank_request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(rerank_request)

        # 결과 재조립
        reranked_docs = []
        for res in results:
            doc_idx = int(res["id"])
            original_doc = documents[doc_idx]
            # JSON 직렬화 문제를 방지하기 위해 numpy float를 python float로 변환
            # (FlashRank 일부 버전 이슈 대응)
            score = float(res["score"])
            original_doc.metadata["relevance_score"] = score
            reranked_docs.append(original_doc)

        return reranked_docs
