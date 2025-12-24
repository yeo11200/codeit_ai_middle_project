import os
import argparse
import sys

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.common.config import ConfigLoader
from src.indexing.vector_store import VectorStoreWrapper
from src.retrieval.search import Retriever
from src.generation.rag import RAGChain

def main():
    parser = argparse.ArgumentParser(description="RAG 챗봇: 추론 (Inference) / QA 모드")
    parser.add_argument("--config", type=str, default="config/local.yaml", help="설정 파일 경로")
    parser.add_argument("--mode", type=str, default="qa", choices=["qa", "search"], help="실행 모드 (qa: 질의응답, search: 단순검색)")
    parser.add_argument("--query", type=str, required=True, help="사용자 질문")
    args = parser.parse_args()

    # 1. 설정 로드
    config_loader = ConfigLoader(args.config)
    config = config_loader.config

    # 2. 벡터 저장소 및 검색기 초기화
    vector_store = VectorStoreWrapper(config)
    vector_store.initialize(embedding_function=None) # Mock
    
    # RAG 체인 초기화 (리랭킹 포함)
    rag_chain = RAGChain(config, vector_store)
    
    # 3. 검색 (Retrieve)
    print(f"문서 검색 중 (Hybrid + Rerank): '{args.query}'")
    retrieved_docs = rag_chain.retriever.invoke(args.query)
    
    if args.mode == "search":
        print(f"상위 {len(retrieved_docs)}개 검색 결과:")
        for idx, doc in enumerate(retrieved_docs):
            print(f"[{idx+1}] {doc.page_content[:100]}... (출처: {doc.metadata.get('source', 'unknown')})")
        return

    # 4. 답변 생성 (QA Mode)
    answer = rag_chain.generate_answer(args.query, retrieved_docs)
    
    print("\n=== 질문 (Question) ===")
    print(args.query)
    print("\n=== 답변 (Answer) ===")
    print(answer)
    print("\n=== 참고 문서 (Sources) ===")
    for doc in retrieved_docs:
        print(f"- {doc.metadata.get('source', 'unknown')}")

if __name__ == "__main__":
    main()
