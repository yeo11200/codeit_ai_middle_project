from typing import List, Any
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

class VectorStoreWrapper:
    def __init__(self, config: dict):
        self.config = config
        self.collection_name = config.get("vector_db", {}).get("collection_name", "default_collection")
        self.persist_dir = config.get("vector_db", {}).get("persist_directory", "./data/index")
        self.client = None
        self.vector_store = None

    def initialize(self, embedding_function: Any = None):
        """
        벡터 스토어 클라이언트(Chroma 등)를 초기화합니다.
        
        Args:
            embedding_function: 텍스트를 벡터로 변환하는 임베딩 함수 (기본값: OpenAI)
        """
        db_type = self.config.get("vector_db", {}).get("type", "chroma")
        print(f"{db_type} 벡터 스토어를 {self.persist_dir} 경로에서 초기화합니다...")
        
        # 임베딩 함수가 제공되지 않은 경우 OpenAI Embeddings를 기본으로 사용합니다.
        if embedding_function is None:
            model_name = self.config.get("model", {}).get("embedding_name", "text-embedding-3-small")
            # 환경 변수에 OPENAI_API_KEY가 설정되어 있어야 합니다.
            embedding_function = OpenAIEmbeddings(model=model_name)

        if db_type == "chroma":
            # ChromaDB를 사용하여 벡터 스토어 생성 (로컬 디렉토리에 영구 저장됨)
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=embedding_function,
                persist_directory=self.persist_dir
            )
        else:
            # 지원하지 않는 DB 타입일 경우 경고 메시지 출력 후 Chroma를 기본값으로 사용
            print(f"경고: 지원하지 않는 DB 타입({db_type})입니다. 기본값인 Chroma를 사용합니다.")
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=embedding_function,
                persist_directory=self.persist_dir
            )

    def add_documents(self, documents: List[Any]):
        """
        벡터 스토어에 문서를 추가(적재)합니다.
        
        Args:
            documents: LangChain Document 객체 리스트
        """
        if self.vector_store:
            print(f"벡터 스토어에 {len(documents)}개의 청크를 추가합니다...")
            # ChromaDB에 문서 추가 (자동으로 임베딩 후 저장)
            self.vector_store.add_documents(documents)
            print("문서가 성공적으로 추가되었습니다.")
        else:
            print("벡터 스토어가 초기화되지 않았습니다.")

    def similarity_search(self, query: str, k: int = 5):
        """
        주어진 질문(Query)과 유사한 문서를 검색합니다.
        """
        print(f"다음 검색어로 검색 중: {query}")
        if self.vector_store:
            return self.vector_store.similarity_search(query, k=k)
        return []
