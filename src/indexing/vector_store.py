import os
import chromadb # DB를 직접 들여다보기 위해 추가
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

class VectorStoreWrapper:
    def __init__(self, config):
        # 1. DB 경로 고정 (진단 결과 반영)
        self.persist_directory = "./data/index"
        
        # 2. [핵심] 방 이름(Collection) 자동 찾기
        # 설정 파일이 없으니, 직접 DB를 뒤져서 존재하는 방 이름을 찾아냅니다.
        self.collection_name = "langchain" # 기본값
        try:
            if os.path.exists(self.persist_directory):
                client = chromadb.PersistentClient(path=self.persist_directory)
                collections = client.list_collections()
                if collections:
                    found_name = collections[0].name
                    print(f"🕵️‍♂️ 자동 감지된 방 이름: '{found_name}'")
                    self.collection_name = found_name
                else:
                    print("⚠️ DB는 찾았는데 방(Collection)이 하나도 없습니다.")
        except Exception as e:
            print(f"⚠️ 방 이름 자동 감지 실패 (기본값 사용): {e}")

        # 3. 임베딩 설정
        embeddings_cfg = config.get('embeddings', {})
        model_name = embeddings_cfg.get('model', 'text-embedding-3-small')
        self.embeddings = OpenAIEmbeddings(model=model_name)
        self.vector_store = None

    def initialize(self):
        if os.path.exists(self.persist_directory):
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            print(f"✅ 벡터 스토어 연결 완료! (경로: {self.persist_directory}, 방 이름: {self.collection_name})")
        else:
            print(f"❌ '{self.persist_directory}' 경로를 찾을 수 없습니다.")

    def get_retriever(self):
        return self.vector_store.as_retriever()

    def get_all_documents(self):
        try:
            # DB 내부 데이터를 조회
            data = self.vector_store.get()
            sources = set()
            
            # 메타데이터에서 파일명 추출
            if data and 'metadatas' in data and data['metadatas']:
                for meta in data['metadatas']:
                    if meta:
                        # source가 없으면 file_path 등 다른 키도 찾아봄
                        src = meta.get('source') or meta.get('file_path')
                        if src:
                            sources.add(os.path.basename(src))
            
            doc_list = sorted(list(sources))
            print(f"📂 추출된 문서 목록({len(doc_list)}개): {doc_list}")
            return doc_list
        except Exception as e:
            print(f"⚠️ 문서 목록 조회 실패: {e}")
            return []