import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 1. 환경변수 로드 (공용 키 사용)
load_dotenv()

# 2. DB 경로 설정 (아까 만든 그 폴더)
DB_PATH = "./rfp_database"

def test_search():
    # 저장된 DB 불러오기
    print("DB를 불러오는 중...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # DB가 실제로 존재하는지 확인
    if not os.path.exists(DB_PATH):
        print(f"에러: '{DB_PATH}' 폴더가 없습니다. main.py를 먼저 실행하세요!")
        return

    vectordb = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    
    
    # 3. 질문 던져보기
    query = "제안서 평가 방법은?"  # <- 궁금한 질문 아무거나
    print(f"질문: {query}")
    
    # 검색 (유사도 높은 문서 3개 가져오기)
    docs = vectordb.similarity_search(query, k=3)
    
    print(f"\n검색 결과 ({len(docs)}개 발견):")
    print("-" * 50)
    for i, doc in enumerate(docs):
        print(f"[{i+1}] 출처: {os.path.basename(doc.metadata['source'])}")
        print(f"내용 요약: {doc.page_content[:100]}...") # 앞부분 100자만 출력
        print("-" * 50)

if __name__ == "__main__":
    test_search()