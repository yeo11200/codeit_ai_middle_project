import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. 설정 로드
load_dotenv()
DB_PATH = "./rfp_database"

# 2. DB 및 검색기(Retriever) 설정
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
if not os.path.exists(DB_PATH):
    print(f"❌ 에러: '{DB_PATH}'가 없습니다. main.py를 먼저 실행하세요.")
    exit()

# DB 불러오기
vectordb = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
# 검색기 생성 (유사도 높은 문서 3개를 가져오도록 설정)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# 3. 프롬프트(지시사항) 만들기
# system: AI에게 역할을 부여합니다.
# human: 실제 질문과 검색된 문맥(context)을 넣어줍니다.
template = """
당신은 '제안요청서(RFP) 전문가'입니다. 
아래의 [문맥(Context)]을 바탕으로 질문에 대해 명확하고 친절하게 답변해 주세요.
만약 문맥에 없는 내용이라면 "제공된 문서에는 해당 내용이 없습니다."라고 솔직하게 말해주세요.

[문맥(Context)]:
{context}


질문: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 4. LLM(두뇌) 설정
model = ChatOpenAI(model="gpt-5-mini", temperature=0)

# 5. 문서 내용을 텍스트로 합치는 함수
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

# 6. RAG 체인 연결 (검색 -> 프롬프트 -> LLM -> 답변출력)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# --- [실제 실행 부분] ---
if __name__ == "__main__":
    print("🤖 AI에게 질문을 던지는 중입니다...")
    
    # 질문 입력
    question = "제안서 평가 방법은 어떻게 돼?"
    print(f"❓ 질문: {question}\n")
    
    # 답변 생성
    response = rag_chain.invoke(question)
    
    print("✅ AI 답변:")
    print("-" * 50)
    print(response)
    print("-" * 50)