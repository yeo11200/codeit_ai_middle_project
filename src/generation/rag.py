from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class RAGChain:
    def __init__(self, config, vector_store_wrapper):
        self.config = config
        self.vector_store_wrapper = vector_store_wrapper
        
        # [안전장치] 설정값이 없으면 기본 모델(gpt-5-mini) 사용
        llm_cfg = config.get('llm', {})
        model_name = llm_cfg.get('model', 'gpt-5-mini')
        
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0
        )
        
        self.prompt = ChatPromptTemplate.from_template("""
        당신은 RFP(제안요청서) 분석 전문가입니다. 
        아래 제공된 [Context]를 바탕으로 질문에 대해 명확하고 구체적으로 답변해 주세요.
        답변 시, 근거가 되는 내용이 어느 문서의 어떤 부분인지 참고하여 답변하세요.

        [Context]
        {context}

        [Question]
        {question}

        [Answer]:
        """)

    def generate_answer(self, question, selected_docs=None):
        # 1. 검색 필터 설정
        search_kwargs = {"k": 5}
        if selected_docs:
             search_kwargs["filter"] = {"source": {"$in": selected_docs}}

        # 2. 동적 검색기 생성
        retriever = self.vector_store_wrapper.vector_store.as_retriever(
            search_kwargs=search_kwargs
        )
        
        # 3. 문서 검색 및 답변 생성
        docs = retriever.invoke(question)
        
        def format_docs(documents):
            return "\n\n".join([d.page_content for d in documents])

        chain = self.prompt | self.llm | StrOutputParser()
        
        answer = chain.invoke({
            "context": format_docs(docs),
            "question": question
        })
        
        return answer, docs