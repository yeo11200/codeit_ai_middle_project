from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

class RAGChain:
    def __init__(self, config, vector_store_wrapper, model_name="gemma3:12b"):
        self.config = config
        self.vector_store_wrapper = vector_store_wrapper
        self.model_name = model_name
        
        # 1. 검색기(Retriever) 설정
        self.retriever = self.vector_store_wrapper.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        # 2. LLM 설정
        self.llm = ChatOllama(
            model=self.model_name,
            
            # [청수님 설정] temperature=0.1 (창의성 억제, 사실 기반 답변 유도)
            temperature=0.1,
            
            # [청수님 설정] top_p=0.95 (상위 95% 확률 내에서 단어 선택)
            top_p=0.95,
            
            # [청수님 설정] repetition_penalty=1.1 -> Ollama에서는 'repeat_penalty'
            repeat_penalty=1.1,
            
            # [청수님 설정] max_new_tokens=512 -> Ollama에서는 'num_predict'
            num_predict=512
        )

        # 3. 프롬프트 템플릿
        self.prompt = ChatPromptTemplate.from_template("""
        당신은 RFP(제안요청서) 분석 전문가입니다.
        아래 [Context]에 있는 문서 내용만을 바탕으로 질문에 대해 정확하고 구체적으로 답변하세요.
        문서에 없는 내용은 지어내지 말고 "문서에서 정보를 찾을 수 없습니다"라고 답하세요.
        답변 끝에는 반드시 참고한 문서의 출처나 섹션명을 괄호() 안에 명시해주세요.
        예: (문서의 "사업개요" 부분 참조)

        [Context]
        {context}

        [Question]
        {question}

        [Answer]
        """)

        # 4. 체인 구성 (수정됨: 리트리버를 체인에서 뺌)
        # 이제 체인은 이미 완성된 'context' 문자열과 'question'만 받아서 처리합니다.
        self.chain = (
            self.prompt
            | self.llm
            | StrOutputParser()
        )

    def generate_answer(self, question, selected_docs=[]):
        """
        1. 문서 검색 (Retrieve)
        2. 텍스트 변환 (Format)
        3. 답변 생성 (Generate)
        """
        
        # 필터링 설정
        search_kwargs = {"k": 5}
        if selected_docs:
            if len(selected_docs) == 1:
                search_kwargs["filter"] = {"source": selected_docs[0]}
            else:
                search_kwargs["filter"] = {
                    "$or": [{"source": doc} for doc in selected_docs]
                }
        self.retriever.search_kwargs = search_kwargs
        
        # [단계 1] 문서를 먼저 가져옵니다.
        retrieved_docs = self.retriever.invoke(question)
        
        # [단계 2] 가져온 문서에서 '글자'만 뽑아서 하나의 문자열로 합칩니다. (중요!)
        # 이렇게 하면 AI는 절대 Document 객체(이상한 코드)를 볼 수 없습니다.
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # [단계 3] 깨끗한 텍스트를 체인에 넣어줍니다.
        answer = self.chain.invoke({
            "context": context_text, 
            "question": question
        })
        
        return answer, retrieved_docs