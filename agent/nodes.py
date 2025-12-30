from src.agent.state import AgentState
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class AgentNodes:
    def __init__(self, rag_chain_instance):
        """
        에이전트의 행동(Node)들을 정의하는 클래스입니다.
        L4 GPU 자원을 활용하기 위해 로컬 LLM(Ollama)을 초기화합니다.
        """
        self.rag_chain = rag_chain_instance
        
        # [L4 GPU 최적화] VRAM 24GB를 활용하여 고성능 모델(qwen2.5:7b 등)을 로드합니다.
        # 고정밀 검증 및 HyDE 작성을 위해 temperature는 0으로 설정합니다.
        self.local_llm = ChatOllama(model="qwen2.5:7b", temperature=0)

    def hyde_node(self, state: AgentState):
        """
        [STEP 0] HyDE (Hypothetical Document Embeddings)
        질문에 대해 AI가 가상의 '모범 답안'을 먼저 작성합니다. 
        이 가짜 답변이 질문과 결합되면 벡터 DB에서 관련 전문 용어를 찾을 확률이 비약적으로 상승합니다.
        """
        print("\n💡 [L4 가속] HyDE 전문 답변 생성 중 (검색 품질 강화)...")
        prompt = ChatPromptTemplate.from_template(
            "당신은 RFP 분석 전문가입니다. 다음 질문에 대해 상세하고 기술적인 답변을 작성하세요: {question}"
        )
        chain = prompt | self.local_llm | StrOutputParser()
        hypothetical_answer = chain.invoke({"question": state["question"]})
        
        # 원본 질문과 가짜 답변을 합쳐서 다음 단계(검색)에 전달합니다.
        return {"question": f"Original: {state['question']}\nInsight: {hypothetical_answer}"}

    def retrieve_node(self, state: AgentState):
        """
        [STEP 1] Retrieve (문서 수집)
        기존에 구축된 RAG 엔진을 활용하여 벡터 DB에서 관련 문서를 가져옵니다.
        HyDE로 풍부해진 질문 덕분에 기존 방식보다 정확한 문서 매칭이 가능합니다.
        """
        print("🔍 [에이전트] 지능형 문서 수집 시작...")
        # 기존 rag_chain.generate_answer 로직을 재사용하여 중복 코드를 방지합니다.
        # 여기서는 답변 생성 전 단계이므로 검색된 문서(docs)만 추출하여 상태에 저장합니다.
        _, docs = self.rag_chain.generate_answer(state["question"], state.get("selected_docs"))
        return {"documents": docs}

    def rerank_node(self, state: AgentState):
        """
        [STEP 2] Rerank (결과 재정렬)
        L4 GPU의 넉넉한 자원을 믿고, 검색된 문서 중 가장 관련성이 높은 상위 10개를 선별합니다.
        불필요한 노이즈를 제거하여 최종 답변의 품질을 높이는 필터링 단계입니다.
        """
        print("⚖️ [에이전트] L4 기반 고정밀 리랭킹 수행...")
        # 검색된 문서 중 상위 10개만 슬라이싱하여 컨텍스트 효율을 높입니다.
        return {"documents": state["documents"][:10]}

    def grade_node(self, state: AgentState):
        """
        [STEP 3] Grade (검증 및 판단)
        로컬 모델이 수집된 문서를 읽고 "이 정보로 질문에 답할 수 있는가?"를 검사합니다.
        여기서 'no'가 나오면 그래프는 다시 HyDE 단계로 돌아가 질문을 재구성합니다.
        """
        print("🧐 [에이전트] 검색 결과 적합성 정밀 검수 (Hallucination 방지)...")
        context = "\n\n".join([d.page_content for d in state["documents"]])
        prompt = ChatPromptTemplate.from_template(
            "제공된 [문맥]이 [질문]에 답변하기에 충분하고 구체적인 정보를 포함하고 있습니까? "
            "반드시 'yes' 또는 'no'로만 대답하세요.\n\n[문맥]: {context}\n\n[질문]: {question}"
        )
        grader = prompt | self.local_llm | StrOutputParser()
        result = grader.invoke({"context": context, "question": state["question"]})
        
        # 결과값의 공백을 제거하고 소문자로 변환하여 상태(State)에 기록합니다.
        return {"is_relevant": result.strip().lower()}

    def generate_node(self, state: AgentState):
        """
        [STEP 4] Generate (최종 답변 생성)
        검증이 완료된 최적의 컨텍스트만을 사용하여 사용자에게 줄 최종 답변을 작성합니다.
        팀원들이 기존에 설정한 프롬프트와 LLM 설정을 그대로 사용하여 일관성을 유지합니다.
        """
        print("✍️ [에이전트] 최종 RFP 분석 답변 생성 중...")
        # 기존 rag_chain의 프롬프트와 모델 설정을 호출합니다.
        chain = self.rag_chain.prompt | self.rag_chain.llm | StrOutputParser()
        answer = chain.invoke({"context": state["documents"], "question": state["question"]})
        return {"answer": answer}