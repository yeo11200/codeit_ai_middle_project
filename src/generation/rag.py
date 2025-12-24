from typing import List, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.retrieval.hybrid import get_hybrid_retriever
from src.retrieval.reranker import FlashrankRerank
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever

class RAGChain:
    def __init__(self, config: dict, vector_store_wrapper):
        self.vector_store_wrapper = vector_store_wrapper
        self.llm_name = config.get("model.llm_name", "gpt-5")
        
        # LLM 모델 초기화
        # 환경 변수(OPENAI_API_KEY)가 주입되어 있어야 함
        self.llm = ChatOpenAI(model_name=self.llm_name, temperature=0)
        
        # 1. 하이브리드 검색기 설정 (BM25 + Vector)
        # config에서 retrieval.top_k 값을 가져옵니다 (기본값 3)
        k = config.get("retrieval.top_k", 3)
        self.base_retriever = get_hybrid_retriever(
            vector_store=self.vector_store_wrapper.vector_store,
            k=k
        )
        
        # 기본 리트리버는 base_retriever로 설정
        self.retriever = self.base_retriever

        # 2. 리랭킹(Re-ranking) 검색기 설정 (선택 사항)
        # 검색 정확도를 높이기 위해 FlashRank를 사용합니다.
        self.reranker_retriever = None
        if config.get("retrieval.use_reranker", True):
            compressor = FlashrankRerank()
            self.reranker_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=self.base_retriever
            )
            # 리랭커가 활성화된 경우 이를 기본 리트리버로 설정
            self.retriever = self.reranker_retriever

    def get_retriever(self, fast_mode: bool = False):
        """
        모드에 따라 적절한 검색기를 반환합니다.
        
        Args:
            fast_mode (bool): True일 경우 리랭킹을 건너뛰고 기본(하이브리드) 검색기 사용
        """
        if fast_mode:
            return self.base_retriever
        
        # 고속 모드가 꺼져있지만 리랭커가 초기화되지 않았다면 base를 반환
        return self.reranker_retriever if self.reranker_retriever else self.base_retriever

    def generate_answer(self, query: str, context_docs: List[Any]) -> str:
        """
        사용자의 질문과 검색된 문서를 바탕으로 답변을 생성합니다.
        """
        # 검색된 문서 포맷팅
        if not context_docs:
            context_text = "No relevant context found."
        else:
            # 문서 내용을 하나의 텍스트로 합침
            context_text = "\n\n".join([f"[Document {i+1}]\n{doc.page_content}" for i, doc in enumerate(context_docs)])
        
        # [수정] 2024-12-24: 사용자가 기관에 대해 물어봤을 때, 직접적인 소개가 없어도 관련 문서를 요약하도록 프롬프트 개선
        system_prompt = """당신은 제안요청서(RFP) 문서를 분석하는 전문 어시스턴트입니다.
제공된 문맥(Context)에 기반하여 사용자의 질문에 답변하세요.

[지침]
1. 문맥에서 정확한 답을 찾을 수 있다면, 그 내용을 바탕으로 답변하세요.
2. 만약 질문한 대상(기관, 기업 등)에 대한 '직접적인 소개'가 없더라도, 해당 대상이 작성한 문서나 관련된 사업 내용이 있다면 **그 내용을 바탕으로 요약해서 설명**해 주세요.
   (예: "OOO에 대한 상세한 소개는 없지만, 해당 기관은 'XXX 시스템 고도화' 사업을 발주했습니다. 이 사업은..."와 같이 답변)
3. 정말로 관련된 내용이 문맥에 전혀 없을 때만 "제공된 문서 내용에서 찾을 수 없습니다."라고 답하세요.
4. 답변은 공손하고 전문적인 어조의 한국어로 작성하세요.

중요: 답변은 간결하고 명확하게 작성하세요. 가능하면 핵심 내용을 3~5문장 이내로 요약하십시오. 불필요한 부연 설명은 피하세요."""

        user_prompt = f"""Context:
{context_text}

Question:
{query}

Answer:"""
        
        print(f"{self.llm_name} 모델로 답변 생성 중...")
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"답변 생성 중 오류 발생: {e}"

    def stream_answer(self, query: str, context_docs: List[Any], level: str = "보통"):
        """
        LLM 답변을 실시간 스트리밍으로 생성합니다.
        
        Args:
            query (str): 사용자 질문
            context_docs (List[Any]): 검색된 관련 문서 리스트
            level (str): 답변 길이/상세도 레벨 ('상세', '보통', '요약', '초요약')
        """
        # 문서 포맷팅 (generate_answer와 중복 로직)
        if not context_docs:
            context_text = "관련된 문서를 찾을 수 없습니다."
        else:
            context_text = "\n\n".join([f"[Document {i+1}]\n{doc.page_content}" for i, doc in enumerate(context_docs)])
            
        # 레벨별 지시 사항 정의: 사용자 UI 슬라이더 값에 따라 프롬프트 조정
        level_instructions = {
            "상세": "답변을 매우 구체적이고 상세하게 작성하세요. 문서의 내용을 빠짐없이 설명하고, 필요한 경우 배경 설명도 포함하세요.",
            "보통": "답변을 적절한 길이로 자연스럽게 작성하세요. 핵심 내용과 부가 설명을 균형 있게 포함하세요.",
            "요약": "답변을 간결하게 요약하세요. 핵심 내용 위주로 3~5문장 이내로 작성하세요. 불필요한 설명은 제외하세요.",
            "초요약": "답변을 극도로 짧게 요약하세요. 가장 중요한 핵심 결론만 1~2문장(100자 이내)으로 작성하세요."
        }
        
        # 선택된 레벨이 없으면 '보통'을 기본값으로 사용
        instruction = level_instructions.get(level, level_instructions["보통"])

        # [수정] 2024-12-24: 프롬프트 개선 (유연한 답변 허용)
        system_prompt = f"""당신은 제안요청서(RFP) 문서를 분석하는 전문 어시스턴트입니다.
제공된 문맥(Context)에 기반하여 사용자의 질문에 답변하세요.

[지침]
1. 문맥에서 정확한 답을 찾을 수 있다면, 그 내용을 바탕으로 답변하세요.
2. 만약 질문한 대상(기관, 기업 등)에 대한 '직접적인 소개'가 없더라도, 해당 대상이 작성한 문서나 관련된 사업 내용이 있다면 **그 내용을 바탕으로 요약해서 설명**해 주세요.
   (예: "OOO에 대한 상세한 소개는 없지만, 해당 기관은 'XXX 시스템 고도화' 사업을 발주했습니다. 이 사업은..."와 같이 답변)
3. 정말로 관련된 내용이 문맥에 전혀 없을 때만 "제공된 문서 내용에서 찾을 수 없습니다."라고 답하세요.
4. 답변은 공손하고 전문적인 어조의 한국어로 작성하세요.

중요: {instruction}
"""
        user_prompt = f"""Context:
{context_text}

Question:
{query}

Answer:"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        # 스트리밍 방식으로 토큰 생성 (Generator yield)
        try:
            for chunk in self.llm.stream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            yield f"답변 생성 중 오류 발생: {e}"
