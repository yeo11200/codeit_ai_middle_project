"""RAG (Retrieval-Augmented Generation) chain implementation."""

from typing import Dict, List

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from src.common.logger import get_logger


SYSTEM_PROMPT = """당신은 RFP(제안요청서) 문서 분석 전문가입니다. 
제공된 문서 컨텍스트를 기반으로 사용자의 질문에 정확하고 상세하게 답변하세요.

규칙:
1. 반드시 제공된 컨텍스트만을 기반으로 답변하세요.
2. 컨텍스트에 없는 정보는 추측하지 말고 "문서에 명시되지 않음"이라고 명시하세요.
3. 답변의 출처를 명확히 표시하세요 (문서 ID, 섹션명 등).
4. 한국어로 자연스럽고 전문적으로 답변하세요.
"""

USER_PROMPT_TEMPLATE = """다음은 RFP 문서의 관련 부분입니다:

{context}

사용자 질문: {question}

위 컨텍스트를 기반으로 질문에 답변하세요. 답변 끝에 출처를 명시하세요.
"""


class RAGChain:
    """RAG chain for question answering with retrieved context."""
    
    def __init__(self, llm, retrieval_agent, config: Dict):
        """
        Initialize RAG chain.
        
        Args:
            llm: LangChain LLM instance
            retrieval_agent: RetrievalAgent instance
            config: Configuration dictionary
        """
        self.llm = llm
        self.retrieval_agent = retrieval_agent
        self.config = config.get("rag", {})
        self.logger = get_logger(__name__)
        
        # Create prompt template
        system_template = SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT)
        human_template = HumanMessagePromptTemplate.from_template(USER_PROMPT_TEMPLATE)
        
        self.prompt = ChatPromptTemplate.from_messages([
            system_template,
            human_template
        ])
    
    def generate(self, query: str, top_k: int = 5) -> Dict:
        """
        Generate answer using RAG.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
        
        Returns:
            Dictionary with answer and sources
        """
        # Retrieve relevant chunks
        retrieval_results = self.retrieval_agent.retrieve(query, top_k=top_k)
        
        # Build context
        context = self._build_context(retrieval_results["results"])
        
        # Generate prompt
        messages = self.prompt.format_messages(
            context=context,
            question=query
        )
        
        # Call LLM with fallback support
        try:
            response = self.llm.invoke(messages)
        except Exception as e:
            error_str = str(e)
            # Check if it's a model access error
            if "model_not_found" in error_str or "403" in error_str or "does not have access" in error_str:
                self.logger.warning(f"LLM model access error: {e}. Trying fallback models...")
                # Try fallback models
                response = self._try_fallback_llm(messages, error_str)
            else:
                raise
        
        return {
            "answer": response.content,
            "sources": retrieval_results["results"],
            "query": query
        }
    
    def _try_fallback_llm(self, messages, original_error: str):
        """Try fallback LLM models if primary model fails."""
        from src.common.llm_utils import LLM_FALLBACK_MODELS
        from langchain_openai import ChatOpenAI
        import os
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")
        
        # Get current LLM config
        try:
            current_model = self.llm.model_name
        except:
            try:
                current_model = str(self.llm.model)
            except:
                current_model = "unknown"
        
        try:
            temperature = self.llm.temperature
        except:
            temperature = 0.2
        
        try:
            max_tokens = self.llm.max_tokens
        except:
            max_tokens = 2000
        
        for fallback_model in LLM_FALLBACK_MODELS:
            if fallback_model == current_model:
                continue  # Skip if already tried
            
            try:
                self.logger.info(f"Trying fallback LLM model: {fallback_model}")
                fallback_llm = ChatOpenAI(
                    model=fallback_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key=api_key
                )
                
                response = fallback_llm.invoke(messages)
                
                # Update self.llm for future calls
                self.llm = fallback_llm
                self.logger.info(f"Successfully switched to fallback model: {fallback_model}")
                
                return response
                
            except Exception as e:
                error_str = str(e)
                if "model_not_found" in error_str or "403" in error_str or "does not have access" in error_str:
                    self.logger.debug(f"Fallback model '{fallback_model}' not available: {e}")
                else:
                    self.logger.debug(f"Fallback model '{fallback_model}' failed: {e}")
                continue
        
        # If all fallback models failed
        raise ValueError(
            f"All LLM models failed. Original error: {original_error}. "
            f"Tried fallback models: {LLM_FALLBACK_MODELS}. "
            "Please check your OpenAI API access or update config/local.yaml with an available model."
        )
    
    def _build_context(self, results: List[Dict]) -> str:
        """Build context string from retrieval results."""
        context_parts = []
        
        for i, result in enumerate(results, 1):
            metadata = result.get("metadata", {})
            business_name = metadata.get("사업명", metadata.get("business_name", "N/A"))
            section_name = metadata.get("section_name", "N/A")
            
            doc_info = f"[문서: {business_name}, 섹션: {section_name}]"
            chunk_text = result.get("chunk_text", "")
            
            context_parts.append(f"{doc_info}\n{chunk_text}\n")
        
        return "\n---\n".join(context_parts)

