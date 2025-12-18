"""Proposal generation module - generates proposals based on RFP documents."""

import json
from typing import Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from src.common.logger import get_logger


PROPOSAL_PROMPT = """다음 RFP 문서를 분석하여 제안서를 작성하세요.

RFP 내용:
{context}

다음 8개 섹션으로 제안서를 작성하세요:

1. 사업 이해 및 배경
2. 제안 개요
3. 기술 제안
4. 사업 수행 계획
5. 조직 및 인력 구성
6. 예산 및 제안 금액
7. 기대 효과 및 성과
8. 차별화 포인트

각 섹션을 2-3문단으로 작성하세요. RFP 요구사항을 반영하고 전문적으로 작성하세요.
"""


class ProposalGenerator:
    """Generate proposals based on RFP documents."""
    
    def __init__(self, llm, retrieval_agent, config: Dict):
        """
        Initialize ProposalGenerator.
        
        Args:
            llm: LangChain LLM instance
            retrieval_agent: RetrievalAgent instance
            config: Configuration dictionary
        """
        self.llm = llm
        self.retrieval_agent = retrieval_agent
        self.config = config.get("proposal", {})
        self.logger = get_logger(__name__)
        
        # Create prompt template
        system_template = SystemMessagePromptTemplate.from_template(
            "당신은 전문 제안서 작성 전문가입니다. RFP 문서를 분석하여 발주 기관의 요구사항을 완벽히 이해하고, "
            "경쟁력 있고 설득력 있는 제안서를 작성하세요. 기술적 정확성과 비즈니스 가치를 균형있게 제시하세요."
        )
        human_template = HumanMessagePromptTemplate.from_template(PROPOSAL_PROMPT)
        
        self.prompt = ChatPromptTemplate.from_messages([
            system_template,
            human_template
        ])
    
    def generate_from_query(
        self,
        query: str,
        top_k: int = 30,
        company_info: Optional[Dict] = None
    ) -> Dict:
        """
        Generate proposal based on search query.
        
        Args:
            query: Search query to find relevant RFP documents
            top_k: Number of chunks to retrieve
            company_info: Optional company information to include in proposal
        
        Returns:
            Dictionary with proposal content and metadata
        """
        # Retrieve relevant documents
        retrieval_results = self.retrieval_agent.retrieve(query, top_k=top_k)
        
        if not retrieval_results["results"]:
            self.logger.warning(f"No documents found for query: {query}")
            return {
                "proposal": "관련 RFP 문서를 찾을 수 없습니다.",
                "sources": [],
                "query": query
            }
        
        # Build context from retrieved chunks
        # Limit context length to avoid token limits (reduce to 10 chunks for proposals)
        max_context_chunks = min(top_k, 10)  # Limit to 10 chunks for proposals
        chunks_to_use = retrieval_results["results"][:max_context_chunks]
        context = self._build_context(chunks_to_use)
        
        # Add company info to context if provided
        if company_info:
            company_context = self._format_company_info(company_info)
            context = f"{company_context}\n\n{context}"
        
        self.logger.info(f"Context built: {len(chunks_to_use)} chunks, {len(context)} chars")
        
        # Generate proposal
        messages = self.prompt.format_messages(context=context)
        
        self.logger.info(f"Generating proposal for query: {query}, context length: {len(context)} chars")
        
        try:
            response = self.llm.invoke(messages)
            proposal_text = response.content if hasattr(response, 'content') else str(response)
            
            self.logger.info(f"LLM response received, length: {len(proposal_text) if proposal_text else 0} chars")
            
            # Check if response is empty - try with increased max_tokens
            if not proposal_text or not proposal_text.strip():
                self.logger.warning("LLM returned empty proposal. Retrying with increased max_tokens...")
                proposal_text = self._retry_with_increased_tokens(messages)
                
        except Exception as e:
            error_str = str(e)
            self.logger.error(f"LLM invoke failed: {error_str}")
            # Check if it's a model access error
            if "model_not_found" in error_str or "403" in error_str or "does not have access" in error_str:
                self.logger.warning(f"LLM model access error: {e}. Trying fallback models...")
                response = self._try_fallback_llm(messages, error_str)
                proposal_text = response.content if hasattr(response, 'content') else str(response)
            else:
                self.logger.error(f"Failed to generate proposal: {e}", exc_info=True)
                raise
        
        # Final check
        if not proposal_text or not proposal_text.strip():
            self.logger.error("Proposal generation returned empty text after all attempts")
            proposal_text = "제안서 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
        
        # Extract source document IDs
        source_doc_ids = list(set([
            r.get("metadata", {}).get("doc_id", "unknown")
            for r in retrieval_results["results"]
        ]))
        
        return {
            "proposal": proposal_text,
            "sources": source_doc_ids,
            "query": query,
            "total_chunks_used": len(retrieval_results["results"])
        }
    
    def generate_from_doc_id(
        self,
        doc_id: str,
        top_k: int = 30,
        company_info: Optional[Dict] = None
    ) -> Dict:
        """
        Generate proposal for a specific document.
        
        Args:
            doc_id: Document ID
            top_k: Number of chunks to retrieve
            company_info: Optional company information to include in proposal
        
        Returns:
            Dictionary with proposal content and metadata
        """
        # Search for document chunks
        query = f"문서 {doc_id}"
        retrieval_results = self.retrieval_agent.retrieve(query, top_k=top_k)
        
        # Filter to only chunks from this document
        doc_chunks = [
            r for r in retrieval_results["results"]
            if r.get("metadata", {}).get("doc_id") == doc_id
        ]
        
        if not doc_chunks:
            self.logger.warning(f"No chunks found for document {doc_id}")
            return {
                "proposal": "문서를 찾을 수 없습니다.",
                "sources": [],
                "doc_id": doc_id
            }
        
        # Build context
        # Limit context length to avoid token limits (reduce to 10 chunks for proposals)
        max_context_chunks = min(top_k, 10)  # Limit to 10 chunks for proposals
        chunks_to_use = doc_chunks[:max_context_chunks]
        context = self._build_context(chunks_to_use)
        
        # Add company info if provided
        if company_info:
            company_context = self._format_company_info(company_info)
            context = f"{company_context}\n\n{context}"
        
        self.logger.info(f"Context built: {len(chunks_to_use)} chunks, {len(context)} chars")
        
        # Generate proposal
        messages = self.prompt.format_messages(context=context)
        
        try:
            response = self.llm.invoke(messages)
            proposal_text = response.content if hasattr(response, 'content') else str(response)
            
            self.logger.info(f"LLM response received, length: {len(proposal_text) if proposal_text else 0} chars")
            
            # Check if response is empty - try with increased max_tokens
            if not proposal_text or not proposal_text.strip():
                self.logger.warning("LLM returned empty proposal. Retrying with increased max_tokens...")
                proposal_text = self._retry_with_increased_tokens(messages)
                
        except Exception as e:
            error_str = str(e)
            # Check if it's a model access error
            if "model_not_found" in error_str or "403" in error_str or "does not have access" in error_str:
                self.logger.warning(f"LLM model access error: {e}. Trying fallback models...")
                response = self._try_fallback_llm(messages, error_str)
                proposal_text = response.content if hasattr(response, 'content') else str(response)
            else:
                self.logger.error(f"Failed to generate proposal: {e}", exc_info=True)
                raise
        
        # Final check
        if not proposal_text or not proposal_text.strip():
            self.logger.error("Proposal generation returned empty text after all attempts")
            proposal_text = "제안서 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
        
        return {
            "proposal": proposal_text,
            "sources": [doc_id],
            "doc_id": doc_id,
            "total_chunks_used": len(doc_chunks)
        }
    
    def _build_context(self, chunks: List[Dict]) -> str:
        """Build context from chunks."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            chunk_text = chunk.get('chunk_text', '')
            # Limit each chunk to 500 chars to avoid token limits
            if len(chunk_text) > 500:
                chunk_text = chunk_text[:500] + "..."
            context_parts.append(f"[{i}] {chunk_text}")
        return "\n\n".join(context_parts)
    
    def _format_company_info(self, company_info: Dict) -> str:
        """Format company information for context."""
        parts = ["[회사 정보]"]
        
        if company_info.get("company_name"):
            parts.append(f"회사명: {company_info['company_name']}")
        if company_info.get("description"):
            parts.append(f"회사 소개: {company_info['description']}")
        if company_info.get("strengths"):
            parts.append(f"핵심 역량: {', '.join(company_info['strengths'])}")
        if company_info.get("experience"):
            parts.append(f"주요 경험: {company_info['experience']}")
        if company_info.get("technologies"):
            parts.append(f"기술 스택: {', '.join(company_info['technologies'])}")
        
        return "\n".join(parts)
    
    def _retry_with_increased_tokens(self, messages):
        """Retry with increased max_tokens if response was empty."""
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
        
        # Try with increased max_tokens (proposals need more tokens)
        increased_tokens = 4000
        
        try:
            self.logger.info(f"Retrying with max_tokens={increased_tokens}")
            retry_llm = ChatOpenAI(
                model=current_model,
                temperature=temperature,
                max_tokens=increased_tokens,
                api_key=api_key
            )
            
            response = retry_llm.invoke(messages)
            
            # Try multiple ways to get content
            proposal_text = None
            if hasattr(response, 'content'):
                proposal_text = response.content
            elif hasattr(response, 'text'):
                proposal_text = response.text
            elif hasattr(response, 'message'):
                if hasattr(response.message, 'content'):
                    proposal_text = response.message.content
            else:
                proposal_text = str(response)
            
            if proposal_text and proposal_text.strip():
                # Update self.llm for future calls
                self.llm = retry_llm
                self.logger.info(f"Successfully generated proposal with increased max_tokens")
                return proposal_text
            else:
                # Still empty, try fallback models
                self.logger.warning("Still empty after increasing tokens, trying fallback models...")
                return self._try_fallback_llm(messages, "Empty response after token increase")
                
        except Exception as e:
            self.logger.warning(f"Retry with increased tokens failed: {e}, trying fallback models...")
            return self._try_fallback_llm(messages, str(e))
    
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
        
        # Use increased tokens for proposals
        max_tokens = 4000
        
        for fallback_model in LLM_FALLBACK_MODELS:
            if fallback_model == current_model:
                continue  # Skip if already tried
            
            try:
                self.logger.info(f"Trying fallback LLM model: {fallback_model} with max_tokens={max_tokens}")
                fallback_llm = ChatOpenAI(
                    model=fallback_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key=api_key
                )
                
                response = fallback_llm.invoke(messages)
                
                # Try multiple ways to get content
                proposal_text = None
                if hasattr(response, 'content'):
                    proposal_text = response.content
                elif hasattr(response, 'text'):
                    proposal_text = response.text
                elif hasattr(response, 'message'):
                    if hasattr(response.message, 'content'):
                        proposal_text = response.message.content
                else:
                    proposal_text = str(response)
                
                if proposal_text and proposal_text.strip():
                    # Update self.llm for future calls
                    self.llm = fallback_llm
                    self.logger.info(f"Successfully switched to fallback model: {fallback_model}")
                    return proposal_text
                else:
                    self.logger.warning(f"Fallback model '{fallback_model}' returned empty response")
                    continue
                
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

