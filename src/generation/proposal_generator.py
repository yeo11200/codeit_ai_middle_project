"""Proposal generation module - generates proposals based on RFP documents."""

import json
from typing import Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from src.common.logger import get_logger


PROPOSAL_PROMPT = """당신은 전문 제안서 작성 전문가입니다. 아래 제공된 RFP(제안요청서) 문서의 모든 정보를 철저히 분석하고, 그 내용을 바탕으로 발주 기관에 제출할 전문적인 사업 제안서를 작성하세요.

중요: RFP 문서에 명시된 구체적인 정보(사업명, 예산, 마감일, 요구사항, 기술 스펙 등)를 반드시 반영하여 작성하세요. 단순 요약이 아닌, 실제 제안서 형태로 작성하세요.

=== RFP 문서 내용 ===
{context}
=== RFP 문서 내용 끝 ===

위 RFP 문서를 분석하여 다음 8개 섹션으로 구성된 전문 제안서를 작성하세요. 각 섹션은 최소 3-5문단으로 상세하게 작성하세요:

## 1. 사업 이해 및 배경
- RFP 문서에 명시된 사업의 핵심 목적과 배경을 상세히 설명
- 발주 기관이 추진하는 사업의 필요성과 중요성 분석
- 사업 추진 배경 및 정책적 맥락 설명
- 우리가 이 사업을 이해한 내용을 구체적으로 기술

## 2. 제안 개요
- 우리의 핵심 가치 제안 및 접근 철학
- RFP 요구사항에 대한 우리의 해석과 접근 방식
- 우리가 제공할 솔루션의 핵심 차별화 포인트
- 사업 성공을 위한 우리의 전략적 관점

## 3. 기술 제안
- RFP에 명시된 기술 요구사항을 충족하는 시스템 아키텍처 제안
- 핵심 기술 스택 및 플랫폼 구성 방안
- 주요 기능 모듈 및 시스템 구성도
- 기술적 우수성 및 혁신성 (성능, 보안, 확장성 등)
- RFP에 명시된 기술 스펙을 반영한 구체적 기술 방안

## 4. 사업 수행 계획
- RFP에 명시된 일정을 반영한 상세 프로젝트 일정표
- 단계별 수행 계획 및 마일스톤
- 각 단계별 주요 산출물 및 성과물
- 리스크 관리 방안 및 대응 전략
- 품질 관리 체계

## 5. 조직 및 인력 구성
- 프로젝트 조직도 및 역할 분담
- 핵심 인력 구성 및 역할 (PM, 기술책임자, 개발자 등)
- 각 인력의 전문성 및 경험
- 발주 기관과의 협업 체계

## 6. 예산 및 제안 금액
- RFP에 명시된 예산 범위를 고려한 제안 금액
- 예산 구성 내역 (인건비, 개발비, 운영비 등)
- 가격 경쟁력 및 가치 제안
- 비용 대비 효과 분석

## 7. 기대 효과 및 성과
- RFP 목표 달성을 위한 구체적 성과 지표
- 정량적 성과 (처리 속도, 사용자 수, 시스템 안정성 등)
- 정성적 성과 (사용자 만족도, 업무 효율성 향상 등)
- ROI 분석 및 장기적 가치

## 8. 차별화 포인트
- 경쟁사 대비 우리의 핵심 경쟁 우위
- 우리의 기술력, 특허, 노하우
- 유사 사업 수행 경험 및 성공 사례
- 발주 기관에 제공할 추가 가치

작성 지침:
1. RFP 문서에 명시된 모든 구체적 정보(사업명, 예산, 일정, 요구사항 등)를 반드시 반영하세요
2. 각 섹션을 최소 3-5문단으로 상세하게 작성하세요
3. 전문적이고 설득력 있는 문체로 작성하세요
4. 구체적인 수치, 일정, 기술 명세를 포함하세요
5. 발주 기관의 요구사항을 정확히 이해하고 충족하는 내용으로 작성하세요

지금부터 제안서를 작성하세요:
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
            "당신은 정부 및 공공기관 사업 제안서 작성 전문가입니다. RFP 문서의 모든 정보를 철저히 분석하고, "
            "그 내용을 바탕으로 발주 기관에 제출할 전문적이고 구체적인 사업 제안서를 작성하세요. "
            "RFP에 명시된 사업명, 예산, 일정, 기술 요구사항 등 모든 구체적 정보를 반드시 반영하여 작성하세요. "
            "단순 요약이 아닌, 실제 제안서 형태로 최소 2000자 이상의 상세한 내용을 작성하세요."
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
            
            # Debug: Log full response object
            self.logger.debug(f"Response type: {type(response)}")
            self.logger.debug(f"Response repr: {repr(response)}")
            
            # Extract content using helper method
            proposal_text = self._extract_response_content(response)
            
            self.logger.info(f"LLM response received, length: {len(proposal_text) if proposal_text else 0} chars")
            if proposal_text and len(proposal_text) > 0:
                self.logger.debug(f"Response preview (first 200 chars): {proposal_text[:200]}")
            else:
                self.logger.error(f"Empty response! Full response object: {response}")
            
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
                proposal_text = self._extract_response_content(response)
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
            
            # Extract content using helper method
            proposal_text = self._extract_response_content(response)
            
            self.logger.info(f"LLM response received, length: {len(proposal_text) if proposal_text else 0} chars")
            if proposal_text and len(proposal_text) > 0:
                self.logger.debug(f"Response preview (first 200 chars): {proposal_text[:200]}")
            else:
                self.logger.error(f"Empty response! Full response object: {response}")
            
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
                proposal_text = self._extract_response_content(response)
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
        """Build context from chunks with metadata."""
        context_parts = []
        
        # Extract metadata from first chunk (usually contains document-level info)
        first_chunk_meta = chunks[0].get('metadata', {}) if chunks else {}
        doc_meta_info = []
        
        if first_chunk_meta.get('사업명'):
            doc_meta_info.append(f"사업명: {first_chunk_meta['사업명']}")
        if first_chunk_meta.get('공고 번호'):
            doc_meta_info.append(f"공고 번호: {first_chunk_meta['공고 번호']}")
        if first_chunk_meta.get('사업 금액'):
            doc_meta_info.append(f"사업 예산: {first_chunk_meta['사업 금액']:,}원")
        if first_chunk_meta.get('발주 기관'):
            doc_meta_info.append(f"발주 기관: {first_chunk_meta['발주 기관']}")
        if first_chunk_meta.get('입찰 참여 마감일'):
            doc_meta_info.append(f"입찰 참여 마감일: {first_chunk_meta['입찰 참여 마감일']}")
        if first_chunk_meta.get('사업 요약'):
            doc_meta_info.append(f"사업 요약: {first_chunk_meta['사업 요약']}")
        
        if doc_meta_info:
            context_parts.append("=== RFP 문서 기본 정보 ===\n" + "\n".join(doc_meta_info) + "\n")
        
        # Add chunks with more context
        for i, chunk in enumerate(chunks, 1):
            chunk_text = chunk.get('chunk_text', '')
            metadata = chunk.get('metadata', {})
            
            # Include more text (up to 800 chars) for better context
            if len(chunk_text) > 800:
                chunk_text = chunk_text[:800] + "..."
            
            section_info = ""
            if metadata.get('section_name'):
                section_info = f" [섹션: {metadata['section_name']}]"
            
            context_parts.append(f"[문서 부분 {i}{section_info}]\n{chunk_text}")
        
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
    
    def _extract_response_content(self, response):
        """Extract content from LLM response object."""
        proposal_text = None
        
        # Try multiple methods to extract content
        if hasattr(response, 'content'):
            proposal_text = response.content
        elif hasattr(response, 'text'):
            proposal_text = response.text
        elif hasattr(response, 'message'):
            if hasattr(response.message, 'content'):
                proposal_text = response.message.content
        else:
            proposal_text = str(response)
        
        # If still None or empty, try deeper inspection
        if not proposal_text:
            try:
                # Try accessing as AIMessage or similar
                if hasattr(response, '__dict__'):
                    for key, val in response.__dict__.items():
                        if isinstance(val, str) and val:
                            proposal_text = val
                            break
                        elif hasattr(val, 'content'):
                            proposal_text = val.content
                            break
            except Exception as e:
                self.logger.debug(f"Error in deep inspection: {e}")
        
        return proposal_text if proposal_text else ""
    
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
            
            # Extract content - same logic as main call
            proposal_text = self._extract_response_content(response)
            
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
                
                # Extract content - same logic as main call
                proposal_text = self._extract_response_content(response)
                
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

