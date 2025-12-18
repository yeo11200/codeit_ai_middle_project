"""Proposal generation module - generates proposals based on RFP documents."""

import json
from typing import Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from src.common.logger import get_logger


PROPOSAL_PROMPT = """RFP 문서를 분석하여 전문 제안서를 작성하세요.

RFP 정보:
{context}

다음 8개 섹션으로 제안서를 작성하세요. 각 섹션을 3-5문단으로 상세히 작성하세요:

## 1. 사업 이해 및 배경
RFP에 명시된 사업 목적, 배경, 필요성을 설명하세요.

## 2. 제안 개요  
우리의 핵심 가치 제안과 접근 방식을 설명하세요.

## 3. 기술 제안
RFP 기술 요구사항을 충족하는 시스템 아키텍처, 기술 스택, 핵심 기능을 제안하세요.

## 4. 사업 수행 계획
RFP 일정을 반영한 프로젝트 일정표, 단계별 수행 계획, 마일스톤을 제시하세요.

## 5. 조직 및 인력 구성
프로젝트 조직도, 핵심 인력 구성 및 역할, 전문성을 설명하세요.

## 6. 예산 및 제안 금액
RFP 예산을 고려한 제안 금액, 예산 구성 내역, 가격 경쟁력을 제시하세요.

## 7. 기대 효과 및 성과
정량적/정성적 성과 지표, ROI 분석을 제시하세요.

## 8. 차별화 포인트
경쟁 우위, 기술력, 유사 사업 경험을 설명하세요.

**반드시 8개 섹션을 모두 포함하여 최소 2000자 이상의 전문 제안서를 작성하세요. 지금 바로 작성하세요:**
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
            "당신은 전문 제안서 작성 전문가입니다. RFP 문서를 분석하여 발주 기관에 제출할 전문 제안서를 작성하세요. "
            "RFP의 사업명, 예산, 일정, 요구사항을 반영하여 최소 2000자 이상의 상세한 제안서를 작성하세요. "
            "8개 섹션을 모두 포함하여 지금 바로 작성하세요."
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
        # Limit to 5 chunks for faster response
        max_context_chunks = min(top_k, 5)  # Limit to 5 chunks for speed
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
        self.logger.info(f"LLM model: {self.llm.model_name if hasattr(self.llm, 'model_name') else 'unknown'}")
        self.logger.info(f"LLM max_tokens: {self.llm.max_tokens if hasattr(self.llm, 'max_tokens') else 'unknown'}")
        
        try:
            # Ensure max_tokens is set correctly
            if hasattr(self.llm, 'max_tokens') and self.llm.max_tokens < 3000:
                self.logger.warning(f"LLM max_tokens ({self.llm.max_tokens}) is too low for proposals. Using 4000.")
                # Create new LLM with higher max_tokens
                from langchain_openai import ChatOpenAI
                import os
                api_key = os.getenv("OPENAI_API_KEY")
                self.llm = ChatOpenAI(
                    model=self.llm.model_name if hasattr(self.llm, 'model_name') else str(self.llm.model),
                    temperature=self.llm.temperature if hasattr(self.llm, 'temperature') else 0.2,
                    max_tokens=4000,
                    api_key=api_key
                )
            
            response = self.llm.invoke(messages)
            
            # Debug: Log full response object
            self.logger.debug(f"Response type: {type(response)}")
            self.logger.debug(f"Response repr: {repr(response)[:500]}")
            
            # Extract content using helper method
            proposal_text = self._extract_response_content(response)
            
            self.logger.info(f"LLM response received, length: {len(proposal_text) if proposal_text else 0} chars")
            if proposal_text and len(proposal_text) > 0:
                self.logger.debug(f"Response preview (first 500 chars): {proposal_text[:500]}")
                if len(proposal_text) < 100:
                    self.logger.error(f"Response too short! Only {len(proposal_text)} chars. Full response: {proposal_text}")
            else:
                self.logger.error(f"Empty response! Full response object: {response}")
            
            # Check if response is too short - retry with increased max_tokens
            # Lower threshold to 200 chars to be more lenient
            if not proposal_text or len(proposal_text.strip()) < 200:
                self.logger.warning(f"LLM returned very short proposal ({len(proposal_text) if proposal_text else 0} chars). Retrying with increased max_tokens...")
                retry_result = self._retry_with_increased_tokens(messages)
                # If retry returns error message, use original response if it exists
                if retry_result and len(retry_result) > 100 and "모든 LLM 모델" in retry_result:
                    if proposal_text and len(proposal_text.strip()) > 0:
                        self.logger.warning(f"Retry failed, using original response ({len(proposal_text)} chars)")
                        proposal_text = proposal_text
                    else:
                        proposal_text = retry_result
                else:
                    proposal_text = retry_result
                
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
        
        # Final check - accept any response, even if short
        if not proposal_text or not proposal_text.strip():
            self.logger.error(f"Proposal generation returned empty text after all attempts")
            proposal_text = "제안서 생성에 실패했습니다. 서버 로그를 확인하거나 다른 LLM 모델을 시도해주세요."
        elif len(proposal_text.strip()) < 200:
            self.logger.warning(f"Proposal is very short ({len(proposal_text)} chars) but returning it anyway")
            # Add a note to the proposal
            proposal_text = f"""{proposal_text}

[주의: 이 제안서는 매우 짧게 생성되었습니다. LLM 모델 설정을 확인하거나 다른 모델을 시도해주세요.]"""
        
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
        # Limit to 5 chunks for faster response
        max_context_chunks = min(top_k, 5)  # Limit to 5 chunks for speed
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
        
        # Final check - accept any response, even if short
        if not proposal_text or not proposal_text.strip():
            self.logger.error(f"Proposal generation returned empty text after all attempts")
            proposal_text = "제안서 생성에 실패했습니다. 서버 로그를 확인하거나 다른 LLM 모델을 시도해주세요."
        elif len(proposal_text.strip()) < 200:
            self.logger.warning(f"Proposal is very short ({len(proposal_text)} chars) but returning it anyway")
            # Add a note to the proposal
            proposal_text = f"""{proposal_text}

[주의: 이 제안서는 매우 짧게 생성되었습니다. LLM 모델 설정을 확인하거나 다른 모델을 시도해주세요.]"""
        
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
            
            # Limit each chunk to 600 chars for faster processing
            if len(chunk_text) > 600:
                chunk_text = chunk_text[:600] + "..."
            
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
        
        # Log what we got
        if proposal_text:
            self.logger.debug(f"Extracted content: {len(proposal_text)} chars, preview: {proposal_text[:100]}")
        else:
            self.logger.error(f"Failed to extract content from response: {type(response)}, dir: {dir(response)}")
        
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
            
            self.logger.info(f"Retry response length: {len(proposal_text) if proposal_text else 0} chars")
            
            if proposal_text and len(proposal_text.strip()) >= 500:
                # Update self.llm for future calls
                self.llm = retry_llm
                self.logger.info(f"Successfully generated proposal with increased max_tokens ({len(proposal_text)} chars)")
                return proposal_text
            else:
                # Still too short, try fallback models
                self.logger.warning(f"Still too short after increasing tokens ({len(proposal_text) if proposal_text else 0} chars), trying fallback models...")
                fallback_result = self._try_fallback_llm(messages, "Short response after token increase")
                # If fallback also returns error message, return the original short response
                if fallback_result and len(fallback_result) > 100 and "모든 LLM 모델" in fallback_result:
                    # Fallback failed, return original short response
                    self.logger.warning(f"Fallback also failed, returning original short response ({len(proposal_text)} chars)")
                    return proposal_text if proposal_text else "제안서 생성에 실패했습니다."
                return fallback_result
                
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
                
                self.logger.info(f"Fallback model '{fallback_model}' response length: {len(proposal_text) if proposal_text else 0} chars")
                
                if proposal_text and proposal_text.strip():
                    # Accept even short responses from fallback models
                    # Update self.llm for future calls
                    self.llm = fallback_llm
                    self.logger.info(f"Successfully switched to fallback model: {fallback_model} (response: {len(proposal_text)} chars)")
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
        
        # If all fallback models failed, return a helpful error message instead of raising
        error_msg = (
            f"모든 LLM 모델이 짧은 응답을 반환했습니다. "
            f"원본 에러: {original_error}. "
            f"시도한 모델: {LLM_FALLBACK_MODELS}. "
            "config/local.yaml에서 사용 가능한 다른 모델로 변경해주세요."
        )
        self.logger.error(error_msg)
        return error_msg

