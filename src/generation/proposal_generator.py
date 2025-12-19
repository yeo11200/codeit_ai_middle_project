"""Proposal generation module - generates proposals based on RFP documents."""

import json
from typing import Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from src.common.logger import get_logger


PROPOSAL_PROMPT = """RFP 문서를 분석하여 제안서를 작성하세요.

RFP 정보:
{context}
{conversation_context}
{previous_proposal_context}

8개 섹션으로 제안서를 작성하세요 (각 섹션 3-5문단):

1. 사업 이해 및 배경
2. 제안 개요
3. 기술 제안
4. 사업 수행 계획
5. 조직 및 인력 구성
6. 예산 및 제안 금액
7. 기대 효과 및 성과
8. 차별화 포인트
{additional_instructions}

{update_instruction}

**중요: 최소 2000자 이상 작성하세요. 지금 바로 작성하세요:**
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
            "당신은 제안서 작성 전문가입니다. RFP를 분석하여 최소 2000자 이상의 전문 제안서를 작성하세요."
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
        company_info: Optional[Dict] = None,
        additional_notes: Optional[str] = None,
        custom_sections: Optional[List[str]] = None,
        conversation_history: Optional[List[Dict]] = None,
        previous_proposal: Optional[str] = None
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
        # Limit to 3 chunks to ensure enough tokens for response
        max_context_chunks = min(top_k, 3)  # Limit to 3 chunks to save tokens
        chunks_to_use = retrieval_results["results"][:max_context_chunks]
        context = self._build_context(chunks_to_use)
        
        # Estimate tokens (rough: 1 token ≈ 2-3 Korean chars)
        estimated_input_tokens = len(context) // 2.5
        self.logger.info(f"Estimated input tokens: ~{estimated_input_tokens:.0f} tokens (context: {len(context)} chars)")
        
        # Add company info to context if provided
        if company_info:
            company_context = self._format_company_info(company_info)
            context = f"{company_context}\n\n{context}"
        
        self.logger.info(f"Context built: {len(chunks_to_use)} chunks, {len(context)} chars")
        
        # Add additional notes to context if provided
        if additional_notes:
            context = f"{context}\n\n[추가 요청사항]\n{additional_notes}"
        
        # Add custom sections to prompt if provided
        custom_sections_text = ""
        if custom_sections:
            custom_sections_text = "\n\n추가로 다음 내용도 포함하세요:\n" + "\n".join([f"- {section}" for section in custom_sections])
        
        # Build conversation context
        conversation_context = ""
        if conversation_history:
            conv_text = "\n\n[이전 대화 기록]\n"
            for msg in conversation_history:
                role = msg.get("role", "user") if isinstance(msg, dict) else (msg.role if hasattr(msg, "role") else "user")
                content = msg.get("content", "") if isinstance(msg, dict) else (msg.content if hasattr(msg, "content") else str(msg))
                role_kr = "사용자" if role == "user" else "어시스턴트"
                conv_text += f"{role_kr}: {content}\n"
            conversation_context = conv_text
        
        # Build previous proposal context
        previous_proposal_context = ""
        update_instruction = ""
        if previous_proposal:
            previous_proposal_context = f"\n\n[이전 제안서]\n{previous_proposal[:2000]}..."  # Limit to 2000 chars
            update_instruction = "\n**중요: 위의 이전 제안서와 대화 기록을 반영하여 제안서를 개선하고 업데이트하세요. 이전 내용을 유지하면서 새로운 요구사항을 반영하세요.**"
        
        # Generate proposal
        messages = self.prompt.format_messages(
            context=context,
            conversation_context=conversation_context,
            previous_proposal_context=previous_proposal_context,
            additional_instructions=custom_sections_text if custom_sections_text else "",
            update_instruction=update_instruction
        )
        
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
        analysis = self._analyze_response_quality(proposal_text, messages)
        
        if not proposal_text or not proposal_text.strip():
            self.logger.error(f"Proposal generation returned empty text after all attempts")
            proposal_text = f"""제안서 생성에 실패했습니다.

{analysis}

서버 로그를 확인하거나 다른 LLM 모델을 시도해주세요."""
        elif len(proposal_text.strip()) < 200:
            self.logger.warning(f"Proposal is very short ({len(proposal_text)} chars) but returning it anyway")
            # Add analysis to the proposal
            proposal_text = f"""{proposal_text}

---
[응답 분석]
{analysis}
---
"""
        
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
        company_info: Optional[Dict] = None,
        additional_notes: Optional[str] = None,
        custom_sections: Optional[List[str]] = None,
        conversation_history: Optional[List[Dict]] = None,
        previous_proposal: Optional[str] = None
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
        # Limit to 3 chunks to ensure enough tokens for response
        max_context_chunks = min(top_k, 3)  # Limit to 3 chunks to save tokens
        chunks_to_use = doc_chunks[:max_context_chunks]
        context = self._build_context(chunks_to_use)
        
        # Estimate tokens (rough: 1 token ≈ 2-3 Korean chars)
        estimated_input_tokens = len(context) // 2.5
        self.logger.info(f"Estimated input tokens: ~{estimated_input_tokens:.0f} tokens (context: {len(context)} chars)")
        
        # Add company info if provided
        if company_info:
            company_context = self._format_company_info(company_info)
            context = f"{company_context}\n\n{context}"
        
        self.logger.info(f"Context built: {len(chunks_to_use)} chunks, {len(context)} chars")
        
        # Add additional notes to context if provided
        if additional_notes:
            context = f"{context}\n\n[추가 요청사항]\n{additional_notes}"
        
        # Add custom sections to prompt if provided
        custom_sections_text = ""
        if custom_sections:
            custom_sections_text = "\n\n추가로 다음 내용도 포함하세요:\n" + "\n".join([f"- {section}" for section in custom_sections])
        
        # Generate proposal
        messages = self.prompt.format_messages(
            context=context,
            additional_instructions=custom_sections_text if custom_sections_text else ""
        )
        
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
        analysis = self._analyze_response_quality(proposal_text, messages)
        
        if not proposal_text or not proposal_text.strip():
            self.logger.error(f"Proposal generation returned empty text after all attempts")
            proposal_text = f"""제안서 생성에 실패했습니다.

{analysis}

서버 로그를 확인하거나 다른 LLM 모델을 시도해주세요."""
        elif len(proposal_text.strip()) < 200:
            self.logger.warning(f"Proposal is very short ({len(proposal_text)} chars) but returning it anyway")
            # Add analysis to the proposal
            proposal_text = f"""{proposal_text}

---
[응답 분석]
{analysis}
---
"""
        
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
            
            # Limit each chunk to 400 chars to save tokens for response
            if len(chunk_text) > 400:
                chunk_text = chunk_text[:400] + "..."
            
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
    
    def _analyze_response_quality(self, proposal_text: str, messages) -> str:
        """Analyze why the response might be short."""
        analysis_parts = []
        
        # Get response info
        response_length = len(proposal_text) if proposal_text else 0
        analysis_parts.append(f"생성된 응답 길이: {response_length}자")
        
        # Get LLM info
        try:
            model_name = self.llm.model_name if hasattr(self.llm, 'model_name') else str(self.llm.model)
            max_tokens = self.llm.max_tokens if hasattr(self.llm, 'max_tokens') else 'unknown'
            analysis_parts.append(f"사용된 LLM 모델: {model_name}")
            analysis_parts.append(f"설정된 max_tokens: {max_tokens}")
        except:
            analysis_parts.append(f"사용된 LLM 모델: 확인 불가")
        
        # Analyze message length and estimate tokens
        try:
            total_message_length = sum(len(str(msg)) for msg in messages)
            # Rough estimate: 1 token ≈ 2.5 Korean characters
            estimated_input_tokens = total_message_length / 2.5
            analysis_parts.append(f"입력 프롬프트 길이: 약 {total_message_length}자")
            analysis_parts.append(f"예상 입력 토큰 수: 약 {estimated_input_tokens:.0f} 토큰")
            
            # Check if input is too long
            try:
                max_tokens = self.llm.max_tokens if hasattr(self.llm, 'max_tokens') else 4000
                if isinstance(max_tokens, int):
                    # Typical model context window is 8000-16000 tokens
                    # If input + max_tokens > context window, response space is limited
                    total_needed = estimated_input_tokens + max_tokens
                    if total_needed > 8000:
                        analysis_parts.append(f"⚠️ 입력({estimated_input_tokens:.0f}) + 출력({max_tokens}) = {total_needed:.0f} 토큰으로 컨텍스트 윈도우를 초과할 수 있습니다")
            except:
                pass
        except:
            pass
        
        # Possible reasons
        reasons = []
        if response_length == 0:
            reasons.append("• LLM이 응답을 생성하지 못했습니다 (모델 접근 권한 문제 가능)")
        elif response_length < 50:
            reasons.append("• LLM이 매우 짧은 응답만 생성했습니다 (모델 제한 또는 프롬프트 문제 가능)")
        elif response_length < 200:
            reasons.append("• LLM이 짧은 응답을 생성했습니다")
            reasons.append("• max_tokens 설정이 충분하지 않을 수 있습니다")
            reasons.append("• 입력 프롬프트가 너무 길어서 응답 공간이 부족할 수 있습니다")
        
        if max_tokens != 'unknown' and isinstance(max_tokens, int) and max_tokens < 2000:
            reasons.append(f"• max_tokens({max_tokens})가 제안서 생성에 부족할 수 있습니다 (권장: 4000 이상)")
        
        if reasons:
            analysis_parts.append("\n가능한 원인:")
            analysis_parts.extend(reasons)
        
        # Response preview
        if proposal_text:
            preview = proposal_text[:200].replace('\n', ' ')
            analysis_parts.append(f"\n응답 미리보기: {preview}...")
        
        return "\n".join(analysis_parts)
    
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

