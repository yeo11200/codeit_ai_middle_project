"""FastAPI application for RFP RAG system."""

import sys
from pathlib import Path
from typing import Optional, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

from src.common.config import load_config
from src.common.logger import get_logger
from src.indexing.vector_store import VectorStore
from src.indexing.embedder import Embedder
from src.retrieval.retrieval_agent import RetrievalAgent
from src.generation.generation_agent import GenerationAgent
from src.common.llm_utils import create_llm_with_fallback


# Initialize FastAPI app
app = FastAPI(
    title="RFP RAG API",
    description="RFP 문서 검색 및 질문 답변 API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agents (initialized on startup)
config = None
retrieval_agent = None
generation_agent = None
logger = None


# Request/Response models
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10
    filters: Optional[Dict] = None
    use_hybrid: Optional[bool] = False
    use_rerank: Optional[bool] = True


class QARequest(BaseModel):
    query: str


class SummarizeRequest(BaseModel):
    doc_id: str
    top_k: Optional[int] = 20


class ExtractRequest(BaseModel):
    doc_id: str
    schema: Optional[Dict] = None


class SearchResponse(BaseModel):
    query: str
    results: List[Dict]
    total_found: int
    search_time: float


class QAResponse(BaseModel):
    answer: str
    sources: List[Dict]
    confidence: str
    query: str


class SummarizeResponse(BaseModel):
    summary: str
    key_points: List[str]
    budget: Optional[str]
    deadline: Optional[str]
    requirements: List[str]
    doc_id: str


class ExtractResponse(BaseModel):
    extracted_info: Dict
    doc_id: str


class ConversationMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ProposalRequest(BaseModel):
    query: Optional[str] = None
    doc_id: Optional[str] = None
    top_k: Optional[int] = 30
    company_info: Optional[Dict] = None
    additional_notes: Optional[str] = None  # 추가로 제안서에 포함할 내용 (예: "제안 내용에 대한 상세한 기술·일정·예산 산출근거는 추가 자료로 제출하겠습니다.")
    custom_sections: Optional[List[str]] = None  # 추가로 포함할 커스텀 섹션 목록
    conversation_history: Optional[List[ConversationMessage]] = None  # 이전 대화 기록 (채팅 빌드업용)
    previous_proposal: Optional[str] = None  # 이전에 생성된 제안서 (업데이트용)


class ProposalResponse(BaseModel):
    proposal: str
    sources: List[str]
    query: Optional[str] = None
    doc_id: Optional[str] = None
    total_chunks_used: int
    response_length: Optional[int] = None
    model_used: Optional[str] = None
    analysis: Optional[str] = None


class ProposalChatRequest(BaseModel):
    message: str  # 사용자 메시지
    conversation_history: Optional[List[ConversationMessage]] = None  # 이전 대화 기록
    query: Optional[str] = None  # RFP 검색 쿼리 (첫 메시지에서 설정)
    doc_id: Optional[str] = None  # 특정 문서 ID (첫 메시지에서 설정)
    company_info: Optional[Dict] = None
    previous_proposal: Optional[str] = None  # 이전에 생성된 제안서


class ProposalChatResponse(BaseModel):
    response: str  # 어시스턴트 응답
    is_proposal: bool  # 제안서인지 일반 답변인지
    proposal: Optional[str] = None  # 제안서 내용 (is_proposal=True일 때)
    sources: Optional[List[str]] = None
    conversation_history: List[ConversationMessage]  # 업데이트된 대화 기록


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize agents on startup."""
    global config, retrieval_agent, generation_agent, logger
    
    logger = get_logger(__name__)
    logger.info("Initializing RFP RAG API...")
    
    # Load configuration
    config_path = project_root / "config" / "local.yaml"
    config = load_config(str(config_path))
    logger.info(f"Loaded config from: {config_path}")
    
    # Initialize vector store and embedder
    vector_store = VectorStore(config["indexing"])
    embedder = Embedder(config["indexing"])
    
    # Initialize retrieval agent
    retrieval_agent = RetrievalAgent(config["retrieval"], vector_store, embedder)
    
    # Initialize LLM with increased max_tokens for proposals
    llm_max_tokens = max(config["generation"]["llm"]["max_tokens"], 4000)  # At least 4000 for proposals
    llm = create_llm_with_fallback(
        primary_model=config["generation"]["llm"]["model"],
        temperature=config["generation"]["llm"]["temperature"],
        max_tokens=llm_max_tokens,
    )
    logger.info(f"LLM initialized with max_tokens={llm_max_tokens}")
    
    # Initialize generation agent
    generation_agent = GenerationAgent(config["generation"], llm, retrieval_agent)
    
    logger.info("RFP RAG API initialized successfully!")


# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint - health check."""
    return {
        "status": "ok",
        "service": "RFP RAG API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "agents_initialized": retrieval_agent is not None and generation_agent is not None
    }


# Search endpoint
@app.post("/api/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Search for relevant documents.
    
    Args:
        request: Search request with query and options
    
    Returns:
        Search results with scores and metadata
    """
    if retrieval_agent is None:
        raise HTTPException(status_code=503, detail="Retrieval agent not initialized")
    
    try:
        results = retrieval_agent.retrieve(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters,
            use_hybrid=request.use_hybrid,
            use_rerank=request.use_rerank
        )
        
        return SearchResponse(
            query=results["query"],
            results=results["results"],
            total_found=results["total_found"],
            search_time=results["search_time"]
        )
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# Q&A endpoint
@app.post("/api/qa", response_model=QAResponse)
async def qa(request: QARequest):
    """
    Answer a question using RAG.
    
    Args:
        request: QA request with query
    
    Returns:
        Answer with sources and confidence
    """
    if generation_agent is None:
        raise HTTPException(status_code=503, detail="Generation agent not initialized")
    
    try:
        result = generation_agent.answer_question(request.query)
        
        return QAResponse(
            answer=result["answer"],
            sources=result["sources"],
            confidence=result.get("confidence", "medium"),
            query=result.get("query", request.query)
        )
    except Exception as e:
        logger.error(f"Q&A failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Q&A failed: {str(e)}")


# Summarize endpoint
@app.post("/api/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    """
    Summarize a document.
    
    Args:
        request: Summarize request with doc_id
    
    Returns:
        Document summary with key information
    """
    if generation_agent is None:
        raise HTTPException(status_code=503, detail="Generation agent not initialized")
    
    try:
        result = generation_agent.summarize_document(request.doc_id)
        
        return SummarizeResponse(
            summary=result["summary"],
            key_points=result.get("key_points", []),
            budget=result.get("budget"),
            deadline=result.get("deadline"),
            requirements=result.get("requirements", []),
            doc_id=result.get("doc_id", request.doc_id)
        )
    except Exception as e:
        logger.error(f"Summarization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")


# Extract endpoint
@app.post("/api/extract", response_model=ExtractResponse)
async def extract(request: ExtractRequest):
    """
    Extract structured information from a document.
    
    Args:
        request: Extract request with doc_id and optional schema
    
    Returns:
        Extracted structured information
    """
    if generation_agent is None:
        raise HTTPException(status_code=503, detail="Generation agent not initialized")
    
    try:
        result = generation_agent.extract_info(
            doc_id=request.doc_id,
            schema=request.schema
        )
        
        return ExtractResponse(
            extracted_info=result,
            doc_id=request.doc_id
        )
    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


# Generate Proposal endpoint
@app.post("/api/generate-proposal", response_model=ProposalResponse)
async def generate_proposal(request: ProposalRequest):
    """
    Generate a proposal based on RFP documents.
    
    Args:
        request: Proposal request with query or doc_id
    
    Returns:
        Generated proposal with sources
    """
    if generation_agent is None:
        raise HTTPException(status_code=503, detail="Generation agent not initialized")
    
    if not request.query and not request.doc_id:
        raise HTTPException(
            status_code=400,
            detail="Either 'query' or 'doc_id' must be provided"
        )
    
    try:
        # Convert conversation_history from Pydantic models to dicts
        conv_history = None
        if request.conversation_history:
            conv_history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.conversation_history
            ]
        
        result = generation_agent.generate_proposal(
            query=request.query,
            doc_id=request.doc_id,
            top_k=request.top_k,
            company_info=request.company_info,
            additional_notes=request.additional_notes,
            custom_sections=request.custom_sections,
            conversation_history=conv_history,
            previous_proposal=request.previous_proposal
        )
        
        # Extract analysis if present
        proposal_text = result["proposal"]
        analysis = None
        if "---\n[응답 분석]" in proposal_text:
            parts = proposal_text.split("---\n[응답 분석]\n")
            if len(parts) > 1:
                analysis_part = parts[1].split("\n---")[0]
                analysis = analysis_part.strip()
                # Remove analysis from proposal for cleaner output
                proposal_text = parts[0].strip()
        
        # Get model info
        model_used = None
        try:
            if generation_agent.proposal_generator.llm:
                model_used = generation_agent.proposal_generator.llm.model_name if hasattr(generation_agent.proposal_generator.llm, 'model_name') else str(generation_agent.proposal_generator.llm.model)
        except:
            pass
        
        return ProposalResponse(
            proposal=proposal_text,
            sources=result["sources"],
            query=result.get("query"),
            doc_id=result.get("doc_id"),
            total_chunks_used=result.get("total_chunks_used", 0),
            response_length=len(proposal_text),
            model_used=model_used,
            analysis=analysis
        )
    except Exception as e:
        logger.error(f"Proposal generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Proposal generation failed: {str(e)}"
        )


# Proposal Chat endpoint (대화형 제안서 빌드업)
@app.post("/api/proposal-chat", response_model=ProposalChatResponse)
async def proposal_chat(request: ProposalChatRequest):
    """
    대화형 제안서 빌드업 채팅 엔드포인트.
    
    사용자와 대화를 주고받으며 점진적으로 제안서를 빌드업합니다.
    - 일반 질문: 친절하게 답변
    - 제안서 생성 요청: 제안서 생성
    
    Args:
        request: 채팅 요청 (메시지, 대화 히스토리 등)
    
    Returns:
        응답 및 업데이트된 대화 히스토리
    """
    if generation_agent is None:
        raise HTTPException(status_code=503, detail="Generation agent not initialized")
    
    try:
        # Convert conversation history
        conv_history = []
        if request.conversation_history:
            conv_history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.conversation_history
            ]
        
        # Add current user message to history
        user_message = {"role": "user", "content": request.message}
        conv_history.append(user_message)
        
        # Detect intent: proposal generation request or general question
        message_lower = request.message.lower()
        proposal_keywords = [
            "제안서", "제안서 작성", "제안서 만들어", "제안서 생성", 
            "제안서 작성해", "제안서 만들어줘", "제안서 생성해줘",
            "제안서 완성", "제안서 최종", "제안서 최종본"
        ]
        is_proposal_request = any(keyword in message_lower for keyword in proposal_keywords)
        
        # If it's a proposal request, generate proposal
        if is_proposal_request:
            # Use query or doc_id from request or conversation history
            query = request.query
            doc_id = request.doc_id
            
            # If not provided, try to extract from conversation
            if not query and not doc_id:
                # Look for query in conversation history
                for msg in conv_history:
                    if msg["role"] == "user" and len(msg["content"]) > 10:
                        # Use first substantial user message as query
                        query = msg["content"]
                        break
            
            if not query and not doc_id:
                # Default query
                query = "사업"
            
            # Generate proposal
            result = generation_agent.generate_proposal(
                query=query,
                doc_id=doc_id,
                top_k=10,
                company_info=request.company_info,
                conversation_history=conv_history[:-1],  # Exclude current message
                previous_proposal=request.previous_proposal
            )
            
            proposal_text = result["proposal"]
            
            # Create friendly response
            response_text = f"""네, 제안서를 작성해드렸습니다!

{proposal_text}

추가로 수정하거나 보완할 부분이 있으시면 말씀해주세요."""
            
            # Update conversation history with assistant response
            conv_history.append({"role": "assistant", "content": response_text})
            
            return ProposalChatResponse(
                response=response_text,
                is_proposal=True,
                proposal=proposal_text,
                sources=result.get("sources", []),
                conversation_history=[
                    ConversationMessage(role=msg["role"], content=msg["content"])
                    for msg in conv_history
                ]
            )
        
        # Otherwise, answer as a friendly assistant
        else:
            # First, try to answer using RAG
            try:
                # Use query or doc_id if available
                search_query = request.query or request.message
                
                # Answer question using RAG
                qa_result = generation_agent.answer_question(search_query)
                answer = qa_result.get("answer", "")
                
                # Make response more friendly and conversational
                if "뭐" in request.message or "물어" in request.message or "질문" in request.message:
                    friendly_response = f"""네, 물론입니다! {search_query} 관련해서 무엇이든 물어보세요.

제안서 작성에 도움이 되는 정보를 찾아드릴 수 있습니다. 예를 들어:
- 사업 개요 및 배경
- 기술 요구사항
- 일정 및 예산
- 제안서 작성 방법

어떤 부분이 궁금하신가요?"""
                else:
                    # Use RAG answer but make it friendly
                    if answer:
                        friendly_response = f"""네, {search_query}에 대해 찾아본 내용입니다:

{answer}

추가로 궁금한 점이 있으시면 언제든 말씀해주세요. 제안서 작성이 필요하시면 "제안서 작성해줘" 또는 "제안서 만들어줘"라고 말씀해주세요."""
                    else:
                        friendly_response = f"""네, {search_query}에 대해 도와드리겠습니다.

제안서 작성에 필요한 정보를 찾아드릴 수 있습니다. 구체적으로 어떤 부분이 궁금하신가요?

또는 "제안서 작성해줘"라고 말씀하시면 바로 제안서를 작성해드릴 수 있습니다."""
            except Exception as e:
                logger.warning(f"RAG answer failed, using default friendly response: {e}")
                friendly_response = f"""안녕하세요! {request.message}에 대해 도와드리겠습니다.

제안서 작성에 도움이 되는 정보를 찾아드릴 수 있습니다. 구체적으로 어떤 부분이 궁금하신가요?

또는 "제안서 작성해줘"라고 말씀하시면 바로 제안서를 작성해드릴 수 있습니다."""
            
            # Update conversation history
            conv_history.append({"role": "assistant", "content": friendly_response})
            
            return ProposalChatResponse(
                response=friendly_response,
                is_proposal=False,
                proposal=None,
                sources=None,
                conversation_history=[
                    ConversationMessage(role=msg["role"], content=msg["content"])
                    for msg in conv_history
                ]
            )
    
    except Exception as e:
        logger.error(f"Proposal chat failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Proposal chat failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

