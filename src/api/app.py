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


class ProposalRequest(BaseModel):
    query: Optional[str] = None
    doc_id: Optional[str] = None
    top_k: Optional[int] = 30
    company_info: Optional[Dict] = None


class ProposalResponse(BaseModel):
    proposal: str
    sources: List[str]
    query: Optional[str] = None
    doc_id: Optional[str] = None
    total_chunks_used: int


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
        result = generation_agent.generate_proposal(
            query=request.query,
            doc_id=request.doc_id,
            top_k=request.top_k,
            company_info=request.company_info
        )
        
        return ProposalResponse(
            proposal=result["proposal"],
            sources=result["sources"],
            query=result.get("query"),
            doc_id=result.get("doc_id"),
            total_chunks_used=result.get("total_chunks_used", 0)
        )
    except Exception as e:
        logger.error(f"Proposal generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Proposal generation failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

