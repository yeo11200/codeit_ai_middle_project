"""Q&A chain for question answering."""

from typing import Dict, List, Optional

from src.generation.rag_chain import RAGChain
from src.common.logger import get_logger


class QAChain:
    """Question answering chain."""
    
    def __init__(self, llm, retrieval_agent, config: Dict):
        """
        Initialize QA chain.
        
        Args:
            llm: LangChain LLM instance
            retrieval_agent: RetrievalAgent instance
            config: Configuration dictionary
        """
        self.rag_chain = RAGChain(llm, retrieval_agent, config)
        self.logger = get_logger(__name__)
    
    def answer(
        self,
        query: str,
        retrieval_results: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Generate answer for a question.
        
        Args:
            query: User question
            retrieval_results: Optional pre-retrieved results
        
        Returns:
            Dictionary with answer, sources, confidence, query
        """
        # Use RAG chain to generate answer
        result = self.rag_chain.generate(query)
        
        # Assess confidence based on retrieval scores
        sources = result.get("sources", [])
        confidence = self._assess_confidence(sources)
        
        return {
            "answer": result["answer"],
            "sources": sources,
            "confidence": confidence,
            "query": query
        }
    
    def _assess_confidence(self, sources: List[Dict]) -> str:
        """
        Assess confidence level based on retrieval quality.
        
        Args:
            sources: List of source documents with scores
        
        Returns:
            Confidence level: "high", "medium", or "low"
        """
        if not sources:
            return "low"
        
        # Check average score
        scores = [s.get("score", 0.0) for s in sources]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        if avg_score >= 0.8:
            return "high"
        elif avg_score >= 0.5:
            return "medium"
        else:
            return "low"

