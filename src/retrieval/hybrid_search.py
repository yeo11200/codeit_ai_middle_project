"""Hybrid search combining vector and keyword search."""

from typing import List, Dict, Optional

from src.common.logger import get_logger


class HybridSearch:
    """Hybrid search combining vector and BM25 (optional)."""
    
    def __init__(self, vector_search, bm25_index=None):
        """
        Initialize HybridSearch.
        
        Args:
            vector_search: VectorSearch instance
            bm25_index: Optional BM25 index (not implemented in basic version)
        """
        self.vector_search = vector_search
        self.bm25_index = bm25_index
        self.logger = get_logger(__name__)
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None,
        use_bm25: bool = False,
        alpha: float = 0.7
    ) -> List[Dict]:
        """
        Perform hybrid search.
        
        Args:
            query: Query text
            top_k: Number of results
            filters: Metadata filters
            use_bm25: Whether to use BM25 (not implemented yet)
            alpha: Weight for vector score (1-alpha for BM25)
        
        Returns:
            List of search results
        """
        # For now, just use vector search
        # BM25 integration can be added later
        if use_bm25 and self.bm25_index is None:
            self.logger.warning("BM25 index not available, using vector search only")
        
        # Perform vector search
        results = self.vector_search.search(
            query=query,
            top_k=top_k * 2,  # Get more results for reranking
            filters=filters
        )
        
        # Normalize scores to 0-1 range
        if results:
            max_score = max(r["score"] for r in results) if results else 1.0
            min_score = min(r["score"] for r in results) if results else 0.0
            score_range = max_score - min_score if max_score > min_score else 1.0
            
            for result in results:
                # Normalize to 0-1
                normalized = (result["score"] - min_score) / score_range if score_range > 0 else result["score"]
                result["score"] = normalized
        
        return results[:top_k]

