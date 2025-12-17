"""Retrieval Agent - unified search interface."""

import time
from typing import Dict, Optional

from src.retrieval.vector_search import VectorSearch
from src.retrieval.filter import MetadataFilter
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.reranker import Reranker
from src.common.logger import get_logger
from src.common.utils import normalize_text


class RetrievalAgent:
    """Main retrieval agent combining all search components."""
    
    def __init__(self, config: Dict, vector_store, embedder):
        """
        Initialize RetrievalAgent.
        
        Args:
            config: Configuration dictionary
            vector_store: VectorStore instance
            embedder: Embedder instance
        """
        self.config = config.get("retrieval", {})
        self.logger = get_logger(__name__)
        
        self.vector_search = VectorSearch(vector_store, embedder)
        self.metadata_filter = MetadataFilter()
        self.hybrid_search = HybridSearch(self.vector_search)
        self.reranker = Reranker()
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None,
        use_hybrid: bool = False,
        use_rerank: bool = True
    ) -> Dict:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional metadata filters
            use_hybrid: Whether to use hybrid search
            use_rerank: Whether to apply MMR reranking
        
        Returns:
            Dictionary with query, results, total_found, search_time
        """
        start_time = time.time()
        
        # Normalize query
        query = normalize_text(query)
        
        # Build metadata filter
        filter_dict = None
        if filters:
            filter_dict = self.metadata_filter.build_filter(**filters)
        
        # Perform search
        if use_hybrid or self.config.get("use_hybrid_search", False):
            alpha = self.config.get("hybrid_alpha", 0.7)
            results = self.hybrid_search.search(
                query=query,
                top_k=top_k * 2,  # Get more for reranking
                filters=filter_dict,
                use_bm25=self.config.get("use_bm25", False),
                alpha=alpha
            )
        else:
            results = self.vector_search.search(
                query=query,
                top_k=top_k * 2,  # Get more for reranking
                filters=filter_dict
            )
        
        # Apply reranking
        if use_rerank or self.config.get("use_rerank", True):
            lambda_param = self.config.get("mmr_lambda", 0.5)
            results = self.reranker.mmr_rerank(
                results=results,
                query=query,
                lambda_param=lambda_param,
                top_k=top_k
            )
        else:
            results = results[:top_k]
        
        # Format results
        formatted_results = self._format_results(results)
        
        search_time = time.time() - start_time
        
        return {
            "query": query,
            "results": formatted_results,
            "total_found": len(formatted_results),
            "search_time": search_time
        }
    
    def _format_results(self, results: list) -> list:
        """Format search results to standard format."""
        formatted = []
        
        for result in results:
            metadata = result.get("metadata", {})
            formatted.append({
                "chunk_id": result.get("id", ""),
                "doc_id": metadata.get("doc_id", ""),
                "chunk_text": result.get("text", ""),
                "score": result.get("score", 0.0),
                "metadata": metadata
            })
        
        return formatted

