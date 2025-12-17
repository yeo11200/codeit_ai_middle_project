"""Reranking module using MMR (Maximal Marginal Relevance)."""

from typing import List, Dict
import numpy as np

from src.common.logger import get_logger


class Reranker:
    """Rerank search results using MMR algorithm."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def mmr_rerank(
        self,
        results: List[Dict],
        query: str,
        lambda_param: float = 0.5,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Rerank results using MMR (Maximal Marginal Relevance).
        
        Args:
            results: Initial search results with 'score' field
            query: Query text (not used in basic implementation)
            lambda_param: Balance between relevance (0) and diversity (1)
            top_k: Number of results to return
        
        Returns:
            Reranked results
        """
        if not results or len(results) <= 1:
            return results[:top_k]
        
        # Simple MMR: select results that maximize relevance while minimizing similarity to already selected
        selected: List[Dict] = []
        remaining = results.copy()
        
        # First result: highest score
        if remaining:
            remaining.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            selected.append(remaining.pop(0))
        
        # Select remaining results
        while remaining and len(selected) < top_k:
            best_idx = 0
            best_score = float("-inf")
            
            for i, candidate in enumerate(remaining):
                # Relevance score
                relevance = candidate.get("score", 0.0)
                
                # Max similarity to already selected (for diversity penalty)
                max_similarity = 0.0
                if selected:
                    # Simple similarity: use text overlap or score similarity
                    # In a full implementation, we'd compute embedding similarity
                    candidate_text = candidate.get("text", "").lower()
                    for sel in selected:
                        sel_text = sel.get("text", "").lower()
                        # Simple word overlap as similarity proxy
                        candidate_words = set(candidate_text.split())
                        sel_words = set(sel_text.split())
                        if candidate_words and sel_words:
                            overlap = len(candidate_words & sel_words) / len(candidate_words | sel_words)
                            max_similarity = max(max_similarity, overlap)
                
                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            selected.append(remaining.pop(best_idx))
        
        return selected

