"""Evaluation metrics calculation."""

from typing import List, Dict
import numpy as np
from rapidfuzz import fuzz

from src.common.logger import get_logger


class Metrics:
    """Calculate evaluation metrics for retrieval and generation."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    @staticmethod
    def recall_at_k(
        relevant_docs: List[str],
        retrieved_docs: List[str],
        k: int
    ) -> float:
        """
        Calculate Recall@K.
        
        Args:
            relevant_docs: List of relevant document IDs
            retrieved_docs: List of retrieved document IDs (in order)
            k: Number of top results to consider
        
        Returns:
            Recall@K score (0.0 to 1.0)
        """
        if not relevant_docs:
            return 0.0
        
        retrieved_k = set(retrieved_docs[:k])
        relevant_set = set(relevant_docs)
        
        intersection = len(retrieved_k & relevant_set)
        return intersection / len(relevant_set)
    
    @staticmethod
    def precision_at_k(
        relevant_docs: List[str],
        retrieved_docs: List[str],
        k: int
    ) -> float:
        """
        Calculate Precision@K.
        
        Args:
            relevant_docs: List of relevant document IDs
            retrieved_docs: List of retrieved document IDs
            k: Number of top results to consider
        
        Returns:
            Precision@K score (0.0 to 1.0)
        """
        if k == 0:
            return 0.0
        
        retrieved_k = set(retrieved_docs[:k])
        relevant_set = set(relevant_docs)
        
        intersection = len(retrieved_k & relevant_set)
        return intersection / k
    
    @staticmethod
    def mrr(
        relevant_docs: List[str],
        retrieved_docs: List[str]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank.
        
        Args:
            relevant_docs: List of relevant document IDs
            retrieved_docs: List of retrieved document IDs
        
        Returns:
            MRR score (0.0 to 1.0)
        """
        if not relevant_docs:
            return 0.0
        
        relevant_set = set(relevant_docs)
        
        for rank, doc_id in enumerate(retrieved_docs, 1):
            if doc_id in relevant_set:
                return 1.0 / rank
        
        return 0.0
    
    @staticmethod
    def f1_at_k(
        relevant_docs: List[str],
        retrieved_docs: List[str],
        k: int
    ) -> float:
        """
        Calculate F1@K.
        
        Args:
            relevant_docs: List of relevant document IDs
            retrieved_docs: List of retrieved document IDs
            k: Number of top results to consider
        
        Returns:
            F1@K score (0.0 to 1.0)
        """
        precision = Metrics.precision_at_k(relevant_docs, retrieved_docs, k)
        recall = Metrics.recall_at_k(relevant_docs, retrieved_docs, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def citation_recall(
        answer: str,
        sources: List[Dict],
        ground_truth_sources: List[str]
    ) -> float:
        """
        Calculate citation recall.
        
        Args:
            answer: Generated answer text
            sources: List of source documents used
            ground_truth_sources: List of ground truth source IDs
        
        Returns:
            Citation recall score (0.0 to 1.0)
        """
        if not ground_truth_sources:
            return 0.0
        
        # Extract source IDs from sources
        source_ids = [s.get("doc_id", s.get("chunk_id", "")) for s in sources]
        source_set = set(source_ids)
        gt_set = set(ground_truth_sources)
        
        intersection = len(source_set & gt_set)
        return intersection / len(gt_set) if gt_set else 0.0
    
    @staticmethod
    def answer_accuracy(
        predicted_answer: str,
        ground_truth_answer: str,
        use_llm_eval: bool = False
    ) -> Dict:
        """
        Calculate answer accuracy.
        
        Args:
            predicted_answer: Generated answer
            ground_truth_answer: Ground truth answer
            use_llm_eval: Whether to use LLM for evaluation (not implemented)
        
        Returns:
            Dictionary with score and method
        """
        if use_llm_eval:
            # LLM-based evaluation would go here
            return {"score": 0.0, "method": "llm_eval"}
        
        # Simple string similarity using rapidfuzz
        similarity = fuzz.ratio(predicted_answer.lower(), ground_truth_answer.lower()) / 100.0
        
        return {
            "score": similarity,
            "method": "string_similarity"
        }
    
    @staticmethod
    def rouge_score(
        predicted_summary: str,
        reference_summary: str
    ) -> Dict:
        """
        Calculate ROUGE scores.
        
        Args:
            predicted_summary: Generated summary
            reference_summary: Reference summary
        
        Returns:
            Dictionary with rouge-1, rouge-2, rouge-l scores
        """
        try:
            from rouge_score import rouge_scorer
            
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(reference_summary, predicted_summary)
            
            return {
                "rouge-1": scores['rouge1'].fmeasure,
                "rouge-2": scores['rouge2'].fmeasure,
                "rouge-l": scores['rougeL'].fmeasure,
            }
        except ImportError:
            # Fallback to simple similarity if rouge-score not available
            similarity = fuzz.ratio(predicted_summary, reference_summary) / 100.0
            return {
                "rouge-1": similarity,
                "rouge-2": similarity,
                "rouge-l": similarity,
            }
    
    @staticmethod
    def bleu_score(
        predicted_text: str,
        reference_text: str
    ) -> float:
        """
        Calculate BLEU score.
        
        Args:
            predicted_text: Generated text
            reference_text: Reference text
        
        Returns:
            BLEU score (0.0 to 1.0)
        """
        try:
            from nltk.translate.bleu_score import sentence_bleu
            
            pred_tokens = predicted_text.split()
            ref_tokens = reference_text.split()
            
            score = sentence_bleu([ref_tokens], pred_tokens)
            return score
        except ImportError:
            # Fallback to simple similarity
            return fuzz.ratio(predicted_text, reference_text) / 100.0
    
    @staticmethod
    def calculate_latency(timings: List[float]) -> Dict:
        """
        Calculate latency statistics.
        
        Args:
            timings: List of timing measurements
        
        Returns:
            Dictionary with mean, median, p95, p99, min, max
        """
        if not timings:
            return {
                "mean": 0.0,
                "median": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "min": 0.0,
                "max": 0.0
            }
        
        sorted_timings = sorted(timings)
        n = len(sorted_timings)
        
        return {
            "mean": np.mean(timings),
            "median": np.median(timings),
            "p95": sorted_timings[int(n * 0.95)] if n > 0 else 0.0,
            "p99": sorted_timings[int(n * 0.99)] if n > 0 else 0.0,
            "min": min(timings),
            "max": max(timings)
        }
    
    @staticmethod
    def calculate_throughput(total_requests: int, total_time: float) -> float:
        """
        Calculate throughput.
        
        Args:
            total_requests: Total number of requests
            total_time: Total time in seconds
        
        Returns:
            Throughput (requests per second)
        """
        if total_time == 0:
            return 0.0
        return total_requests / total_time

