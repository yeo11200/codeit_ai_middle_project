"""Evaluation Agent - system performance evaluation."""

import time
from typing import Dict, List

from tqdm import tqdm

from src.eval.metrics import Metrics
from src.eval.eval_loader import EvalLoader
from src.common.logger import get_logger


class EvalAgent:
    """Agent for evaluating system performance."""
    
    def __init__(self, config: Dict, retrieval_agent, generation_agent):
        """
        Initialize EvalAgent.
        
        Args:
            config: Configuration dictionary
            retrieval_agent: RetrievalAgent instance
            generation_agent: GenerationAgent instance
        """
        self.config = config
        self.retrieval_agent = retrieval_agent
        self.generation_agent = generation_agent
        self.metrics = Metrics()
        self.eval_loader = EvalLoader()
        self.logger = get_logger(__name__)
    
    def evaluate_retrieval(
        self,
        test_set_path: str,
        k_values: List[int] = [1, 5, 10]
    ) -> Dict:
        """
        Evaluate retrieval performance.
        
        Args:
            test_set_path: Path to test set JSONL
            k_values: List of K values to evaluate
        
        Returns:
            Dictionary with retrieval metrics
        """
        test_set = self.eval_loader.load_test_set(test_set_path)
        
        if not test_set:
            self.logger.warning("No test set loaded")
            return {}
        
        results = {
            "recall_at_k": {},
            "precision_at_k": {},
            "mrr": 0.0,
            "f1_at_k": {}
        }
        
        mrr_scores = []
        
        for k in k_values:
            recall_scores = []
            precision_scores = []
            f1_scores = []
            
            for item in tqdm(test_set, desc=f"Evaluating retrieval @{k}"):
                query = item.get("query", "")
                relevant_docs = item.get("relevant_doc_ids", [])
                
                if not query:
                    continue
                
                # Perform retrieval
                try:
                    search_results = self.retrieval_agent.retrieve(
                        query,
                        top_k=max(k_values)
                    )
                    retrieved_docs = [
                        r.get("doc_id", r.get("chunk_id", ""))
                        for r in search_results["results"]
                    ]
                    
                    # Calculate metrics
                    recall = self.metrics.recall_at_k(relevant_docs, retrieved_docs, k)
                    precision = self.metrics.precision_at_k(relevant_docs, retrieved_docs, k)
                    f1 = self.metrics.f1_at_k(relevant_docs, retrieved_docs, k)
                    
                    recall_scores.append(recall)
                    precision_scores.append(precision)
                    f1_scores.append(f1)
                    
                    # MRR (only for max k)
                    if k == max(k_values):
                        mrr = self.metrics.mrr(relevant_docs, retrieved_docs)
                        mrr_scores.append(mrr)
                
                except Exception as e:
                    self.logger.warning(f"Failed to evaluate query '{query}': {e}")
                    continue
            
            if recall_scores:
                results["recall_at_k"][k] = sum(recall_scores) / len(recall_scores)
                results["precision_at_k"][k] = sum(precision_scores) / len(precision_scores)
                results["f1_at_k"][k] = sum(f1_scores) / len(f1_scores)
        
        if mrr_scores:
            results["mrr"] = sum(mrr_scores) / len(mrr_scores)
        
        results["total_queries"] = len(test_set)
        
        return results
    
    def evaluate_generation(self, test_set_path: str) -> Dict:
        """
        Evaluate generation quality.
        
        Args:
            test_set_path: Path to test set JSONL
        
        Returns:
            Dictionary with generation metrics
        """
        test_set = self.eval_loader.load_test_set(test_set_path)
        
        if not test_set:
            return {}
        
        accuracy_scores = []
        citation_recalls = []
        confidences = []
        
        for item in tqdm(test_set, desc="Evaluating generation"):
            query = item.get("query", "")
            ground_truth = item.get("ground_truth_answer", "")
            gt_sources = item.get("evidence_chunks", [])
            
            if not query:
                continue
            
            try:
                # Generate answer
                answer_result = self.generation_agent.answer_question(query)
                predicted_answer = answer_result.get("answer", "")
                sources = answer_result.get("sources", [])
                confidence = answer_result.get("confidence", "low")
                
                # Calculate accuracy
                accuracy = self.metrics.answer_accuracy(predicted_answer, ground_truth)
                accuracy_scores.append(accuracy["score"])
                
                # Citation recall
                citation_recall = self.metrics.citation_recall(
                    predicted_answer,
                    sources,
                    gt_sources
                )
                citation_recalls.append(citation_recall)
                
                # Confidence mapping
                conf_map = {"high": 1.0, "medium": 0.5, "low": 0.0}
                confidences.append(conf_map.get(confidence, 0.0))
            
            except Exception as e:
                self.logger.warning(f"Failed to evaluate generation for '{query}': {e}")
                continue
        
        return {
            "answer_accuracy": sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.0,
            "citation_recall": sum(citation_recalls) / len(citation_recalls) if citation_recalls else 0.0,
            "average_confidence": "medium",  # Simplified
            "total_queries": len(test_set)
        }
    
    def evaluate_performance(
        self,
        test_set_path: str,
        num_queries: int = 100
    ) -> Dict:
        """
        Evaluate system performance (latency, throughput).
        
        Args:
            test_set_path: Path to test set JSONL
            num_queries: Number of queries to test
        
        Returns:
            Dictionary with performance metrics
        """
        test_set = self.eval_loader.load_test_set(test_set_path)
        
        if not test_set:
            return {}
        
        # Sample queries
        test_queries = test_set[:num_queries]
        
        retrieval_timings = []
        generation_timings = []
        
        start_time = time.time()
        
        for item in tqdm(test_queries, desc="Evaluating performance"):
            query = item.get("query", "")
            if not query:
                continue
            
            try:
                # Measure retrieval time
                ret_start = time.time()
                search_results = self.retrieval_agent.retrieve(query, top_k=10)
                retrieval_timings.append(time.time() - ret_start)
                
                # Measure generation time
                gen_start = time.time()
                self.generation_agent.answer_question(query)
                generation_timings.append(time.time() - gen_start)
            
            except Exception as e:
                self.logger.warning(f"Performance test failed for '{query}': {e}")
                continue
        
        total_time = time.time() - start_time
        
        return {
            "retrieval_latency": self.metrics.calculate_latency(retrieval_timings),
            "generation_latency": self.metrics.calculate_latency(generation_timings),
            "throughput": self.metrics.calculate_throughput(len(test_queries), total_time),
            "total_time": total_time
        }
    
    def run_full_evaluation(self, test_set_path: str) -> Dict:
        """
        Run full evaluation (retrieval + generation + performance).
        
        Args:
            test_set_path: Path to test set JSONL
        
        Returns:
            Dictionary with all evaluation results
        """
        self.logger.info("Starting full evaluation...")
        
        k_values = self.config.get("k_values", [1, 5, 10, 20])
        
        # Evaluate retrieval
        self.logger.info("Evaluating retrieval...")
        retrieval_metrics = self.evaluate_retrieval(test_set_path, k_values=k_values)
        
        # Evaluate generation
        self.logger.info("Evaluating generation...")
        generation_metrics = self.evaluate_generation(test_set_path)
        
        # Evaluate performance
        self.logger.info("Evaluating performance...")
        num_perf_queries = self.config.get("performance_test_queries", 100)
        performance_metrics = self.evaluate_performance(test_set_path, num_queries=num_perf_queries)
        
        # Combine results
        results = {
            "retrieval_metrics": retrieval_metrics,
            "generation_metrics": generation_metrics,
            "performance_metrics": performance_metrics,
            "test_set_size": len(self.eval_loader.load_test_set(test_set_path))
        }
        
        return results

