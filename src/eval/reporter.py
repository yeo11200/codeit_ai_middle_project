"""Evaluation report generator."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict

from src.common.utils import ensure_dir, save_json
from src.common.logger import get_logger


class Reporter:
    """Generate evaluation reports."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def generate_report(self, eval_results: Dict, output_path: str) -> None:
        """
        Generate evaluation report in JSON format.
        
        Args:
            eval_results: Evaluation results dictionary
            output_path: Path to save report
        """
        # Add evaluation date
        report = {
            "evaluation_date": datetime.now().isoformat(),
            **eval_results
        }
        
        # Save report
        ensure_dir(str(Path(output_path).parent))
        save_json(report, output_path)
        
        self.logger.info(f"Evaluation report saved to: {output_path}")
    
    def generate_summary(self, eval_results: Dict) -> str:
        """
        Generate text summary of evaluation results.
        
        Args:
            eval_results: Evaluation results dictionary
        
        Returns:
            Text summary string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("Evaluation Summary")
        lines.append("=" * 60)
        
        # Retrieval metrics
        if "retrieval_metrics" in eval_results:
            rm = eval_results["retrieval_metrics"]
            lines.append("\n[Retrieval Metrics]")
            if "recall_at_k" in rm:
                lines.append(f"Recall@10: {rm['recall_at_k'].get(10, 0.0):.3f}")
            if "precision_at_k" in rm:
                lines.append(f"Precision@10: {rm['precision_at_k'].get(10, 0.0):.3f}")
            if "mrr" in rm:
                lines.append(f"MRR: {rm['mrr']:.3f}")
        
        # Generation metrics
        if "generation_metrics" in eval_results:
            gm = eval_results["generation_metrics"]
            lines.append("\n[Generation Metrics]")
            if "answer_accuracy" in gm:
                lines.append(f"Answer Accuracy: {gm['answer_accuracy']:.3f}")
            if "citation_recall" in gm:
                lines.append(f"Citation Recall: {gm['citation_recall']:.3f}")
        
        # Performance metrics
        if "performance_metrics" in eval_results:
            pm = eval_results["performance_metrics"]
            lines.append("\n[Performance Metrics]")
            if "retrieval_latency" in pm:
                rl = pm["retrieval_latency"]
                lines.append(f"Retrieval Latency (mean): {rl.get('mean', 0.0):.3f}s")
            if "generation_latency" in pm:
                gl = pm["generation_latency"]
                lines.append(f"Generation Latency (mean): {gl.get('mean', 0.0):.3f}s")
            if "throughput" in pm:
                lines.append(f"Throughput: {pm['throughput']:.2f} req/s")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)

