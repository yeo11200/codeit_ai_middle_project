"""Evaluation dataset loader."""

from typing import List, Dict

from src.common.utils import load_jsonl
from src.common.logger import get_logger


class EvalLoader:
    """Load evaluation test sets."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def load_test_set(self, file_path: str) -> List[Dict]:
        """
        Load test set from JSONL file.
        
        Args:
            file_path: Path to JSONL file
        
        Returns:
            List of test items
        """
        try:
            test_set = load_jsonl(file_path)
            self.logger.info(f"Loaded {len(test_set)} test items from {file_path}")
            return test_set
        except Exception as e:
            self.logger.error(f"Failed to load test set: {e}")
            return []

