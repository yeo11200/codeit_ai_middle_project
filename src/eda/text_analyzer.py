"""Text quality analyzer for EDA."""

import re
from typing import Dict, List
from collections import Counter
from src.common.logger import get_logger
from src.eda.data_loader import DataLoader


class TextAnalyzer:
    """Analyze text quality and patterns."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.data_loader = DataLoader()
    
    def analyze_text_lengths(self, documents: List[Dict]) -> Dict:
        """
        Analyze text lengths.
        
        Args:
            documents: List of document dictionaries
        
        Returns:
            Dictionary with text length statistics
        """
        if not documents:
            return {}
        
        lengths = []
        lengths_by_type = {}
        
        for doc in documents:
            text = doc.get("text", "")
            text_len = len(text) if text else 0
            lengths.append(text_len)
            
            # Get file type
            file_path = doc.get("file_path", "")
            file_type = self.data_loader.get_file_type_from_path(file_path)
            
            if file_type not in lengths_by_type:
                lengths_by_type[file_type] = []
            lengths_by_type[file_type].append(text_len)
        
        if not lengths:
            return {}
        
        stats = {
            "mean": float(sum(lengths) / len(lengths)),
            "median": float(sorted(lengths)[len(lengths) // 2]),
            "min": int(min(lengths)),
            "max": int(max(lengths)),
            "std": float((sum((x - sum(lengths) / len(lengths))**2 for x in lengths) / len(lengths))**0.5),
        }
        
        # Very short texts (< 100 chars)
        very_short = sum(1 for l in lengths if l < 100)
        
        # Very long texts (> 100,000 chars)
        very_long = sum(1 for l in lengths if l > 100_000)
        
        # Average by file type
        avg_by_type = {
            file_type: float(sum(lens) / len(lens)) if lens else 0.0
            for file_type, lens in lengths_by_type.items()
        }
        
        return {
            "statistics": stats,
            "very_short_count": very_short,
            "very_long_count": very_long,
            "average_by_file_type": avg_by_type,
        }
    
    def analyze_text_quality(self, documents: List[Dict]) -> Dict:
        """
        Analyze text quality.
        
        Args:
            documents: List of document dictionaries
        
        Returns:
            Dictionary with quality metrics
        """
        if not documents:
            return {}
        
        total = len(documents)
        empty_count = 0
        failed_count = 0
        ocr_needed_count = 0
        special_char_ratio_sum = 0
        
        for doc in documents:
            text = doc.get("text", "")
            status = doc.get("status", "")
            
            if not text or len(text.strip()) == 0:
                empty_count += 1
            
            if status == "failed":
                failed_count += 1
            
            # Check if OCR might be needed (very short text but file exists)
            if text and len(text.strip()) < 100:
                file_path = doc.get("file_path", "")
                if file_path:
                    ocr_needed_count += 1
            
            # Calculate special character ratio
            if text:
                special_chars = len(re.findall(r'[^\w\s가-힣]', text))
                special_char_ratio_sum += special_chars / len(text) if len(text) > 0 else 0
        
        return {
            "empty_text_ratio": float(empty_count / total) if total > 0 else 0.0,
            "parsing_failed_ratio": float(failed_count / total) if total > 0 else 0.0,
            "ocr_needed_ratio": float(ocr_needed_count / total) if total > 0 else 0.0,
            "avg_special_char_ratio": float(special_char_ratio_sum / total) if total > 0 else 0.0,
        }
    
    def analyze_content_patterns(self, documents: List[Dict]) -> Dict:
        """
        Analyze content patterns.
        
        Args:
            documents: List of document dictionaries
        
        Returns:
            Dictionary with content patterns
        """
        if not documents:
            return {}
        
        # Collect all text
        all_text = " ".join([doc.get("text", "") for doc in documents if doc.get("text")])
        
        if not all_text:
            return {}
        
        # Extract Korean words (2+ characters)
        korean_words = re.findall(r'[가-힣]{2,}', all_text)
        word_counter = Counter(korean_words)
        top_keywords = dict(word_counter.most_common(50))
        
        # Section title patterns
        section_patterns = {
            "numbered_sections": len(re.findall(r'^\d+\.', all_text, re.MULTILINE)),
            "korean_sections": len(re.findall(r'제\d+[장절]', all_text)),
            "chapter_sections": len(re.findall(r'Chapter\s+\d+', all_text, re.IGNORECASE)),
        }
        
        # Character type ratios
        total_chars = len(all_text)
        if total_chars > 0:
            korean_chars = len(re.findall(r'[가-힣]', all_text))
            english_chars = len(re.findall(r'[a-zA-Z]', all_text))
            numbers = len(re.findall(r'\d', all_text))
            
            char_ratios = {
                "korean": float(korean_chars / total_chars),
                "english": float(english_chars / total_chars),
                "numbers": float(numbers / total_chars),
            }
        else:
            char_ratios = {}
        
        # Table/figure indicators (simple heuristics)
        table_indicators = len(re.findall(r'표\s*\d+|Table\s*\d+', all_text, re.IGNORECASE))
        figure_indicators = len(re.findall(r'그림\s*\d+|Figure\s*\d+|그\.\s*\d+', all_text, re.IGNORECASE))
        
        return {
            "top_keywords": top_keywords,
            "section_patterns": section_patterns,
            "character_ratios": char_ratios,
            "table_indicators": table_indicators,
            "figure_indicators": figure_indicators,
        }
    
    def analyze_parsing_errors(self, documents: List[Dict]) -> Dict:
        """
        Analyze parsing errors.
        
        Args:
            documents: List of document dictionaries
        
        Returns:
            Dictionary with error analysis
        """
        if not documents:
            return {}
        
        errors_by_type = {}
        error_causes = {
            "file_not_found": 0,
            "parsing_error": 0,
            "encoding_error": 0,
            "other": 0,
        }
        failed_files = []
        
        for doc in documents:
            status = doc.get("status", "")
            file_path = doc.get("file_path", "")
            error = doc.get("error", "")
            
            if status == "failed":
                # Get file type
                file_type = self.data_loader.get_file_type_from_path(file_path)
                
                if file_type not in errors_by_type:
                    errors_by_type[file_type] = 0
                errors_by_type[file_type] += 1
                
                # Classify error cause
                error_lower = str(error).lower()
                if "not found" in error_lower or "file not found" in error_lower:
                    error_causes["file_not_found"] += 1
                elif "parsing" in error_lower or "parse" in error_lower:
                    error_causes["parsing_error"] += 1
                elif "encoding" in error_lower or "decode" in error_lower:
                    error_causes["encoding_error"] += 1
                else:
                    error_causes["other"] += 1
                
                failed_files.append({
                    "file_path": file_path,
                    "error": error,
                })
        
        total = len(documents)
        failure_rate_by_type = {
            file_type: float(count / total) if total > 0 else 0.0
            for file_type, count in errors_by_type.items()
        }
        
        return {
            "errors_by_file_type": errors_by_type,
            "failure_rate_by_type": failure_rate_by_type,
            "error_causes": error_causes,
            "failed_files": failed_files[:10],  # Limit to first 10
        }

