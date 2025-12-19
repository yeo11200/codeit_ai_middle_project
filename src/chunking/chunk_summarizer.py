"""Chunk summarizer - group chunks by file and create summaries."""

from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import json
from src.common.logger import get_logger
from src.common.utils import ensure_dir


class ChunkSummarizer:
    """Summarize chunks by file and save as JSON."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def summarize_chunks_by_file(self, chunks: List[Dict]) -> Dict[str, Dict]:
        """
        Group chunks by file (doc_id) and create summaries.
        
        Args:
            chunks: List of chunk dictionaries
        
        Returns:
            Dictionary mapping doc_id to file summary
        """
        # Group chunks by doc_id
        chunks_by_file: Dict[str, List[Dict]] = defaultdict(list)
        
        for chunk in chunks:
            doc_id = chunk.get("doc_id", "unknown")
            chunks_by_file[doc_id].append(chunk)
        
        # Create summaries for each file
        file_summaries: Dict[str, Dict] = {}
        
        for doc_id, file_chunks in chunks_by_file.items():
            # Get metadata from first chunk (should be same for all chunks)
            first_chunk = file_chunks[0]
            metadata = first_chunk.get("metadata", {})
            
            # Calculate statistics
            chunk_lengths = [len(c.get("chunk_text", "")) for c in file_chunks]
            total_length = sum(chunk_lengths)
            avg_length = total_length / len(file_chunks) if file_chunks else 0
            min_length = min(chunk_lengths) if chunk_lengths else 0
            max_length = max(chunk_lengths) if chunk_lengths else 0
            
            # Create summary
            summary = {
                "doc_id": doc_id,
                "file_name": metadata.get("파일명", doc_id),
                "file_path": metadata.get("file_path", ""),
                "file_type": metadata.get("file_type", ""),
                "metadata": {
                    "사업명": metadata.get("사업명", ""),
                    "공고 번호": metadata.get("공고 번호", ""),
                    "발주 기관": metadata.get("발주 기관", ""),
                    "사업 금액": metadata.get("사업 금액", ""),
                    "공개 일자": metadata.get("공개 일자", ""),
                    "입찰 참여 마감일": metadata.get("입찰 참여 마감일", ""),
                },
                "chunking_statistics": {
                    "total_chunks": len(file_chunks),
                    "total_text_length": total_length,
                    "average_chunk_length": round(avg_length, 2),
                    "min_chunk_length": min_length,
                    "max_chunk_length": max_length,
                },
                "chunks": [
                    {
                        "chunk_id": c.get("chunk_id", ""),
                        "chunk_index": c.get("chunk_index", 0),
                        "chunk_text": c.get("chunk_text", ""),
                        "chunk_length": len(c.get("chunk_text", "")),
                        "char_offset_start": c.get("char_offset_start", 0),
                        "char_offset_end": c.get("char_offset_end", 0),
                    }
                    for c in file_chunks
                ]
            }
            
            file_summaries[doc_id] = summary
        
        return file_summaries
    
    def save_summaries_to_files(
        self,
        file_summaries: Dict[str, Dict],
        output_dir: str,
        save_individual: bool = True,
        save_combined: bool = True
    ) -> Dict[str, str]:
        """
        Save file summaries to JSON files.
        
        Args:
            file_summaries: Dictionary of file summaries
            output_dir: Output directory for JSON files
            save_individual: Save individual JSON file for each document
            save_combined: Save combined JSON file with all summaries
        
        Returns:
            Dictionary with paths to saved files
        """
        output_path = Path(output_dir)
        ensure_dir(str(output_path))
        
        saved_files = {}
        
        # Save individual files
        if save_individual:
            individual_dir = output_path / "by_file"
            ensure_dir(str(individual_dir))
            
            for doc_id, summary in file_summaries.items():
                # Create safe filename from doc_id
                safe_filename = self._create_safe_filename(doc_id)
                file_path = individual_dir / f"{safe_filename}.json"
                
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
                
                saved_files[doc_id] = str(file_path)
            
            self.logger.info(f"Saved {len(file_summaries)} individual file summaries to {individual_dir}")
        
        # Save combined file
        if save_combined:
            combined_path = output_path / "chunks_summary.json"
            
            # Create combined structure
            combined_data = {
                "total_files": len(file_summaries),
                "total_chunks": sum(s["chunking_statistics"]["total_chunks"] for s in file_summaries.values()),
                "files": file_summaries
            }
            
            with open(combined_path, "w", encoding="utf-8") as f:
                json.dump(combined_data, f, ensure_ascii=False, indent=2)
            
            saved_files["combined"] = str(combined_path)
            self.logger.info(f"Saved combined summary to {combined_path}")
        
        return saved_files
    
    def _create_safe_filename(self, doc_id: str) -> str:
        """Create a safe filename from doc_id."""
        # Remove or replace unsafe characters
        safe = doc_id.replace("/", "_").replace("\\", "_").replace(":", "_")
        safe = safe.replace("*", "_").replace("?", "_").replace('"', "_")
        safe = safe.replace("<", "_").replace(">", "_").replace("|", "_")
        # Limit length
        if len(safe) > 200:
            safe = safe[:200]
        return safe
    
    def create_summary_only(self, file_summaries: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Create summary-only version (without chunk texts) for overview.
        
        Args:
            file_summaries: Full file summaries with chunks
        
        Returns:
            Summary-only dictionary
        """
        summary_only = {}
        
        for doc_id, full_summary in file_summaries.items():
            summary_only[doc_id] = {
                "doc_id": full_summary["doc_id"],
                "file_name": full_summary["file_name"],
                "file_path": full_summary["file_path"],
                "file_type": full_summary["file_type"],
                "metadata": full_summary["metadata"],
                "chunking_statistics": full_summary["chunking_statistics"],
                # Exclude chunks array
            }
        
        return summary_only

