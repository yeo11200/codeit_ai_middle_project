"""Chunk analyzer for EDA."""

from typing import Dict, List
from collections import Counter
from src.common.logger import get_logger


class ChunkAnalyzer:
    """Analyze chunk statistics and distribution."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def analyze_chunk_statistics(self, chunks: List[Dict]) -> Dict:
        """
        Analyze chunk statistics.
        
        Args:
            chunks: List of chunk dictionaries
        
        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {}
        
        chunk_sizes = [len(chunk.get("chunk_text", "")) for chunk in chunks]
        
        if not chunk_sizes:
            return {}
        
        # Document to chunk mapping
        doc_chunk_counts = Counter([chunk.get("doc_id", "unknown") for chunk in chunks])
        
        stats = {
            "total_chunks": len(chunks),
            "total_documents": len(doc_chunk_counts),
            "avg_chunks_per_doc": float(sum(doc_chunk_counts.values()) / len(doc_chunk_counts)) if doc_chunk_counts else 0.0,
            "chunk_size": {
                "mean": float(sum(chunk_sizes) / len(chunk_sizes)),
                "median": float(sorted(chunk_sizes)[len(chunk_sizes) // 2]),
                "min": int(min(chunk_sizes)),
                "max": int(max(chunk_sizes)),
                "std": float((sum((x - sum(chunk_sizes) / len(chunk_sizes))**2 for x in chunk_sizes) / len(chunk_sizes))**0.5),
            },
        }
        
        return stats
    
    def analyze_chunk_distribution(self, chunks: List[Dict]) -> Dict:
        """
        Analyze chunk distribution.
        
        Args:
            chunks: List of chunk dictionaries
        
        Returns:
            Dictionary with distribution data
        """
        if not chunks:
            return {}
        
        chunk_sizes = [len(chunk.get("chunk_text", "")) for chunk in chunks]
        doc_chunk_counts = [len(list(filter(lambda c: c.get("doc_id") == doc_id, chunks)))
                           for doc_id in set(c.get("doc_id", "unknown") for c in chunks)]
        
        # File type distribution
        file_types = {}
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            file_path = metadata.get("file_path", "")
            if file_path:
                ext = file_path.split(".")[-1].lower() if "." in file_path else "unknown"
                file_type = ext.upper() if ext in ["hwp", "pdf", "docx"] else "UNKNOWN"
                if file_type not in file_types:
                    file_types[file_type] = []
                file_types[file_type].append(len(chunk.get("chunk_text", "")))
        
        avg_by_type = {
            file_type: float(sum(sizes) / len(sizes)) if sizes else 0.0
            for file_type, sizes in file_types.items()
        }
        
        return {
            "chunk_sizes": chunk_sizes[:100],  # Limit for visualization
            "chunks_per_doc": doc_chunk_counts,
            "avg_size_by_file_type": avg_by_type,
        }
    
    def analyze_overlap_effectiveness(self, chunks: List[Dict]) -> Dict:
        """
        Analyze overlap effectiveness.
        
        Args:
            chunks: List of chunk dictionaries
        
        Returns:
            Dictionary with overlap analysis
        """
        if not chunks:
            return {}
        
        # Group chunks by document
        doc_chunks = {}
        for chunk in chunks:
            doc_id = chunk.get("doc_id", "unknown")
            if doc_id not in doc_chunks:
                doc_chunks[doc_id] = []
            doc_chunks[doc_id].append(chunk)
        
        # Analyze overlap
        total_overlap_chars = 0
        chunks_with_overlap = 0
        
        for doc_id, doc_chunk_list in doc_chunks.items():
            # Sort by chunk_index
            sorted_chunks = sorted(doc_chunk_list, key=lambda c: c.get("chunk_index", 0))
            
            for i in range(len(sorted_chunks) - 1):
                current = sorted_chunks[i]
                next_chunk = sorted_chunks[i + 1]
                
                current_end = current.get("char_offset_end", 0)
                next_start = next_chunk.get("char_offset_start", 0)
                
                # Check if there's overlap
                if current_end > next_start:
                    overlap_size = current_end - next_start
                    total_overlap_chars += overlap_size
                    chunks_with_overlap += 1
        
        total_chunks = len(chunks)
        overlap_ratio = float(chunks_with_overlap / total_chunks) if total_chunks > 0 else 0.0
        
        return {
            "chunks_with_overlap": chunks_with_overlap,
            "overlap_ratio": overlap_ratio,
            "avg_overlap_size": float(total_overlap_chars / chunks_with_overlap) if chunks_with_overlap > 0 else 0.0,
        }
    
    def analyze_metadata_coverage(self, chunks: List[Dict]) -> Dict:
        """
        Analyze metadata coverage.
        
        Args:
            chunks: List of chunk dictionaries
        
        Returns:
            Dictionary with metadata coverage statistics
        """
        if not chunks:
            return {}
        
        # Required fields
        required_fields = ["doc_id", "사업명"]
        optional_fields = ["공고 번호", "발주 기관", "사업 금액"]
        
        coverage = {}
        
        for field in required_fields + optional_fields:
            if field == "doc_id":
                # Check doc_id in chunk directly
                present = sum(1 for chunk in chunks if chunk.get("doc_id"))
            else:
                # Check in metadata
                present = sum(1 for chunk in chunks 
                           if chunk.get("metadata", {}).get(field) not in [None, "", "N/A"])
            
            coverage[field] = {
                "present": present,
                "total": len(chunks),
                "coverage_ratio": float(present / len(chunks)) if chunks else 0.0,
            }
        
        return {
            "required_fields": {f: coverage[f] for f in required_fields if f in coverage},
            "optional_fields": {f: coverage[f] for f in optional_fields if f in coverage},
        }

