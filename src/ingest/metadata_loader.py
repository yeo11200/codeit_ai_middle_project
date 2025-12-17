"""Metadata loader from CSV file."""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from src.common.logger import get_logger


class MetadataLoader:
    """Load and manage metadata from CSV file."""
    
    def __init__(self):
        self.metadata_dict: Dict[str, Dict] = {}
        self.logger = get_logger(__name__)
    
    def load_from_csv(self, csv_path: str) -> Dict[str, Dict]:
        """
        Load metadata from CSV file.
        
        Args:
            csv_path: Path to CSV file
        
        Returns:
            Dictionary mapping filename to metadata
        """
        csv_file = Path(csv_path)
        
        if not csv_file.exists():
            self.logger.warning(f"CSV file not found: {csv_path}")
            return {}
        
        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
            
            # Expected columns
            expected_columns = [
                "공고 번호", "공고 차수", "사업명", "사업 금액",
                "발주 기관", "공개 일자", "입찰 참여 시작일",
                "입찰 참여 마감일", "사업 요약", "파일명"
            ]
            
            # Check if '파일명' column exists
            if "파일명" not in df.columns:
                self.logger.warning("'파일명' column not found in CSV")
                return {}
            
            # Build metadata dictionary
            for _, row in df.iterrows():
                filename = str(row.get("파일명", "")).strip()
                if not filename:
                    continue
                
                metadata = {}
                for col in expected_columns:
                    if col in df.columns:
                        value = row.get(col)
                        # Convert to appropriate type
                        if pd.isna(value):
                            metadata[col] = None
                        else:
                            metadata[col] = value
                
                self.metadata_dict[filename] = metadata
            
            self.logger.info(f"Loaded metadata for {len(self.metadata_dict)} files")
            return self.metadata_dict
        
        except Exception as e:
            self.logger.error(f"Failed to load CSV: {e}")
            return {}
    
    def get_metadata(self, filename: str) -> Dict:
        """
        Get metadata for a specific file.
        
        Args:
            filename: Name of the file
        
        Returns:
            Metadata dictionary or empty dict if not found
        """
        # Try exact match first
        if filename in self.metadata_dict:
            return self.metadata_dict[filename]
        
        # Try with just the filename (without path)
        filename_only = Path(filename).name
        if filename_only in self.metadata_dict:
            return self.metadata_dict[filename_only]
        
        # Return empty dict if not found
        return {}

