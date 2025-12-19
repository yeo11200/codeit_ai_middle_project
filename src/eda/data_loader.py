"""Data loader for EDA - loads CSV, JSON, and JSONL files."""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from src.common.logger import get_logger
from src.common.utils import validate_file


class DataLoader:
    """Load data from various sources for EDA."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def load_metadata_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Load metadata from CSV file.
        
        Args:
            csv_path: Path to CSV file (e.g., data/data_list.csv)
        
        Returns:
            DataFrame with metadata
        """
        if not validate_file(csv_path, required=False):
            self.logger.warning(f"CSV file not found: {csv_path}")
            return pd.DataFrame()
        
        try:
            # Try UTF-8 first
            df = pd.read_csv(csv_path, encoding="utf-8")
            self.logger.info(f"Loaded metadata CSV: {len(df)} rows")
            return df
        except UnicodeDecodeError:
            # Try CP949 (Korean encoding)
            try:
                df = pd.read_csv(csv_path, encoding="cp949")
                self.logger.info(f"Loaded metadata CSV (CP949): {len(df)} rows")
                return df
            except Exception as e:
                self.logger.error(f"Failed to load CSV: {e}")
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Failed to load CSV: {e}")
            return pd.DataFrame()
    
    def load_preprocessed_json(self, json_dir: str) -> List[Dict]:
        """
        Load all preprocessed JSON files from directory.
        
        Args:
            json_dir: Directory containing JSON files (e.g., data/preprocessed/)
        
        Returns:
            List of document dictionaries
        """
        json_path = Path(json_dir)
        if not json_path.exists():
            self.logger.warning(f"JSON directory not found: {json_dir}")
            return []
        
        documents = []
        json_files = list(json_path.glob("*.json"))
        
        if not json_files:
            self.logger.warning(f"No JSON files found in {json_dir}")
            return []
        
        self.logger.info(f"Loading {len(json_files)} JSON files from {json_dir}")
        
        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    documents.append(data)
            except Exception as e:
                self.logger.warning(f"Failed to load {json_file}: {e}")
                continue
        
        self.logger.info(f"Loaded {len(documents)} documents")
        return documents
    
    def load_chunks_jsonl(self, jsonl_path: str) -> List[Dict]:
        """
        Load chunks from JSONL file.
        
        Args:
            jsonl_path: Path to JSONL file (e.g., data/features/chunks.jsonl)
        
        Returns:
            List of chunk dictionaries
        """
        if not validate_file(jsonl_path, required=False):
            self.logger.warning(f"JSONL file not found: {jsonl_path}")
            return []
        
        chunks = []
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        chunk = json.loads(line.strip())
                        chunks.append(chunk)
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Failed to parse line {line_num} in {jsonl_path}: {e}")
                        continue
            
            self.logger.info(f"Loaded {len(chunks)} chunks from {jsonl_path}")
            return chunks
        except Exception as e:
            self.logger.error(f"Failed to load JSONL: {e}")
            return []
    
    def get_file_type_from_path(self, file_path: str) -> str:
        """Extract file type from file path."""
        path = Path(file_path)
        ext = path.suffix.lower()
        
        if ext == ".hwp":
            return "HWP"
        elif ext == ".pdf":
            return "PDF"
        elif ext == ".docx":
            return "DOCX"
        else:
            return "UNKNOWN"

