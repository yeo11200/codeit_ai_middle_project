"""ChromaDB-based vector store wrapper."""

from typing import List, Dict

import chromadb
from chromadb.config import Settings

from src.common.logger import get_logger


class VectorStore:
    """Simple wrapper around ChromaDB PersistentClient."""

    def __init__(self, config: dict):
        self.logger = get_logger(__name__)
        vs_cfg = config.get("vector_store", config)

        persist_dir = vs_cfg["persist_dir"]
        collection_name = vs_cfg["collection_name"]

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection_name = collection_name
        self.collection = self._create_or_get_collection(collection_name)

    def _create_or_get_collection(self, name: str):
        """Create or get existing collection."""
        try:
            return self.client.get_collection(name)
        except Exception:
            self.logger.info(f"Creating new Chroma collection: {name}")
            return self.client.create_collection(name)

    def add_documents(
        self,
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict],
        ids: List[str],
    ) -> None:
        """Add documents with embeddings to the collection."""
        # Clean metadata: remove None values and convert to ChromaDB-compatible types
        cleaned_metadatas = [self._clean_metadata(meta) for meta in metadatas]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=cleaned_metadatas,
        )
    
    def _clean_metadata(self, metadata: Dict) -> Dict:
        """
        Clean metadata by removing None values and converting to ChromaDB-compatible types.
        
        ChromaDB only accepts: bool, int, float, str, or SparseVector.
        None values and empty strings are removed.
        Recursively cleans nested structures.
        """
        if not isinstance(metadata, dict):
            return {}
        
        cleaned = {}
        
        for key, value in metadata.items():
            # Skip None values (JSON null becomes None in Python)
            if value is None:
                continue
            
            # Skip empty strings
            if value == "":
                continue
            
            # Handle nested dicts
            if isinstance(value, dict):
                nested_cleaned = self._clean_metadata(value)
                if nested_cleaned:
                    # Convert nested dict to string to avoid nested structures
                    cleaned[key] = str(nested_cleaned)
                continue
            
            # Handle lists/tuples
            if isinstance(value, (list, tuple)):
                # Filter out None values from lists
                filtered_list = [v for v in value if v is not None and v != ""]
                if filtered_list:
                    cleaned[key] = str(filtered_list)
                continue
            
            # Convert to ChromaDB-compatible types
            if isinstance(value, bool):
                cleaned[key] = value
            elif isinstance(value, int):
                cleaned[key] = value
            elif isinstance(value, float):
                # Check for NaN or Inf
                if not (value != value or value == float('inf') or value == float('-inf')):
                    cleaned[key] = value
            elif isinstance(value, str):
                cleaned[key] = value
            else:
                # Convert other types to strings, but skip if None-like
                try:
                    str_value = str(value)
                    if str_value and str_value.lower() not in ['none', 'null', 'nan']:
                        cleaned[key] = str_value
                except Exception:
                    # Skip if conversion fails
                    self.logger.debug(f"Skipping metadata key '{key}' with unconvertible value")
                    continue
        
        return cleaned
