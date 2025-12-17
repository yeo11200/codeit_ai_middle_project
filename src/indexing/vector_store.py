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
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
