"""Indexing Agent - read chunks and build vector index."""

import time
from typing import Dict, List

from tqdm import tqdm

from src.indexing.embedder import Embedder
from src.indexing.vector_store import VectorStore
from src.common.logger import get_logger
from src.common.utils import load_jsonl


class IndexingAgent:
    """Agent to index chunks into ChromaDB."""

    def __init__(self, config: Dict):
        self.logger = get_logger(__name__)
        self.config = config.get("indexing", {})

        self.embedder = Embedder(self.config)
        self.vector_store = VectorStore(self.config)

    def index_chunks(self, chunks_path: str) -> Dict:
        """Index chunks from JSONL file.

        Args:
            chunks_path: Path to chunks JSONL file (e.g. data/features/chunks.jsonl)
        """
        start_time = time.time()
        errors: List[str] = []

        chunks = load_jsonl(chunks_path)
        if not chunks:
            self.logger.warning(f"No chunks found in {chunks_path}")
            return {
                "total_chunks": 0,
                "indexed_chunks": 0,
                "failed_chunks": 0,
                "processing_time": 0.0,
                "average_embedding_time": 0.0,
                "errors": [],
            }

        self.logger.info(f"Loaded {len(chunks)} chunks from {chunks_path}")

        texts: List[str] = []
        metadatas: List[Dict] = []
        ids: List[str] = []

        for ch in chunks:
            texts.append(ch.get("chunk_text", ""))
            metadatas.append(ch.get("metadata", {}))
            ids.append(ch.get("chunk_id"))

        total = len(texts)
        batch_size = self.config.get("batch_size", 100)
        embedding_times: List[float] = []

        indexed = 0
        failed = 0

        for i in tqdm(range(0, total, batch_size), desc="Indexing chunks"):
            batch_texts = texts[i : i + batch_size]
            batch_metas = metadatas[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]

            try:
                t0 = time.time()
                batch_embeddings = self.embedder.embed_batch(batch_texts)
                embedding_times.append(time.time() - t0)

                self.vector_store.add_documents(
                    embeddings=batch_embeddings,
                    documents=batch_texts,
                    metadatas=batch_metas,
                    ids=batch_ids,
                )
                indexed += len(batch_texts)
            except Exception as e:
                msg = f"Failed to index batch {i // batch_size}: {e}"
                self.logger.error(msg)
                errors.append(msg)
                failed += len(batch_texts)

        total_time = time.time() - start_time
        avg_embed_time = (
            sum(embedding_times) / len(embedding_times)
            if embedding_times
            else 0.0
        )

        report = {
            "total_chunks": total,
            "indexed_chunks": indexed,
            "failed_chunks": failed,
            "processing_time": total_time,
            "average_embedding_time": avg_embed_time,
            "errors": errors,
        }

        self.logger.info("=" * 60)
        self.logger.info("Indexing Summary")
        self.logger.info("=" * 60)
        self.logger.info(str(report))

        return report
