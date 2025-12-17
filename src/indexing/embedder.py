"""Embedding generator using OpenAI via LangChain."""

import os
import time
from typing import List

from langchain_openai import OpenAIEmbeddings
from src.common.logger import get_logger


class Embedder:
    """Wrapper around OpenAIEmbeddings with basic retry logic."""

    def __init__(self, config: dict):
        self.logger = get_logger(__name__)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

        model = config.get("embedding_model", "text-embedding-3-large")
        self.batch_size = config.get("batch_size", 100)
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 1.0)

        self.embeddings = OpenAIEmbeddings(model=model, api_key=api_key)

    def embed(self, text: str) -> List[float]:
        """Embed a single text."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts with retry logic."""
        results: List[List[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            attempts = 0
            while True:
                try:
                    batch_embeddings = self.embeddings.embed_documents(batch)
                    results.extend(batch_embeddings)
                    break
                except Exception as e:
                    attempts += 1
                    if attempts > self.max_retries:
                        self.logger.error(f"Embedding failed after {self.max_retries} retries: {e}")
                        raise
                    self.logger.warning(
                        f"Embedding batch failed (attempt {attempts}/{self.max_retries}): {e}. "
                        f"Retrying in {self.retry_delay} seconds..."
                    )
                    time.sleep(self.retry_delay)

        return results
