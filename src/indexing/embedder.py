"""Embedding generator using OpenAI via LangChain."""

import os
import time
from typing import List

from langchain_openai import OpenAIEmbeddings
from src.common.logger import get_logger


class Embedder:
    """Wrapper around OpenAIEmbeddings with basic retry logic and model fallback."""

    # Fallback models in order of preference
    FALLBACK_MODELS = [
        "text-embedding-3-small",  # Smaller but still good
        "text-embedding-ada-002",  # Older but widely available
    ]

    def __init__(self, config: dict):
        self.logger = get_logger(__name__)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment. "
                "Please set it using: export OPENAI_API_KEY=sk-your-key-here\n"
                "See ENV_SETUP.md for detailed instructions."
            )

        self.primary_model = config.get("embedding_model", "text-embedding-3-large")
        self.batch_size = config.get("batch_size", 100)
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 1.0)
        
        # Try primary model first, fallback if needed
        self.model = self.primary_model
        self.embeddings = self._initialize_embeddings(api_key)

    def _initialize_embeddings(self, api_key: str):
        """Initialize embeddings with fallback support."""
        models_to_try = [self.primary_model] + self.FALLBACK_MODELS
        
        for model in models_to_try:
            try:
                embeddings = OpenAIEmbeddings(model=model, api_key=api_key)
                # Test with a small batch
                test_text = ["test"]
                embeddings.embed_documents(test_text)
                
                if model != self.primary_model:
                    self.logger.warning(
                        f"Primary model '{self.primary_model}' not available. "
                        f"Using fallback model: '{model}'"
                    )
                else:
                    self.logger.info(f"Using embedding model: {model}")
                
                self.model = model
                return embeddings
                
            except Exception as e:
                if "model_not_found" in str(e) or "403" in str(e) or "does not have access" in str(e):
                    self.logger.debug(f"Model '{model}' not available: {e}")
                    continue
                else:
                    # Other errors, try next model
                    self.logger.debug(f"Error with model '{model}': {e}")
                    continue
        
        # If all models failed
        raise ValueError(
            f"None of the embedding models are available: {models_to_try}. "
            "Please check your OpenAI API access or use a different model."
        )

    def embed(self, text: str) -> List[float]:
        """Embed a single text."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts with retry logic and model fallback."""
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
                    error_str = str(e)
                    
                    # Check if it's a model access error
                    if ("model_not_found" in error_str or 
                        "403" in error_str or 
                        "does not have access" in error_str):
                        
                        # Try fallback models
                        if self.model == self.primary_model:
                            self.logger.warning(
                                f"Primary model '{self.primary_model}' failed. "
                                "Trying fallback models..."
                            )
                            if self._try_fallback_models():
                                continue  # Retry with new model
                        
                        # If already using fallback or all failed
                        self.logger.error(
                            f"Embedding failed: Model '{self.model}' not accessible. {e}"
                        )
                        raise
                    
                    # Other errors - retry
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
    
    def _try_fallback_models(self) -> bool:
        """Try to switch to a fallback model. Returns True if successful."""
        api_key = os.getenv("OPENAI_API_KEY")
        
        for fallback_model in self.FALLBACK_MODELS:
            if fallback_model == self.model:
                continue  # Already tried this one
            
            try:
                self.logger.info(f"Trying fallback model: {fallback_model}")
                new_embeddings = OpenAIEmbeddings(model=fallback_model, api_key=api_key)
                # Test with small batch
                test_text = ["test"]
                new_embeddings.embed_documents(test_text)
                
                self.embeddings = new_embeddings
                self.model = fallback_model
                self.logger.info(f"Successfully switched to model: {fallback_model}")
                return True
                
            except Exception as e:
                self.logger.debug(f"Fallback model '{fallback_model}' failed: {e}")
                continue
        
        return False
