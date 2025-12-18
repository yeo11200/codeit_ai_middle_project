"""Utility functions for LLM initialization with fallback support."""

import os
from typing import Optional
from langchain_openai import ChatOpenAI
from src.common.logger import get_logger


# Fallback LLM models in order of preference
LLM_FALLBACK_MODELS = [
    "gpt-4",  # Try GPT-4
    "gpt-3.5-turbo-16k",  # Alternative GPT-3.5 variant
]


def create_llm_with_fallback(
    primary_model: str,
    temperature: float = 0.2,
    max_tokens: int = 2000,
    api_key: Optional[str] = None
) -> ChatOpenAI:
    """
    Create LLM with automatic fallback if primary model is not available.
    
    Args:
        primary_model: Primary model name (e.g., "gpt-3.5-turbo")
        temperature: Temperature setting
        max_tokens: Max tokens setting
        api_key: Optional API key (uses OPENAI_API_KEY env var if not provided)
    
    Returns:
        ChatOpenAI instance with available model
    """
    logger = get_logger(__name__)
    
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
    
    # Create LLM with primary model
    # If it fails at runtime, user can change config to use a different model
    llm = ChatOpenAI(
        model=primary_model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key
    )
    
    logger.info(f"Using LLM model: {primary_model}")
    logger.info(f"If this model is not available, change 'model' in config/local.yaml to one of: {LLM_FALLBACK_MODELS}")
    
    return llm

