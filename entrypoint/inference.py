"""Inference pipeline entrypoint - query processing and answer generation."""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, skip

from src.common.config import load_config
from src.common.logger import get_logger


def main():
    """Main inference pipeline."""
    parser = argparse.ArgumentParser(description="RFP RAG Inference Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/local.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Query string to search",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["search", "qa", "summarize", "extract"],
        default="qa",
        help="Inference mode",
    )
    parser.add_argument(
        "--doc-id",
        type=str,
        help="Document ID (for summarize/extract modes)",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger = get_logger(__name__, log_level=config.get("logging", {}).get("level", "INFO"))
    
    logger.info("=" * 60)
    logger.info("RFP RAG Inference Pipeline")
    logger.info("=" * 60)
    
    try:
        # Initialize agents
        from src.indexing.vector_store import VectorStore
        from src.indexing.embedder import Embedder
        from src.retrieval.retrieval_agent import RetrievalAgent
        from src.generation.generation_agent import GenerationAgent
        from src.common.llm_utils import create_llm_with_fallback
        
        # Vector store and embedder
        vector_store = VectorStore(config["indexing"])
        embedder = Embedder(config["indexing"])
        
        # Retrieval agent
        retrieval_agent = RetrievalAgent(config["retrieval"], vector_store, embedder)
        
        # LLM with fallback
        llm = create_llm_with_fallback(
            primary_model=config["generation"]["llm"]["model"],
            temperature=config["generation"]["llm"]["temperature"],
            max_tokens=config["generation"]["llm"]["max_tokens"],
        )
        
        # Generation agent
        generation_agent = GenerationAgent(config["generation"], llm, retrieval_agent)
        
        if args.mode == "search":
            if not args.query:
                logger.error("--query is required for search mode")
                sys.exit(1)
            
            results = retrieval_agent.retrieve(args.query, top_k=10)
            print("\nSearch Results:")
            print("=" * 60)
            for i, result in enumerate(results["results"][:5], 1):
                print(f"\n[{i}] Score: {result['score']:.4f}")
                print(f"Doc: {result['metadata'].get('business_name', 'N/A')}")
                print(f"Text: {result['chunk_text'][:200]}...")
        
        elif args.mode == "qa":
            if not args.query:
                logger.error("--query is required for qa mode")
                sys.exit(1)
            
            answer = generation_agent.answer_question(args.query)
            print("\nAnswer:")
            print("=" * 60)
            print(answer["answer"])
            print("\nSources:")
            for source in answer["sources"][:3]:
                print(f"- {source['doc_id']} (score: {source['score']:.4f})")
        
        elif args.mode == "summarize":
            if not args.doc_id:
                logger.error("--doc-id is required for summarize mode")
                sys.exit(1)
            
            summary = generation_agent.summarize_document(args.doc_id)
            print("\nSummary:")
            print("=" * 60)
            print(summary["summary"])
        
        elif args.mode == "extract":
            if not args.doc_id:
                logger.error("--doc-id is required for extract mode")
                sys.exit(1)
            
            extracted = generation_agent.extract_info(args.doc_id)
            print("\nExtracted Information:")
            print("=" * 60)
            import json
            print(json.dumps(extracted, ensure_ascii=False, indent=2))
        
        logger.info("=" * 60)
        logger.info("Inference completed successfully!")
        logger.info("=" * 60)
    
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

