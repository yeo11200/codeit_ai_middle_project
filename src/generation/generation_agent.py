"""Generation Agent - unified interface for Q&A, summarization, and extraction."""

from typing import Dict, Optional

from src.generation.qa_chain import QAChain
from src.generation.summarizer import Summarizer
from src.generation.extractor import Extractor
from src.common.logger import get_logger


class GenerationAgent:
    """Main generation agent for Q&A, summarization, and extraction."""
    
    def __init__(self, config: Dict, llm, retrieval_agent):
        """
        Initialize GenerationAgent.
        
        Args:
            config: Configuration dictionary
            llm: LangChain LLM instance
            retrieval_agent: RetrievalAgent instance
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        self.qa_chain = QAChain(llm, retrieval_agent, config)
        self.summarizer = Summarizer(llm, retrieval_agent, config)
        self.extractor = Extractor(llm, retrieval_agent, config)
    
    def answer_question(self, query: str) -> Dict:
        """
        Answer a question using RAG.
        
        Args:
            query: User question
        
        Returns:
            Dictionary with answer, sources, confidence
        """
        return self.qa_chain.answer(query)
    
    def summarize_document(self, doc_id: str) -> Dict:
        """
        Summarize a document.
        
        Args:
            doc_id: Document ID
        
        Returns:
            Dictionary with summary and key information
        """
        top_k = self.config.get("rag", {}).get("top_k_for_context", 20)
        return self.summarizer.summarize_document(doc_id, top_k=top_k)
    
    def extract_info(
        self,
        doc_id: str,
        schema: Optional[Dict] = None
    ) -> Dict:
        """
        Extract structured information from a document.
        
        Args:
            doc_id: Document ID
            schema: Optional extraction schema
        
        Returns:
            Dictionary with extracted information
        """
        return self.extractor.extract_structured(doc_id, schema=schema)

