"""Document summarization module."""

from typing import Dict, List

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from src.common.logger import get_logger


SUMMARY_PROMPT = """다음은 RFP 문서의 내용입니다. 핵심 정보를 요약하세요.

문서 내용:
{context}

다음 항목을 포함하여 요약하세요:
1. 사업 개요 (3-4줄)
2. 주요 요구사항 (불릿 포인트)
3. 예산 정보
4. 마감일 및 일정
5. 필수 자격 요건

요약:
"""


class Summarizer:
    """Document summarization using LLM."""
    
    def __init__(self, llm, retrieval_agent, config: Dict):
        """
        Initialize Summarizer.
        
        Args:
            llm: LangChain LLM instance
            retrieval_agent: RetrievalAgent instance
            config: Configuration dictionary
        """
        self.llm = llm
        self.retrieval_agent = retrieval_agent
        self.config = config.get("summarization", {})
        self.logger = get_logger(__name__)
        
        # Create prompt template
        system_template = SystemMessagePromptTemplate.from_template(
            "당신은 RFP 문서 요약 전문가입니다. 문서의 핵심 정보를 간결하고 명확하게 요약하세요."
        )
        human_template = HumanMessagePromptTemplate.from_template(SUMMARY_PROMPT)
        
        self.prompt = ChatPromptTemplate.from_messages([
            system_template,
            human_template
        ])
    
    def summarize_document(
        self,
        doc_id: str,
        top_k: int = 20
    ) -> Dict:
        """
        Summarize a document by retrieving all its chunks.
        
        Args:
            doc_id: Document ID
            top_k: Maximum number of chunks to retrieve
        
        Returns:
            Dictionary with summary and key information
        """
        # Search for all chunks from this document
        # Use doc_id as a filter (if supported) or search with doc_id in query
        query = f"문서 {doc_id}"
        
        # Retrieve chunks (ideally filtered by doc_id, but basic retrieval for now)
        retrieval_results = self.retrieval_agent.retrieve(query, top_k=top_k)
        
        # Filter to only chunks from this document
        relevant_chunks = [
            r for r in retrieval_results["results"]
            if r.get("metadata", {}).get("doc_id") == doc_id
        ]
        
        if not relevant_chunks:
            self.logger.warning(f"No chunks found for document {doc_id}")
            return {
                "summary": "문서를 찾을 수 없습니다.",
                "key_points": [],
                "budget": None,
                "deadline": None,
                "requirements": [],
                "doc_id": doc_id
            }
        
        # Build context from all chunks
        context = self._build_context(relevant_chunks)
        
        # Generate summary
        messages = self.prompt.format_messages(context=context)
        response = self.llm.invoke(messages)
        
        summary_text = response.content
        
        # Extract key information (simple extraction from metadata)
        metadata = relevant_chunks[0].get("metadata", {}) if relevant_chunks else {}
        
        return {
            "summary": summary_text,
            "key_points": self._extract_key_points(summary_text),
            "budget": metadata.get("사업 금액"),
            "deadline": metadata.get("입찰 참여 마감일"),
            "requirements": [],  # Would need more sophisticated extraction
            "doc_id": doc_id
        }
    
    def summarize_section(
        self,
        doc_id: str,
        section_name: str
    ) -> Dict:
        """
        Summarize a specific section of a document.
        
        Args:
            doc_id: Document ID
            section_name: Section name
        
        Returns:
            Dictionary with section summary
        """
        # Similar to summarize_document but filter by section
        query = f"문서 {doc_id} 섹션 {section_name}"
        retrieval_results = self.retrieval_agent.retrieve(query, top_k=10)
        
        # Filter by section
        section_chunks = [
            r for r in retrieval_results["results"]
            if r.get("metadata", {}).get("section_name") == section_name
        ]
        
        if not section_chunks:
            return {
                "summary": "섹션을 찾을 수 없습니다.",
                "section_name": section_name,
                "doc_id": doc_id
            }
        
        context = self._build_context(section_chunks)
        messages = self.prompt.format_messages(context=context)
        response = self.llm.invoke(messages)
        
        return {
            "summary": response.content,
            "section_name": section_name,
            "doc_id": doc_id
        }
    
    def _build_context(self, chunks: List[Dict]) -> str:
        """Build context from chunks."""
        context_parts = []
        for chunk in chunks:
            context_parts.append(chunk.get("chunk_text", ""))
        return "\n\n".join(context_parts)
    
    def _extract_key_points(self, summary: str) -> List[str]:
        """Extract key points from summary (simple bullet point extraction)."""
        lines = summary.split("\n")
        key_points = []
        for line in lines:
            line = line.strip()
            if line and (line.startswith("-") or line.startswith("•") or line.startswith("*")):
                key_points.append(line.lstrip("-•* ").strip())
        return key_points[:10]  # Limit to 10 key points

