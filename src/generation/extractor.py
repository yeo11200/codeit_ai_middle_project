"""Structured information extraction module."""

from typing import Dict, List, Any, Optional
import json
import re

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from src.common.logger import get_logger


EXTRACTION_PROMPT = """다음 문서에서 요청한 정보를 추출하여 JSON 형식으로만 답변하세요.

문서 내용:
{context}

추출할 정보 스키마:
{schema}

JSON 형식으로만 답변하세요. 정보가 없으면 null을 사용하세요.
"""

DEFAULT_EXTRACTION_SCHEMA = {
    "budget": {
        "type": "float",
        "description": "사업 예산 금액 (원)"
    },
    "deadline": {
        "type": "datetime",
        "description": "입찰 참여 마감일"
    },
    "submission_method": {
        "type": "string",
        "description": "제출 방식"
    },
    "required_qualifications": {
        "type": "array",
        "items": {"type": "string"},
        "description": "필수 자격 요건 리스트"
    },
    "evaluation_criteria": {
        "type": "array",
        "items": {"type": "string"},
        "description": "평가 기준 리스트"
    }
}


class Extractor:
    """Extract structured information from documents."""
    
    def __init__(self, llm, retrieval_agent, config: Dict):
        """
        Initialize Extractor.
        
        Args:
            llm: LangChain LLM instance
            retrieval_agent: RetrievalAgent instance
            config: Configuration dictionary
        """
        self.llm = llm
        self.retrieval_agent = retrieval_agent
        self.logger = get_logger(__name__)
        
        # Create prompt template
        system_template = SystemMessagePromptTemplate.from_template(
            "당신은 RFP 문서에서 구조화된 정보를 추출하는 전문가입니다. "
            "요청된 스키마에 맞춰 JSON 형식으로만 답변하세요."
        )
        human_template = HumanMessagePromptTemplate.from_template(EXTRACTION_PROMPT)
        
        self.prompt = ChatPromptTemplate.from_messages([
            system_template,
            human_template
        ])
    
    def extract_structured(
        self,
        doc_id: str,
        schema: Optional[Dict] = None
    ) -> Dict:
        """
        Extract structured information from a document.
        
        Args:
            doc_id: Document ID
            schema: Extraction schema (uses default if not provided)
        
        Returns:
            Dictionary with extracted information
        """
        if schema is None:
            schema = DEFAULT_EXTRACTION_SCHEMA
        
        # Retrieve document chunks
        query = f"문서 {doc_id}"
        retrieval_results = self.retrieval_agent.retrieve(query, top_k=20)
        
        # Filter to document chunks
        doc_chunks = [
            r for r in retrieval_results["results"]
            if r.get("metadata", {}).get("doc_id") == doc_id
        ]
        
        if not doc_chunks:
            self.logger.warning(f"No chunks found for document {doc_id}")
            return self._empty_result(schema)
        
        # Build context
        context = self._build_context(doc_chunks)
        
        # Format schema for prompt
        schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
        
        # Generate extraction
        messages = self.prompt.format_messages(
            context=context,
            schema=schema_str
        )
        
        response = self.llm.invoke(messages)
        
        # Parse JSON from response
        extracted = self._parse_json_response(response.content)
        
        return extracted
    
    def _build_context(self, chunks: List[Dict]) -> str:
        """Build context from chunks."""
        context_parts = []
        for chunk in chunks:
            context_parts.append(chunk.get("chunk_text", ""))
        return "\n\n".join(context_parts)
    
    def _parse_json_response(self, response_text: str) -> Dict:
        """Parse JSON from LLM response."""
        # Try to extract JSON from response
        # Look for JSON block
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                self.logger.warning("Failed to parse JSON from response")
        
        # Fallback: try parsing entire response
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            self.logger.error("Could not parse JSON from LLM response")
            return {}
    
    def _empty_result(self, schema: Dict) -> Dict:
        """Create empty result based on schema."""
        result = {}
        for key, value in schema.items():
            if value.get("type") == "array":
                result[key] = []
            else:
                result[key] = None
        return result

