"""Section-based chunker (optional, simple stub).

현재는 실제 섹션 인식 로직 없이 기본 TextChunker에 위임하는 형태로 둡니다.
필요 시 목차 기반 섹션 인식 로직을 여기에 확장할 수 있습니다.
"""

from typing import List, Dict


class SectionChunker:
    """Stub implementation for section-based chunking."""

    def detect_sections(self, text: str) -> List[Dict]:
        """Detect sections in text.

        현재는 섹션 인식을 하지 않고, 전체를 하나의 섹션으로 간주합니다.
        """
        if not text:
            return []
        return [
            {
                "name": "전체",
                "level": 1,
                "start_pos": 0,
                "end_pos": len(text),
            }
        ]

    def chunk_by_sections(
        self,
        text: str,
        sections: List[Dict],
        base_metadata: Dict,
        text_chunker,
    ) -> List[Dict]:
        """Chunk text by sections using provided TextChunker.

        Args:
            text: 전체 텍스트
            sections: 섹션 정보 리스트
            base_metadata: 기본 메타데이터
            text_chunker: TextChunker 인스턴스
        """
        all_chunks: List[Dict] = []
        for section in sections:
            start = section["start_pos"]
            end = section["end_pos"]
            section_text = text[start:end]

            # 섹션 메타데이터 추가
            metadata = {**base_metadata}
            metadata["section_name"] = section.get("name")
            metadata["section_level"] = section.get("level")

            # doc_id는 base_metadata에서 가져오거나, 호출 측에서 채워줘야 함
            doc_id = metadata.get("doc_id", "unknown")
            chunks = text_chunker.chunk(section_text, doc_id=doc_id, metadata=metadata)

            # 섹션 오프셋 보정
            for ch in chunks:
                ch["char_offset_start"] += start
                ch["char_offset_end"] += start
            all_chunks.extend(chunks)

        return all_chunks
