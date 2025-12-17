"""Basic text chunker implementation."""

from typing import List, Dict, Optional

from src.common.constants import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    MIN_CHUNK_SIZE,
)


class TextChunker:
    """Simple character-based chunker with overlap.

    이 구현은 우선 파이프라인을 돌리기 위한 기본 버전입니다.
    문장/단락 경계를 고려하지 않고, 일정 길이 기준으로 나누고 overlap만 적용합니다.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        separators: Optional[list] = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = min(chunk_overlap, chunk_size // 2)
        self.separators = separators or []

    def chunk(self, text: str, doc_id: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """Split text into chunks.

        Args:
            text: 원본 텍스트
            doc_id: 문서 ID (파일명 또는 공고번호)
            metadata: 청크에 공통으로 붙일 메타데이터

        Returns:
            청크 딕셔너리 리스트
        """
        if not text:
            return []

        metadata = metadata or {}
        chunks: List[Dict] = []
        n = len(text)
        index = 0
        chunk_index = 0

        while index < n:
            end = min(index + self.chunk_size, n)
            chunk_text = text[index:end]

            # 너무 작은 마지막 조각이면 이전 청크에 합치도록 나중에 후처리
            chunks.append(
                {
                    "chunk_id": f"{doc_id}_{chunk_index}",
                    "doc_id": doc_id,
                    "chunk_index": chunk_index,
                    "chunk_text": chunk_text,
                    "char_offset_start": index,
                    "char_offset_end": end,
                    "metadata": metadata,
                }
            )

            chunk_index += 1

            if end == n:
                break

            # 다음 시작 위치: overlap 적용 (하지만 뒤로 너무 많이 가지 않도록)
            index = max(end - self.chunk_overlap, index + MIN_CHUNK_SIZE)

        # 마지막 청크가 너무 작으면 이전 청크에 병합
        if len(chunks) >= 2 and len(chunks[-1]["chunk_text"]) < MIN_CHUNK_SIZE:
            last = chunks.pop()
            prev = chunks[-1]
            prev["chunk_text"] += last["chunk_text"]
            prev["char_offset_end"] = last["char_offset_end"]

        return chunks
