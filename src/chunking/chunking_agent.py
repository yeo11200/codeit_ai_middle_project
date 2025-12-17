"""Chunking Agent - split preprocessed documents into chunks."""

from pathlib import Path
from typing import Dict, List

from src.chunking.chunker import TextChunker
from src.chunking.section_chunker import SectionChunker
from src.common.logger import get_logger
from src.common.utils import load_json, save_jsonl
from src.common.constants import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    MIN_CHUNK_SIZE,
)


class ChunkingAgent:
    """Agent to chunk preprocessed documents into smaller units."""

    def __init__(self, config: Dict):
        chunk_cfg = config.get("chunking", {})
        self.logger = get_logger(__name__)
        
        self.chunk_size = chunk_cfg.get("chunk_size", DEFAULT_CHUNK_SIZE)
        self.chunk_overlap = chunk_cfg.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)
        self.min_chunk_size = chunk_cfg.get("min_chunk_size", MIN_CHUNK_SIZE)
        self.use_section_based = chunk_cfg.get("use_section_based", False)
        self.separators = chunk_cfg.get("separators", None)
        
        self.text_chunker = TextChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
        )
        self.section_chunker = SectionChunker()

    def process_document(self, text: str, doc_id: str, metadata: Dict) -> List[Dict]:
        """Chunk a single document.

        Args:
            text: 문서 텍스트
            doc_id: 문서 ID (파일명 등)
            metadata: 메타데이터 딕셔너리
        """
        # doc_id를 메타데이터에도 넣어두기
        metadata = {**metadata, "doc_id": doc_id}

        if self.use_section_based:
            try:
                sections = self.section_chunker.detect_sections(text)
                if sections:
                    return self.section_chunker.chunk_by_sections(
                        text=text,
                        sections=sections,
                        base_metadata=metadata,
                        text_chunker=self.text_chunker,
                    )
            except Exception as e:
                self.logger.warning(f"Section-based chunking failed, fallback to basic: {e}")

        # 기본 청킹
        return self.text_chunker.chunk(text, doc_id=doc_id, metadata=metadata)

    def process_batch(self, input_dir: str, output_path: str) -> None:
        """Process all preprocessed JSON files and write chunks to JSONL.

        Args:
            input_dir: Ingest 결과 JSON들이 있는 디렉토리 (예: data/preprocessed)
            output_path: 청킹 결과 JSONL 경로 (예: data/features/chunks.jsonl)
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        json_files = sorted(input_path.glob("*.json"))
        if not json_files:
            self.logger.warning(f"No JSON files found in {input_dir}. Run ingest step first.")
            return

        all_chunks: List[Dict] = []
        total_lengths: List[int] = []

        self.logger.info(f"Found {len(json_files)} preprocessed files to chunk")

        for fp in json_files:
            try:
                data = load_json(str(fp))
                text = data.get("text", "")
                meta = data.get("metadata", {}) or {}

                # doc_id 우선순위: 메타데이터 공고번호 -> 파일명
                doc_id = str(meta.get("공고 번호") or fp.stem)

                chunks = self.process_document(text, doc_id=doc_id, metadata=meta)
                all_chunks.extend(chunks)
                total_lengths.extend(len(c["chunk_text"]) for c in chunks)

            except Exception as e:
                self.logger.error(f"Failed to chunk file {fp}: {e}")
                continue

        if not all_chunks:
            self.logger.warning("No chunks were created. Check input data.")
            return

        # Save to JSONL
        save_jsonl(all_chunks, output_path)

        # Stats
        total_chunks = len(all_chunks)
        avg_len = sum(total_lengths) / total_chunks if total_chunks else 0
        min_len = min(total_lengths) if total_lengths else 0
        max_len = max(total_lengths) if total_lengths else 0

        self.logger.info("=" * 60)
        self.logger.info("Chunking Summary")
        self.logger.info("=" * 60)
        self.logger.info(f"Input dir: {input_dir}")
        self.logger.info(f"Output file: {output_path}")
        self.logger.info(f"Total chunks: {total_chunks}")
        self.logger.info(f"Avg chunk length: {avg_len:.1f}")
        self.logger.info(f"Min chunk length: {min_len}")
        self.logger.info(f"Max chunk length: {max_len}")
