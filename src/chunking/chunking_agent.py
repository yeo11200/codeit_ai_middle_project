"""Chunking Agent - split preprocessed documents into chunks."""

from pathlib import Path
from typing import Dict, List

from src.chunking.chunker import TextChunker
from src.chunking.section_chunker import SectionChunker
from src.chunking.optimized_chunker import OptimizedChunker
from src.chunking.chunk_summarizer import ChunkSummarizer
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
        self.optimized_chunker = OptimizedChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            min_chunk_size=self.min_chunk_size,
        )
        self.section_chunker = SectionChunker()
        
        # Use optimized chunker by default (can be disabled)
        self.use_optimized = chunk_cfg.get("use_optimized", True)
        
        # Chunk summarizer for file-based JSON output
        self.summarizer = ChunkSummarizer()
        self.save_file_summaries = chunk_cfg.get("save_file_summaries", True)
        self.summary_output_dir = chunk_cfg.get("summary_output_dir", "data/features/summaries")

    def process_document(self, text: str, doc_id: str, metadata: Dict) -> List[Dict]:
        """Chunk a single document.

        Args:
            text: 문서 텍스트
            doc_id: 문서 ID (파일명 등)
            metadata: 메타데이터 딕셔너리
        """
        # doc_id를 메타데이터에도 넣어두기
        metadata = {**metadata, "doc_id": doc_id}
        
        # Get file type from metadata
        file_type = None
        if "file_path" in metadata:
            file_path = metadata["file_path"]
            if file_path:
                ext = Path(file_path).suffix.lower().replace(".", "").upper()
                if ext in ["PDF", "HWP", "DOCX"]:
                    file_type = ext

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
                self.logger.warning(f"Section-based chunking failed, fallback to optimized: {e}")

        # Use optimized chunker (default) or basic chunker
        if self.use_optimized:
            return self.optimized_chunker.chunk(
                text=text,
                doc_id=doc_id,
                metadata=metadata,
                file_type=file_type
            )
        else:
            return self.text_chunker.chunk(text, doc_id=doc_id, metadata=metadata)

    def process_batch(
        self,
        input_dir: str,
        output_path: str,
        save_summaries: bool = True
    ) -> None:
        """Process all preprocessed JSON files and write chunks to JSONL.

        Args:
            input_dir: Ingest 결과 JSON들이 있는 디렉토리 (예: data/preprocessed)
            output_path: 청킹 결과 JSONL 경로 (예: data/features/chunks.jsonl)
            save_summaries: Save file-based JSON summaries (default: True)
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        json_files = sorted(input_path.glob("*.json"))
        if not json_files:
            error_msg = (
                f"No JSON files found in {input_dir}. "
                "Please run ingest step first to generate preprocessed JSON files."
            )
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        all_chunks: List[Dict] = []
        total_lengths: List[int] = []
        chunks_by_file: Dict[str, List[Dict]] = {}  # Track chunks by file

        self.logger.info(f"Found {len(json_files)} preprocessed files to chunk")

        processed_count = 0
        empty_text_count = 0
        error_count = 0
        
        for fp in json_files:
            try:
                data = load_json(str(fp))
                text = data.get("text", "")
                meta = data.get("metadata", {}) or {}

                # 텍스트가 비어있는지 확인
                if not text or not text.strip():
                    empty_text_count += 1
                    if empty_text_count <= 3:  # 처음 3개만 로그
                        self.logger.warning(
                            f"Empty or missing text in {fp.name}. "
                            f"Keys in JSON: {list(data.keys())}"
                        )
                    continue

                # doc_id 우선순위: 메타데이터 공고번호 -> 파일명
                doc_id = str(meta.get("공고 번호") or fp.stem)

                chunks = self.process_document(text, doc_id=doc_id, metadata=meta)
                
                if not chunks:
                    self.logger.warning(f"No chunks created for {fp.name} (text length: {len(text)})")
                    continue
                
                all_chunks.extend(chunks)
                total_lengths.extend(len(c["chunk_text"]) for c in chunks)
                chunks_by_file[doc_id] = chunks  # Store chunks by file
                processed_count += 1

            except Exception as e:
                error_count += 1
                self.logger.error(f"Failed to chunk file {fp}: {e}", exc_info=True)
                continue

        # 통계 로그
        self.logger.info(f"Processed files: {processed_count}/{len(json_files)}")
        self.logger.info(f"Files with empty text: {empty_text_count}")
        self.logger.info(f"Files with errors: {error_count}")
        
        if not all_chunks:
            error_msg = (
                f"No chunks were created from {len(json_files)} files. "
                f"Processed: {processed_count}, Empty text: {empty_text_count}, Errors: {error_count}. "
                "Check if input JSON files contain valid 'text' field with non-empty content."
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Save to JSONL
        save_jsonl(all_chunks, output_path)
        self.logger.info(f"Saved {len(all_chunks)} chunks to JSONL: {output_path}")

        # Create and save file-based summaries
        if save_summaries and self.save_file_summaries and chunks_by_file:
            self.logger.info("Creating file-based summaries...")
            file_summaries = self.summarizer.summarize_chunks_by_file(all_chunks)
            saved_files = self.summarizer.save_summaries_to_files(
                file_summaries=file_summaries,
                output_dir=self.summary_output_dir,
                save_individual=True,
                save_combined=True
            )
            self.logger.info(f"File summaries saved to: {self.summary_output_dir}")
            
            # Also save summary-only version (without chunk texts)
            summary_only = self.summarizer.create_summary_only(file_summaries)
            summary_only_path = Path(self.summary_output_dir) / "chunks_summary_only.json"
            import json
            with open(summary_only_path, "w", encoding="utf-8") as f:
                json.dump(summary_only, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Summary-only file saved to: {summary_only_path}")

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
        self.logger.info(f"Total files processed: {processed_count}")
        self.logger.info(f"Total chunks: {total_chunks}")
        self.logger.info(f"Avg chunk length: {avg_len:.1f}")
        self.logger.info(f"Min chunk length: {min_len}")
        self.logger.info(f"Max chunk length: {max_len}")
        if save_summaries and self.save_file_summaries:
            self.logger.info(f"File summaries: {self.summary_output_dir}")
