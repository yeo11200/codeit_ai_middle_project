"""Ingest Agent - Document parsing and preprocessing."""

from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm

from src.ingest.pdf_parser import PDFParser
from src.ingest.hwp_parser import HWPParser
from src.ingest.normalizer import TextNormalizer
from src.ingest.metadata_loader import MetadataLoader
from src.common.logger import get_logger
from src.common.utils import ensure_dir, save_json, get_file_extension
from src.common.constants import (
    SUPPORTED_PDF_EXTENSIONS,
    SUPPORTED_HWP_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    STATUS_SUCCESS,
    STATUS_FAILED,
    STATUS_WARNING,
)


class IngestAgent:
    """Main agent for document ingestion and preprocessing."""
    
    def __init__(self, config: Dict):
        """
        Initialize Ingest Agent.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get("ingest", {})
        self.pdf_parser = PDFParser()
        self.hwp_parser = HWPParser()
        self.normalizer = TextNormalizer()
        self.metadata_loader = MetadataLoader()
        self.logger = get_logger(__name__)
        
        # Load metadata if CSV path provided
        csv_path = self.config.get("metadata_csv")
        if csv_path:
            self.metadata_loader.load_from_csv(csv_path)
    
    def process_file(
        self,
        file_path: str,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Process a single file.
        
        Args:
            file_path: Path to file
            metadata: Optional metadata dictionary
        
        Returns:
            Dictionary with processed text and metadata
        """
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Detect file type
        file_ext = get_file_extension(file_path)
        file_type = file_ext.lower()
        
        # Get metadata
        if metadata is None:
            filename = file_path_obj.name
            metadata = self.metadata_loader.get_metadata(filename)
        
        # Select parser based on file type
        parser_result = None
        status = STATUS_SUCCESS
        
        try:
            if file_ext in SUPPORTED_PDF_EXTENSIONS:
                parser_result = self.pdf_parser.parse(file_path)
            elif file_ext in SUPPORTED_HWP_EXTENSIONS:
                parser_result = self.hwp_parser.parse(file_path)
            else:
                raise ValueError(
                    f"Unsupported file format: {file_ext}. "
                    f"Supported: {SUPPORTED_EXTENSIONS}"
                )
            
            # Extract text
            text = parser_result.get("text", "")
            
            # Normalize text
            normalized_text = self.normalizer.normalize(text)
            
            # Validate minimum text length
            min_length = self.config.get("min_text_length", 100)
            if len(normalized_text.strip()) < min_length:
                self.logger.warning(
                    f"Text length ({len(normalized_text)}) is below minimum "
                    f"({min_length}) for file: {file_path}"
                )
                status = STATUS_WARNING
            
            # Merge metadata
            result_metadata = {
                "file_path": str(file_path),
                "file_name": file_path_obj.name,
                "file_type": file_type,
                **metadata,
            }
            
            # Add parser-specific metadata
            if "metadata" in parser_result:
                result_metadata.update(parser_result["metadata"])
            
            return {
                "text": normalized_text,
                "metadata": result_metadata,
                "file_path": str(file_path),
                "file_type": file_type,
                "status": status,
            }
        
        except Exception as e:
            self.logger.error(f"Failed to process file {file_path}: {e}")
            return {
                "text": "",
                "metadata": metadata or {},
                "file_path": str(file_path),
                "file_type": file_type,
                "status": STATUS_FAILED,
                "error": str(e),
            }
    
    def process_batch(
        self,
        input_dir: str,
        output_dir: str,
        csv_path: Optional[str] = None
    ):
        """
        Process batch of files.
        
        Args:
            input_dir: Input directory containing files
            output_dir: Output directory for processed files
            csv_path: Optional CSV path for metadata (if not in config)
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        ensure_dir(str(output_path))
        
        # Load metadata if provided
        if csv_path:
            self.metadata_loader.load_from_csv(csv_path)
        
        # Find all supported files
        files = []
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(input_path.glob(f"*{ext}"))
        
        if not files:
            self.logger.warning(f"No supported files found in {input_dir}")
            return
        
        self.logger.info(f"Found {len(files)} files to process")
        
        # Process files with progress bar
        success_count = 0
        failed_count = 0
        warning_count = 0
        
        for file_path in tqdm(files, desc="Processing files"):
            try:
                result = self.process_file(str(file_path))
                
                # Save result
                output_file = output_path / f"{file_path.stem}.json"
                save_json(result, str(output_file))
                
                # Count status
                if result["status"] == STATUS_SUCCESS:
                    success_count += 1
                elif result["status"] == STATUS_WARNING:
                    warning_count += 1
                else:
                    failed_count += 1
            
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")
                failed_count += 1
                continue
        
        # Summary
        self.logger.info("=" * 60)
        self.logger.info("Batch Processing Summary")
        self.logger.info("=" * 60)
        self.logger.info(f"Total files: {len(files)}")
        self.logger.info(f"Success: {success_count}")
        self.logger.info(f"Warnings: {warning_count}")
        self.logger.info(f"Failed: {failed_count}")
        self.logger.info(f"Output directory: {output_dir}")

