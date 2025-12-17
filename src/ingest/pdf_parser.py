"""PDF parser module."""

from pathlib import Path
from typing import Dict
import pypdf
from src.common.logger import get_logger
from src.common.constants import MIN_TEXT_LENGTH


class PDFParser:
    """Parse PDF files and extract text."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def parse(self, file_path: str) -> Dict:
        """
        Parse PDF file and extract text.
        
        Args:
            file_path: Path to PDF file
        
        Returns:
            Dictionary with text, page_count, is_image_based, metadata
        """
        pdf_path = Path(file_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        try:
            # Extract text
            text = self._extract_text(str(pdf_path))
            
            # Get page count
            page_count = self._get_page_count(str(pdf_path))
            
            # Detect if image-based
            is_image_based = self._detect_image_based(text)
            
            if is_image_based:
                self.logger.warning(
                    f"PDF appears to be image-based: {file_path}. "
                    "OCR may be needed for better extraction."
                )
            
            # Validate text length
            if len(text.strip()) < MIN_TEXT_LENGTH:
                self.logger.warning(
                    f"Extracted text is too short ({len(text)} chars) "
                    f"for file: {file_path}"
                )
            
            return {
                "text": text,
                "page_count": page_count,
                "is_image_based": is_image_based,
                "metadata": {
                    "file_path": str(pdf_path),
                    "file_size": pdf_path.stat().st_size,
                }
            }
        
        except Exception as e:
            self.logger.error(f"Failed to parse PDF {file_path}: {e}")
            raise ValueError(f"PDF parsing failed: {e}")
    
    def _extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF using PyPDF."""
        text_parts = []
        
        try:
            with open(pdf_path, "rb") as file:
                pdf_reader = pypdf.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to extract text from page {page_num}: {e}"
                        )
                        continue
            
            text = "\n".join(text_parts)
            return text
        
        except Exception as e:
            self.logger.error(f"Error reading PDF {pdf_path}: {e}")
            raise
    
    def _get_page_count(self, pdf_path: str) -> int:
        """Get number of pages in PDF."""
        try:
            with open(pdf_path, "rb") as file:
                pdf_reader = pypdf.PdfReader(file)
                return len(pdf_reader.pages)
        except Exception:
            return 0
    
    def _detect_image_based(self, text: str) -> bool:
        """
        Detect if PDF is image-based (scanned).
        
        Simple heuristic: if text is very short relative to file size,
        it's likely image-based.
        """
        if not text or len(text.strip()) < 50:
            return True
        
        # If text length is less than 100 characters, likely image-based
        if len(text.strip()) < 100:
            return True
        
        return False
    
    def _extract_with_ocr(self, pdf_path: str) -> str:
        """
        Extract text using OCR (optional, not implemented by default).
        
        This would require pdf2image and pytesseract.
        """
        self.logger.warning("OCR extraction not implemented. Install pdf2image and pytesseract.")
        return ""

