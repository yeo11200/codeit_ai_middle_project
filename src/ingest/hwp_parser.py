"""HWP parser module."""

from pathlib import Path
from typing import Dict
import olefile
from src.common.logger import get_logger


class HWPParser:
    """Parse HWP files and extract text."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def parse(self, file_path: str) -> Dict:
        """
        Parse HWP file and extract text.
        
        Args:
            file_path: Path to HWP file
        
        Returns:
            Dictionary with text and metadata
        """
        hwp_path = Path(file_path)
        
        if not hwp_path.exists():
            raise FileNotFoundError(f"HWP file not found: {file_path}")
        
        try:
            # Try to extract text using olefile
            text = self._extract_text(str(hwp_path))
            metadata = self._extract_metadata(str(hwp_path))
            
            if not text or len(text.strip()) < 10:
                self.logger.warning(
                    f"Extracted text from HWP is very short or empty: {file_path}"
                )
            
            return {
                "text": text,
                "metadata": {
                    "file_path": str(hwp_path),
                    "file_size": hwp_path.stat().st_size,
                    **metadata
                }
            }
        
        except Exception as e:
            self.logger.error(f"Failed to parse HWP {file_path}: {e}")
            # Return empty text instead of raising error
            return {
                "text": "",
                "metadata": {
                    "file_path": str(hwp_path),
                    "error": str(e)
                }
            }
    
    def _extract_text(self, hwp_path: str) -> str:
        """
        Extract text from HWP file.
        
        Note: HWP parsing is complex. This is a basic implementation.
        For better results, consider using pyhwp or other specialized libraries.
        """
        text_parts = []
        
        try:
            if not olefile.isOleFile(hwp_path):
                self.logger.warning(f"File is not a valid OLE file: {hwp_path}")
                return ""
            
            ole = olefile.OleFileIO(hwp_path)
            
            # Try to read BodyText stream (common in HWP files)
            try:
                if ole.exists("BodyText"):
                    stream = ole.openstream("BodyText")
                    data = stream.read()
                    # Basic text extraction (HWP format is complex)
                    # This is a simplified approach
                    try:
                        text = data.decode("utf-8", errors="ignore")
                        # Filter out non-printable characters
                        text = "".join(c for c in text if c.isprintable() or c in "\n\r\t")
                        text_parts.append(text)
                    except Exception:
                        pass
            except Exception as e:
                self.logger.debug(f"Could not read BodyText stream: {e}")
            
            ole.close()
            
            return "\n".join(text_parts)
        
        except Exception as e:
            self.logger.warning(f"Error reading HWP {hwp_path}: {e}")
            return ""
    
    def _extract_metadata(self, hwp_path: str) -> Dict:
        """Extract metadata from HWP file."""
        metadata = {}
        
        try:
            if olefile.isOleFile(hwp_path):
                ole = olefile.OleFileIO(hwp_path)
                
                # Try to get summary information
                if ole.exists("\x05SummaryInformation"):
                    try:
                        summary = ole.get_metadata()
                        if summary:
                            metadata.update({
                                "title": summary.title or "",
                                "author": summary.author or "",
                                "subject": summary.subject or "",
                            })
                    except Exception:
                        pass
                
                ole.close()
        
        except Exception as e:
            self.logger.debug(f"Could not extract metadata: {e}")
        
        return metadata

