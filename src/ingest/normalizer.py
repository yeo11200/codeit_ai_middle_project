"""Text normalization module."""

import re
from typing import Optional


class TextNormalizer:
    """Normalize extracted text from documents."""
    
    def normalize(self, text: str) -> str:
        """
        Normalize text according to specified rules.
        
        Args:
            text: Input text to normalize
        
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # 1. Remove BOM
        text = text.lstrip("\ufeff")
        
        # 2. Convert tabs to spaces
        text = text.replace("\t", " ")
        
        # 3. Normalize consecutive spaces (preserve newlines)
        # Split by newlines, normalize each line, then rejoin
        lines = text.split("\n")
        normalized_lines = []
        for line in lines:
            # Replace multiple spaces with single space
            line = re.sub(r" +", " ", line)
            normalized_lines.append(line)
        text = "\n".join(normalized_lines)
        
        # 4. Limit excessive newlines (more than 2 consecutive)
        text = re.sub(r"\n{3,}", "\n\n", text)
        
        # 5. Remove special control characters but preserve basic punctuation
        # Keep: Korean, English, numbers, basic punctuation (. , ! ? : ; - _ ' " ( ) [ ])
        text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", text)
        
        return text.strip()

