"""Metadata filtering module."""

from typing import Dict, Optional, List
from rapidfuzz import fuzz, process

from src.common.logger import get_logger


class MetadataFilter:
    """Build and apply metadata filters for search."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def build_filter(
        self,
        organization: Optional[str] = None,
        business_name: Optional[str] = None,
        amount_min: Optional[float] = None,
        amount_max: Optional[float] = None,
        deadline_start: Optional[str] = None,
        deadline_end: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Build ChromaDB where clause from filter parameters.
        
        Args:
            organization: 발주 기관명
            business_name: 사업명
            amount_min: 최소 금액
            amount_max: 최대 금액
            deadline_start: 마감일 시작 (ISO format)
            deadline_end: 마감일 끝 (ISO format)
        
        Returns:
            ChromaDB where clause dict or None
        """
        where_clause = {}
        
        if organization:
            # Note: ChromaDB metadata filtering requires exact match or $in
            # For fuzzy matching, we'd need to pre-filter the organization list
            where_clause["발주 기관"] = {"$eq": organization}
        
        if business_name:
            where_clause["사업명"] = {"$eq": business_name}
        
        if amount_min is not None or amount_max is not None:
            amount_filter = {}
            if amount_min is not None:
                amount_filter["$gte"] = amount_min
            if amount_max is not None:
                amount_filter["$lte"] = amount_max
            if amount_filter:
                where_clause["사업 금액"] = amount_filter
        
        if deadline_start or deadline_end:
            deadline_filter = {}
            if deadline_start:
                deadline_filter["$gte"] = deadline_start
            if deadline_end:
                deadline_filter["$lte"] = deadline_end
            if deadline_filter:
                where_clause["입찰 참여 마감일"] = deadline_filter
        
        return where_clause if where_clause else None
    
    def fuzzy_match_organization(
        self,
        query: str,
        organizations: List[str],
        threshold: float = 0.8
    ) -> List[str]:
        """
        Find organizations matching query using fuzzy matching.
        
        Args:
            query: Query string
            organizations: List of organization names
            threshold: Minimum similarity score (0-1)
        
        Returns:
            List of matching organization names
        """
        if not organizations:
            return []
        
        # Use rapidfuzz to find matches
        matches = process.extract(
            query,
            organizations,
            scorer=fuzz.ratio,
            limit=10
        )
        
        # Filter by threshold
        result = [
            match[0]
            for match in matches
            if match[1] / 100.0 >= threshold
        ]
        
        return result
    
    def keyword_match_business_name(
        self,
        query: str,
        business_names: List[str]
    ) -> List[str]:
        """
        Find business names containing query keywords.
        
        Args:
            query: Query string
            business_names: List of business names
        
        Returns:
            List of matching business names
        """
        if not business_names:
            return []
        
        query_lower = query.lower()
        keywords = query_lower.split()
        
        matches = []
        for name in business_names:
            name_lower = name.lower()
            # Check if all keywords are in the name
            if all(keyword in name_lower for keyword in keywords):
                matches.append(name)
        
        return matches

