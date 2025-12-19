"""Metadata analyzer for EDA."""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime
from src.common.logger import get_logger
from src.eda.data_loader import DataLoader


class MetadataAnalyzer:
    """Analyze metadata from CSV and documents."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.data_loader = DataLoader()
    
    def analyze_basic_stats(self, df: pd.DataFrame) -> Dict:
        """
        Analyze basic statistics.
        
        Args:
            df: DataFrame with metadata
        
        Returns:
            Dictionary with basic statistics
        """
        if df.empty:
            return {}
        
        # Get file types from file paths
        if "파일명" in df.columns:
            df["file_type"] = df["파일명"].apply(
                lambda x: self.data_loader.get_file_type_from_path(str(x)) if pd.notna(x) else "UNKNOWN"
            )
        else:
            df["file_type"] = "UNKNOWN"
        
        stats = {
            "total_documents": len(df),
            "file_type_distribution": df["file_type"].value_counts().to_dict() if "file_type" in df.columns else {},
            "unique_institutions": df["발주 기관"].nunique() if "발주 기관" in df.columns else 0,
            "unique_announcements": df["공고 번호"].nunique() if "공고 번호" in df.columns else 0,
        }
        
        return stats
    
    def analyze_budget_distribution(self, df: pd.DataFrame) -> Dict:
        """
        Analyze budget distribution.
        
        Args:
            df: DataFrame with metadata
        
        Returns:
            Dictionary with budget statistics
        """
        if df.empty or "사업 금액" not in df.columns:
            return {}
        
        # Convert to numeric, handling non-numeric values
        budget_series = pd.to_numeric(df["사업 금액"], errors="coerce")
        budget_series = budget_series.dropna()
        
        if budget_series.empty:
            return {}
        
        # Basic statistics
        stats = {
            "mean": float(budget_series.mean()),
            "median": float(budget_series.median()),
            "min": float(budget_series.min()),
            "max": float(budget_series.max()),
            "std": float(budget_series.std()),
            "q25": float(budget_series.quantile(0.25)),
            "q75": float(budget_series.quantile(0.75)),
        }
        
        # Budget ranges (in 원)
        ranges = {
            "under_100m": int((budget_series < 100_000_000).sum()),
            "100m_to_500m": int(((budget_series >= 100_000_000) & (budget_series < 500_000_000)).sum()),
            "500m_to_1b": int(((budget_series >= 500_000_000) & (budget_series < 1_000_000_000)).sum()),
            "over_1b": int((budget_series >= 1_000_000_000).sum()),
        }
        
        # Institution average budget
        institution_avg = {}
        if "발주 기관" in df.columns:
            df_with_budget = df[df["사업 금액"].notna()].copy()
            df_with_budget["사업 금액"] = pd.to_numeric(df_with_budget["사업 금액"], errors="coerce")
            institution_avg = df_with_budget.groupby("발주 기관")["사업 금액"].mean().to_dict()
            # Convert to float for JSON serialization
            institution_avg = {k: float(v) for k, v in institution_avg.items()}
        
        return {
            "statistics": stats,
            "ranges": ranges,
            "institution_averages": institution_avg,
        }
    
    def analyze_temporal_distribution(self, df: pd.DataFrame) -> Dict:
        """
        Analyze temporal distribution.
        
        Args:
            df: DataFrame with metadata
        
        Returns:
            Dictionary with temporal statistics
        """
        result = {}
        
        # Parse dates
        date_columns = {
            "공개 일자": "publication_date",
            "입찰 참여 마감일": "deadline_date",
        }
        
        for col_kr, col_en in date_columns.items():
            if col_kr not in df.columns:
                continue
            
            try:
                # Try to parse dates
                dates = pd.to_datetime(df[col_kr], errors="coerce", format="%Y-%m-%d %H:%M:%S")
                dates = dates.dropna()
                
                if dates.empty:
                    continue
                
                result[col_en] = {
                    "year_distribution": dates.dt.year.value_counts().to_dict(),
                    "month_distribution": dates.dt.month.value_counts().to_dict(),
                    "day_of_week_distribution": dates.dt.dayofweek.value_counts().to_dict(),
                }
            except Exception as e:
                self.logger.warning(f"Failed to parse {col_kr}: {e}")
        
        # Calculate deadline period
        if "공개 일자" in df.columns and "입찰 참여 마감일" in df.columns:
            try:
                pub_dates = pd.to_datetime(df["공개 일자"], errors="coerce")
                deadline_dates = pd.to_datetime(df["입찰 참여 마감일"], errors="coerce")
                
                # Calculate days between
                periods = (deadline_dates - pub_dates).dt.days
                periods = periods.dropna()
                
                if not periods.empty:
                    result["deadline_period"] = {
                        "mean_days": float(periods.mean()),
                        "median_days": float(periods.median()),
                        "min_days": int(periods.min()),
                        "max_days": int(periods.max()),
                    }
            except Exception as e:
                self.logger.warning(f"Failed to calculate deadline period: {e}")
        
        return result
    
    def analyze_institution_distribution(self, df: pd.DataFrame) -> Dict:
        """
        Analyze institution distribution.
        
        Args:
            df: DataFrame with metadata
        
        Returns:
            Dictionary with institution statistics
        """
        if df.empty or "발주 기관" not in df.columns:
            return {}
        
        institution_counts = df["발주 기관"].value_counts()
        top_10 = institution_counts.head(10).to_dict()
        
        # Institution statistics
        institution_stats = {}
        for inst in top_10.keys():
            inst_df = df[df["발주 기관"] == inst]
            stats = {
                "count": int(len(inst_df)),
                "total_budget": 0.0,
                "avg_budget": 0.0,
            }
            
            if "사업 금액" in inst_df.columns:
                budgets = pd.to_numeric(inst_df["사업 금액"], errors="coerce").dropna()
                if not budgets.empty:
                    stats["total_budget"] = float(budgets.sum())
                    stats["avg_budget"] = float(budgets.mean())
            
            institution_stats[inst] = stats
        
        return {
            "top_10_institutions": top_10,
            "institution_statistics": institution_stats,
        }
    
    def detect_data_quality_issues(self, df: pd.DataFrame) -> Dict:
        """
        Detect data quality issues.
        
        Args:
            df: DataFrame with metadata
        
        Returns:
            Dictionary with data quality issues
        """
        issues = {
            "missing_values": {},
            "duplicates": {},
            "outliers": {},
            "naming_inconsistencies": [],
        }
        
        if df.empty:
            return issues
        
        # Missing values
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        issues["missing_values"] = {
            col: {
                "count": int(missing[col]),
                "percentage": float(missing_pct[col])
            }
            for col in df.columns
            if missing[col] > 0
        }
        
        # Duplicate announcement numbers
        if "공고 번호" in df.columns:
            duplicates = df[df["공고 번호"].duplicated(keep=False)]
            if not duplicates.empty:
                issues["duplicates"]["announcement_numbers"] = {
                    "count": int(duplicates["공고 번호"].nunique()),
                    "examples": duplicates["공고 번호"].head(5).tolist(),
                }
        
        # Outliers in budget
        if "사업 금액" in df.columns:
            budgets = pd.to_numeric(df["사업 금액"], errors="coerce").dropna()
            if not budgets.empty:
                q1 = budgets.quantile(0.25)
                q3 = budgets.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = budgets[(budgets < lower_bound) | (budgets > upper_bound)]
                if not outliers.empty:
                    issues["outliers"]["budget"] = {
                        "count": int(len(outliers)),
                        "min_outlier": float(outliers.min()),
                        "max_outlier": float(outliers.max()),
                    }
        
        # Naming inconsistencies (simple check for institution names)
        if "발주 기관" in df.columns:
            institutions = df["발주 기관"].dropna().unique()
            # Check for similar names (basic check)
            # This is a simplified version - could be improved with fuzzy matching
            seen = set()
            for inst in institutions:
                # Normalize: remove spaces, convert to lowercase
                normalized = inst.replace(" ", "").lower()
                if normalized in seen:
                    issues["naming_inconsistencies"].append(inst)
                seen.add(normalized)
        
        return issues

