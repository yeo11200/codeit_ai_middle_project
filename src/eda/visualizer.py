"""Visualization tools for EDA."""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional
from src.common.logger import get_logger
from src.common.utils import ensure_dir

# Set Korean font
try:
    plt.rcParams['font.family'] = 'AppleGothic'  # Mac
except:
    try:
        plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
    except:
        try:
            plt.rcParams['font.family'] = 'NanumGothic'  # Linux
        except:
            pass

plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display


class Visualizer:
    """Create visualizations for EDA results."""
    
    def __init__(self, output_dir: str = "data/eda/reports/figures"):
        self.logger = get_logger(__name__)
        self.output_dir = Path(output_dir)
        ensure_dir(str(self.output_dir))
    
    def plot_budget_distribution(self, stats: Dict, output_path: Optional[str] = None) -> str:
        """
        Plot budget distribution.
        
        Args:
            stats: Budget statistics dictionary
            output_path: Optional output path
        
        Returns:
            Path to saved figure
        """
        if not stats or "statistics" not in stats:
            return ""
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        if "ranges" in stats:
            ranges = stats["ranges"]
            labels = ["1억 미만", "1-5억", "5-10억", "10억 이상"]
            values = [
                ranges.get("under_100m", 0),
                ranges.get("100m_to_500m", 0),
                ranges.get("500m_to_1b", 0),
                ranges.get("over_1b", 0),
            ]
            axes[0].bar(labels, values)
            axes[0].set_title("예산 구간별 분포")
            axes[0].set_ylabel("사업 수")
            axes[0].tick_params(axis='x', rotation=45)
        
        # Box plot data (if available)
        if "statistics" in stats:
            stats_data = stats["statistics"]
            # Create box plot data
            box_data = [
                stats_data.get("min", 0),
                stats_data.get("q25", 0),
                stats_data.get("median", 0),
                stats_data.get("q75", 0),
                stats_data.get("max", 0),
            ]
            axes[1].boxplot([box_data], labels=["사업 금액"])
            axes[1].set_title("예산 분포 (박스플롯)")
            axes[1].set_ylabel("금액 (원)")
        
        plt.tight_layout()
        
        if output_path is None:
            output_path = self.output_dir / "budget_distribution.png"
        else:
            output_path = Path(output_path)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved budget distribution plot: {output_path}")
        return str(output_path)
    
    def plot_temporal_distribution(self, stats: Dict, output_path: Optional[str] = None) -> str:
        """
        Plot temporal distribution.
        
        Args:
            stats: Temporal statistics dictionary
            output_path: Optional output path
        
        Returns:
            Path to saved figure
        """
        if not stats:
            return ""
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Year distribution
        if "publication_date" in stats and "year_distribution" in stats["publication_date"]:
            year_data = stats["publication_date"]["year_distribution"]
            years = sorted(year_data.keys())
            counts = [year_data[y] for y in years]
            axes[0].bar(years, counts)
            axes[0].set_title("연도별 공개 일자 분포")
            axes[0].set_xlabel("연도")
            axes[0].set_ylabel("사업 수")
        
        # Month distribution
        if "publication_date" in stats and "month_distribution" in stats["publication_date"]:
            month_data = stats["publication_date"]["month_distribution"]
            months = sorted(month_data.keys())
            counts = [month_data[m] for m in months]
            axes[1].bar(months, counts)
            axes[1].set_title("월별 공개 일자 분포")
            axes[1].set_xlabel("월")
            axes[1].set_ylabel("사업 수")
        
        plt.tight_layout()
        
        if output_path is None:
            output_path = self.output_dir / "temporal_distribution.png"
        else:
            output_path = Path(output_path)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved temporal distribution plot: {output_path}")
        return str(output_path)
    
    def plot_file_type_distribution(self, stats: Dict, output_path: Optional[str] = None) -> str:
        """
        Plot file type distribution.
        
        Args:
            stats: File type statistics
            output_path: Optional output path
        
        Returns:
            Path to saved figure
        """
        if not stats or "file_type_distribution" not in stats:
            return ""
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        file_types = stats["file_type_distribution"]
        
        # Pie chart
        labels = list(file_types.keys())
        sizes = list(file_types.values())
        axes[0].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        axes[0].set_title("파일 형식 분포")
        
        # Bar chart
        axes[1].bar(labels, sizes)
        axes[1].set_title("파일 형식별 문서 수")
        axes[1].set_ylabel("문서 수")
        
        plt.tight_layout()
        
        if output_path is None:
            output_path = self.output_dir / "file_type_distribution.png"
        else:
            output_path = Path(output_path)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved file type distribution plot: {output_path}")
        return str(output_path)
    
    def plot_text_length_distribution(self, stats: Dict, output_path: Optional[str] = None) -> str:
        """
        Plot text length distribution.
        
        Args:
            stats: Text length statistics
            output_path: Optional output path
        
        Returns:
            Path to saved figure
        """
        if not stats or "statistics" not in stats:
            return ""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # This would need actual data, but for now we'll create a placeholder
        # In real implementation, you'd pass the actual length data
        ax.set_title("텍스트 길이 분포")
        ax.set_xlabel("텍스트 길이 (자)")
        ax.set_ylabel("빈도")
        
        # If we have average by file type
        if "average_by_file_type" in stats:
            file_types = list(stats["average_by_file_type"].keys())
            averages = [stats["average_by_file_type"][ft] for ft in file_types]
            ax.bar(file_types, averages)
            ax.set_title("파일 형식별 평균 텍스트 길이")
            ax.set_ylabel("평균 길이 (자)")
        
        plt.tight_layout()
        
        if output_path is None:
            output_path = self.output_dir / "text_length_distribution.png"
        else:
            output_path = Path(output_path)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved text length distribution plot: {output_path}")
        return str(output_path)
    
    def plot_institution_analysis(self, stats: Dict, output_path: Optional[str] = None) -> str:
        """
        Plot institution analysis.
        
        Args:
            stats: Institution statistics
            output_path: Optional output path
        
        Returns:
            Path to saved figure
        """
        if not stats or "top_10_institutions" not in stats:
            return ""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        top_10 = stats["top_10_institutions"]
        institutions = list(top_10.keys())
        counts = list(top_10.values())
        
        ax.barh(institutions, counts)
        ax.set_title("상위 10개 발주 기관 (사업 수)")
        ax.set_xlabel("사업 수")
        
        plt.tight_layout()
        
        if output_path is None:
            output_path = self.output_dir / "institution_analysis.png"
        else:
            output_path = Path(output_path)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved institution analysis plot: {output_path}")
        return str(output_path)

