"""EDA Agent - integrated exploratory data analysis."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from src.common.logger import get_logger
from src.common.utils import ensure_dir, save_json
from src.eda.data_loader import DataLoader
from src.eda.metadata_analyzer import MetadataAnalyzer
from src.eda.text_analyzer import TextAnalyzer
from src.eda.chunk_analyzer import ChunkAnalyzer
from src.eda.visualizer import Visualizer


class EDAAgent:
    """Integrated EDA agent for RFP dataset."""
    
    def __init__(self, config: Dict):
        """
        Initialize EDA Agent.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        self.data_loader = DataLoader()
        self.metadata_analyzer = MetadataAnalyzer()
        self.text_analyzer = TextAnalyzer()
        self.chunk_analyzer = ChunkAnalyzer()
        self.visualizer = None  # Will be initialized with output_dir
    
    def run_full_analysis(
        self,
        output_dir: str = "data/eda/reports",
        analyze_metadata: bool = True,
        analyze_text: bool = True,
        analyze_chunks: bool = True
    ) -> Dict:
        """
        Run full EDA analysis.
        
        Args:
            output_dir: Output directory for reports
            analyze_metadata: Whether to analyze metadata
            analyze_text: Whether to analyze text quality
            analyze_chunks: Whether to analyze chunks
        
        Returns:
            Dictionary with all analysis results
        """
        ensure_dir(output_dir)
        self.visualizer = Visualizer(output_dir=str(Path(output_dir) / "figures"))
        
        self.logger.info("=" * 60)
        self.logger.info("Starting EDA Analysis")
        self.logger.info("=" * 60)
        
        results = {
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metadata_analysis": {},
            "text_analysis": {},
            "chunk_analysis": {},
            "visualizations": {},
        }
        
        # Load data
        self.logger.info("Loading data...")
        config = self.config.get("eda", {})
        
        csv_path = config.get("metadata_csv", "data/data_list.csv")
        json_dir = config.get("preprocessed_dir", "data/preprocessed")
        jsonl_path = config.get("chunks_jsonl", "data/features/chunks.jsonl")
        
        df = self.data_loader.load_metadata_csv(csv_path)
        documents = self.data_loader.load_preprocessed_json(json_dir)
        chunks = self.data_loader.load_chunks_jsonl(jsonl_path)
        
        # Metadata analysis
        if analyze_metadata and not df.empty:
            self.logger.info("Analyzing metadata...")
            results["metadata_analysis"] = {
                "basic_stats": self.metadata_analyzer.analyze_basic_stats(df),
                "budget_distribution": self.metadata_analyzer.analyze_budget_distribution(df),
                "temporal_distribution": self.metadata_analyzer.analyze_temporal_distribution(df),
                "institution_distribution": self.metadata_analyzer.analyze_institution_distribution(df),
                "data_quality_issues": self.metadata_analyzer.detect_data_quality_issues(df),
            }
            
            # Generate visualizations
            if results["metadata_analysis"]["budget_distribution"]:
                results["visualizations"]["budget"] = self.visualizer.plot_budget_distribution(
                    results["metadata_analysis"]["budget_distribution"]
                )
            
            if results["metadata_analysis"]["temporal_distribution"]:
                results["visualizations"]["temporal"] = self.visualizer.plot_temporal_distribution(
                    results["metadata_analysis"]["temporal_distribution"]
                )
            
            if results["metadata_analysis"]["basic_stats"]:
                results["visualizations"]["file_type"] = self.visualizer.plot_file_type_distribution(
                    results["metadata_analysis"]["basic_stats"]
                )
            
            if results["metadata_analysis"]["institution_distribution"]:
                results["visualizations"]["institution"] = self.visualizer.plot_institution_analysis(
                    results["metadata_analysis"]["institution_distribution"]
                )
        
        # Text analysis
        if analyze_text and documents:
            self.logger.info("Analyzing text quality...")
            results["text_analysis"] = {
                "text_lengths": self.text_analyzer.analyze_text_lengths(documents),
                "text_quality": self.text_analyzer.analyze_text_quality(documents),
                "content_patterns": self.text_analyzer.analyze_content_patterns(documents),
                "parsing_errors": self.text_analyzer.analyze_parsing_errors(documents),
            }
            
            # Generate visualization
            if results["text_analysis"]["text_lengths"]:
                results["visualizations"]["text_length"] = self.visualizer.plot_text_length_distribution(
                    results["text_analysis"]["text_lengths"]
                )
        
        # Chunk analysis
        if analyze_chunks and chunks:
            self.logger.info("Analyzing chunks...")
            results["chunk_analysis"] = {
                "chunk_statistics": self.chunk_analyzer.analyze_chunk_statistics(chunks),
                "chunk_distribution": self.chunk_analyzer.analyze_chunk_distribution(chunks),
                "overlap_analysis": self.chunk_analyzer.analyze_overlap_effectiveness(chunks),
                "metadata_coverage": self.chunk_analyzer.analyze_metadata_coverage(chunks),
            }
        
        self.logger.info("=" * 60)
        self.logger.info("EDA Analysis Complete")
        self.logger.info("=" * 60)
        
        return results
    
    def generate_report(self, analyses: Dict, output_path: str) -> None:
        """
        Generate markdown report from analysis results.
        
        Args:
            analyses: Analysis results dictionary
            output_path: Path to save report
        """
        self.logger.info(f"Generating report: {output_path}")
        
        report_lines = []
        report_lines.append("# RFP 데이터셋 EDA 리포트\n")
        report_lines.append(f"**분석 일시**: {analyses.get('analysis_date', 'N/A')}\n")
        report_lines.append("\n---\n")
        
        # Overview
        report_lines.append("## 1. 개요\n")
        report_lines.append("### 데이터셋 정보\n")
        
        if "metadata_analysis" in analyses and "basic_stats" in analyses["metadata_analysis"]:
            basic_stats = analyses["metadata_analysis"]["basic_stats"]
            report_lines.append(f"- 총 문서 수: {basic_stats.get('total_documents', 'N/A')}")
            report_lines.append(f"- 발주 기관 수: {basic_stats.get('unique_institutions', 'N/A')}")
            report_lines.append(f"- 공고 번호 수: {basic_stats.get('unique_announcements', 'N/A')}\n")
        
        report_lines.append("\n---\n")
        
        # Metadata analysis
        if "metadata_analysis" in analyses:
            report_lines.append("## 2. 메타데이터 분석\n")
            
            # Basic stats
            if "basic_stats" in analyses["metadata_analysis"]:
                report_lines.append("### 2.1 기본 통계\n")
                basic_stats = analyses["metadata_analysis"]["basic_stats"]
                report_lines.append(f"- 총 문서 수: {basic_stats.get('total_documents', 'N/A')}")
                report_lines.append(f"- 파일 형식 분포:")
                for file_type, count in basic_stats.get("file_type_distribution", {}).items():
                    report_lines.append(f"  - {file_type}: {count}개")
                report_lines.append("")
            
            # Budget distribution
            if "budget_distribution" in analyses["metadata_analysis"]:
                report_lines.append("### 2.2 예산 분포\n")
                budget = analyses["metadata_analysis"]["budget_distribution"]
                if "statistics" in budget:
                    stats = budget["statistics"]
                    report_lines.append("**통계량**:")
                    report_lines.append(f"- 평균: {stats.get('mean', 0):,.0f}원")
                    report_lines.append(f"- 중앙값: {stats.get('median', 0):,.0f}원")
                    report_lines.append(f"- 최소: {stats.get('min', 0):,.0f}원")
                    report_lines.append(f"- 최대: {stats.get('max', 0):,.0f}원\n")
                
                if "visualizations" in analyses and "budget" in analyses["visualizations"]:
                    report_lines.append(f"![예산 분포]({Path(analyses['visualizations']['budget']).name})\n")
                report_lines.append("")
            
            # Temporal distribution
            if "temporal_distribution" in analyses["metadata_analysis"]:
                report_lines.append("### 2.3 시간 분포\n")
                temporal = analyses["metadata_analysis"]["temporal_distribution"]
                if "publication_date" in temporal and "year_distribution" in temporal["publication_date"]:
                    report_lines.append("**연도별 분포**:")
                    for year, count in sorted(temporal["publication_date"]["year_distribution"].items()):
                        report_lines.append(f"- {year}년: {count}개")
                    report_lines.append("")
                
                if "visualizations" in analyses and "temporal" in analyses["visualizations"]:
                    report_lines.append(f"![시간 분포]({Path(analyses['visualizations']['temporal']).name})\n")
                report_lines.append("")
            
            # Institution analysis
            if "institution_distribution" in analyses["metadata_analysis"]:
                report_lines.append("### 2.4 발주 기관 분석\n")
                inst = analyses["metadata_analysis"]["institution_distribution"]
                if "top_10_institutions" in inst:
                    report_lines.append("**상위 10개 발주 기관**:")
                    for i, (institution, count) in enumerate(inst["top_10_institutions"].items(), 1):
                        report_lines.append(f"{i}. {institution}: {count}개")
                    report_lines.append("")
                
                if "visualizations" in analyses and "institution" in analyses["visualizations"]:
                    report_lines.append(f"![발주 기관 분석]({Path(analyses['visualizations']['institution']).name})\n")
                report_lines.append("")
            
            # Data quality issues
            if "data_quality_issues" in analyses["metadata_analysis"]:
                report_lines.append("### 2.5 데이터 품질 이슈\n")
                issues = analyses["metadata_analysis"]["data_quality_issues"]
                if "missing_values" in issues and issues["missing_values"]:
                    report_lines.append("**결측치**:")
                    for col, info in issues["missing_values"].items():
                        report_lines.append(f"- {col}: {info.get('count', 0)}개 ({info.get('percentage', 0)}%)")
                    report_lines.append("")
                report_lines.append("")
        
        # Text analysis
        if "text_analysis" in analyses:
            report_lines.append("## 3. 텍스트 품질 분석\n")
            
            if "text_lengths" in analyses["text_analysis"]:
                report_lines.append("### 3.1 텍스트 길이 분석\n")
                lengths = analyses["text_analysis"]["text_lengths"]
                if "statistics" in lengths:
                    stats = lengths["statistics"]
                    report_lines.append(f"- 평균: {stats.get('mean', 0):,.0f}자")
                    report_lines.append(f"- 중앙값: {stats.get('median', 0):,.0f}자")
                    report_lines.append(f"- 최소: {stats.get('min', 0):,}자")
                    report_lines.append(f"- 최대: {stats.get('max', 0):,}자\n")
                
                if "visualizations" in analyses and "text_length" in analyses["visualizations"]:
                    report_lines.append(f"![텍스트 길이 분포]({Path(analyses['visualizations']['text_length']).name})\n")
                report_lines.append("")
            
            if "text_quality" in analyses["text_analysis"]:
                report_lines.append("### 3.2 텍스트 품질 지표\n")
                quality = analyses["text_analysis"]["text_quality"]
                report_lines.append(f"- 빈 텍스트 비율: {quality.get('empty_text_ratio', 0)*100:.2f}%")
                report_lines.append(f"- 파싱 실패 비율: {quality.get('parsing_failed_ratio', 0)*100:.2f}%")
                report_lines.append(f"- OCR 필요 비율: {quality.get('ocr_needed_ratio', 0)*100:.2f}%\n")
                report_lines.append("")
        
        # Chunk analysis
        if "chunk_analysis" in analyses:
            report_lines.append("## 4. 청크 분석\n")
            
            if "chunk_statistics" in analyses["chunk_analysis"]:
                report_lines.append("### 4.1 청크 통계\n")
                chunk_stats = analyses["chunk_analysis"]["chunk_statistics"]
                report_lines.append(f"- 총 청크 수: {chunk_stats.get('total_chunks', 'N/A')}")
                report_lines.append(f"- 총 문서 수: {chunk_stats.get('total_documents', 'N/A')}")
                report_lines.append(f"- 문서당 평균 청크 수: {chunk_stats.get('avg_chunks_per_doc', 0):.2f}")
                
                if "chunk_size" in chunk_stats:
                    size_stats = chunk_stats["chunk_size"]
                    report_lines.append(f"- 평균 청크 크기: {size_stats.get('mean', 0):.0f}자")
                    report_lines.append(f"- 중앙값: {size_stats.get('median', 0):.0f}자")
                report_lines.append("")
        
        # Data quality issues and recommendations
        report_lines.append("## 5. 데이터 품질 이슈 및 권장사항\n")
        report_lines.append("### 5.1 발견된 이슈\n")
        
        if "metadata_analysis" in analyses and "data_quality_issues" in analyses["metadata_analysis"]:
            issues = analyses["metadata_analysis"]["data_quality_issues"]
            if issues.get("missing_values"):
                report_lines.append("- **결측치**: 일부 필드에 결측치가 있습니다.")
            if issues.get("duplicates"):
                report_lines.append("- **중복**: 일부 공고 번호가 중복됩니다.")
            if issues.get("outliers"):
                report_lines.append("- **이상치**: 예산 데이터에 이상치가 있습니다.")
        report_lines.append("")
        
        report_lines.append("### 5.2 권장사항\n")
        report_lines.append("- 데이터 정제: 결측치 및 중복 데이터 정리")
        report_lines.append("- 파싱 개선: 실패한 파일 재파싱 시도")
        report_lines.append("- 청킹 최적화: 청크 크기 및 overlap 조정 검토")
        report_lines.append("")
        
        # Insights
        report_lines.append("## 6. 인사이트 및 결론\n")
        report_lines.append("### 주요 발견사항\n")
        
        if "metadata_analysis" in analyses:
            report_lines.append("- 메타데이터 분석을 통해 데이터셋의 전반적인 특성을 파악했습니다.")
        if "text_analysis" in analyses:
            report_lines.append("- 텍스트 품질 분석을 통해 파싱 성공률 및 품질을 확인했습니다.")
        if "chunk_analysis" in analyses:
            report_lines.append("- 청크 분석을 통해 청킹 전략의 효과를 평가했습니다.")
        
        report_lines.append("\n---\n")
        report_lines.append(f"*리포트 생성 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        # Save report
        output_path_obj = Path(output_path)
        ensure_dir(str(output_path_obj.parent))
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        
        self.logger.info(f"Report saved: {output_path}")
        
        # Also save JSON statistics
        stats_path = output_path_obj.parent / f"stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_json(analyses, str(stats_path))
        self.logger.info(f"Statistics saved: {stats_path}")

