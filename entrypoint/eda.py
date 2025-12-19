"""EDA entrypoint script."""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, skip

from src.common.config import load_config
from src.common.logger import get_logger
from src.eda.eda_agent import EDAAgent


def main():
    parser = argparse.ArgumentParser(description="RFP RAG EDA Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/local.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for reports (default: data/eda/reports)"
    )
    parser.add_argument(
        "--analysis",
        type=str,
        default="all",
        help="Analysis types to run: all, metadata, text, chunks (comma-separated)"
    )
    
    args = parser.parse_args()
    
    logger = get_logger(__name__)
    
    logger.info("=" * 60)
    logger.info("RFP RAG EDA Pipeline Started")
    logger.info("=" * 60)
    
    # Load config
    config = load_config(args.config)
    
    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = config.get("eda", {}).get("output_dir", "data/eda/reports")
    
    # Determine which analyses to run
    analysis_types = args.analysis.lower().split(",")
    analyze_metadata = "all" in analysis_types or "metadata" in analysis_types
    analyze_text = "all" in analysis_types or "text" in analysis_types
    analyze_chunks = "all" in analysis_types or "chunks" in analysis_types
    
    try:
        # Initialize EDA Agent
        eda_agent = EDAAgent(config)
        
        # Run analysis
        results = eda_agent.run_full_analysis(
            output_dir=output_dir,
            analyze_metadata=analyze_metadata,
            analyze_text=analyze_text,
            analyze_chunks=analyze_chunks
        )
        
        # Generate report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path(output_dir) / f"eda_report_{timestamp}.md"
        eda_agent.generate_report(results, str(report_path))
        
        logger.info("=" * 60)
        logger.info("EDA Pipeline Completed Successfully")
        logger.info(f"Report saved: {report_path}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"EDA pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

