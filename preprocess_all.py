"""
Unified Preprocessing Entrypoint

Orchestrates the full data pipeline for MIMIC-IV ICU time series:
1. Extraction (data/extract.py): From raw CSVs to intermediate Parquet.
2. Preprocessing (data/preprocess.py): Pivoting, filtering, splitting, and normalization.

Usage:
    python preprocess_all.py --config configs/data/mimic4.yaml [--sample_n 1000]
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path to ensure imports work correctly
sys.path.append(str(Path(__file__).resolve().parent))

from data.extract import run_extraction, load_config
from data.preprocess import run_preprocessing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("preprocessing.log")
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Full MIMIC-IV Data Pipeline")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/data/mimic4.yaml",
        help="Path to data config YAML"
    )
    parser.add_argument(
        "--sample_n", 
        type=int, 
        default=None, 
        help="Only use first N stays for development (extraction phase)"
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logger.error(f"Config file not found: {cfg_path}")
        sys.exit(1)

    # 1. Load Config
    logger.info(f"Loading configuration from {cfg_path}...")
    cfg = load_config(cfg_path)

    # 2. Run Extraction
    logger.info("=" * 60)
    logger.info("PHASE 1: Extraction")
    logger.info("=" * 60)
    try:
        run_extraction(cfg, sample_n=args.sample_n)
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        sys.exit(1)

    # 3. Run Preprocessing
    logger.info("=" * 60)
    logger.info("PHASE 2: Preprocessing")
    logger.info("=" * 60)
    try:
        run_preprocessing(cfg)
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Full pipeline completed successfully!")
    logger.info("Processed data is available in: " + cfg["processed_dir"])
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
