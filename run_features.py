"""Run locally: python run_features.py [input_filename]

Loads the frozen clean dataset (default: data/processed/delu_hourly.parquet;
pass a different filename as the first argument if you've renamed it,
e.g. `python run_features.py delu_hourly_v1.parquet`), builds the full
feature matrix (fundamentals, calendar, price lags/rolling stats), runs
the leakage guard, and saves the result. Prints a summary so the
point-in-time behavior can be sanity-checked against real data before
EDA/modelling (steps 6+).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from src.features import build_feature_matrix
from src.utils import load_config, setup_logging, REPO_ROOT


def main():
    cfg = load_config()
    logger = setup_logging(cfg["logging"]["level"])

    input_filename = sys.argv[1] if len(sys.argv) > 1 else "delu_hourly.parquet"
    in_path = REPO_ROOT / cfg["data"]["processed_dir"] / input_filename
    if not in_path.exists():
        available = list((REPO_ROOT / cfg["data"]["processed_dir"]).glob("*.parquet"))
        raise FileNotFoundError(
            f"{in_path} not found. Run run_ingestion.py first, or if you've "
            f"renamed the file, pass its name: python run_features.py <filename>.\n"
            f"Files found in data/processed/: {[p.name for p in available] or 'none'}"
        )

    logger.info("Loading %s", in_path)
    clean_df = pd.read_parquet(in_path)

    logger.info("Building feature matrix (fundamentals, calendar, price lags/rolling)...")
    feature_df = build_feature_matrix(
        clean_df,
        lags_hours=cfg["features"]["price_lags_hours"],
        rolling_windows_hours=cfg["features"]["rolling_windows_hours"],
    )

    out_path = REPO_ROOT / cfg["data"]["processed_dir"] / "delu_features.parquet"
    feature_df.to_parquet(out_path, index=False)

    print("\n" + "=" * 70)
    print("FEATURE MATRIX BUILT (leakage guard passed)")
    print("=" * 70)
    print(f"Output:  {out_path}")
    print(f"Rows:    {len(feature_df)}")
    print(f"Columns: {list(feature_df.columns)}")
    print("\nMissing values per column:")
    print(feature_df.isna().sum())
    print("\nFirst 3 rows with a full lag history (row 200+):")
    print(feature_df.iloc[200:203].to_string())
    print("=" * 70)


if __name__ == "__main__":
    main()
