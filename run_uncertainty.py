"""Run locally: python run_uncertainty.py <xgboost_run_version> <output_run_version>

Example:
  python run_uncertainty.py xgboost_v1_a03fix uncertainty_v1

Builds a rolling-window empirical-residual-quantile uncertainty layer
(config.yaml's uncertainty.quantiles=[0.1,0.5,0.9],
residual_window_days=180) on top of an already-run XGBoost evaluation's
saved fold predictions, and backtests its own calibration.

Reads the four folds' xgboost_full predictions
(outputs/models/<input_stem>/<xgboost_run_version>/{fold}_predictions.csv)
-- these are already genuinely out-of-sample, walk-forward predictions
with contiguous validation windows (see src/uncertainty.py's module
docstring for why pooling them is safe), so this script does not
refit or re-predict anything; it only re-derives quantile intervals
from residuals that already exist on disk.

STANDING CAVEAT: the underlying price series still depends on
auction_sequence == 1, a documented, still-open assumption pending
external EPEX verification (see README Limitations). This script
inherits that caveat unchanged -- it does not evaluate or resolve it.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from src.models import TARGET_COL
from src.uncertainty import (
    build_continuous_residual_series,
    compute_rolling_residual_quantiles,
    evaluate_interval_calibration,
    evaluate_interval_calibration_by_fold,
)
from src.utils import load_config, setup_logging, REPO_ROOT

FOLD_NAMES = ["fold_1", "fold_2", "fold_3", "regime_stress_test"]
PRED_COL = "xgboost_full_pred"


def resolve_run_args(args: list) -> tuple:
    """xgboost_run_version and output_run_version are always required.
    window_days is OPTIONAL and defaults to config.yaml's
    uncertainty.residual_window_days when omitted -- but for a
    CONSEQUENTIAL, frozen run (e.g. the actual selected specification
    following a sensitivity experiment), pass it explicitly on the
    command line so the value is traceable directly from the invocation
    itself, not implicitly inherited from a config file that could
    later be edited without anyone noticing the frozen run's provenance
    silently became misleading.
    """
    if len(args) not in (2, 3):
        raise SystemExit(
            "Usage:\n"
            "  python run_uncertainty.py <xgboost_run_version> <output_run_version> [window_days]\n\n"
            "Examples:\n"
            "  python run_uncertainty.py xgboost_v1_a03fix uncertainty_v1\n"
            "  python run_uncertainty.py xgboost_v1_a03fix uncertainty_selected_v1 60"
        )
    if len(args) == 2:
        return args[0], args[1], None
    try:
        window_days = int(args[2])
    except ValueError:
        raise SystemExit(f"window_days must be an integer, got '{args[2]}'")
    if window_days <= 0:
        raise SystemExit(f"window_days must be positive, got {window_days}")
    return args[0], args[1], window_days


def load_fold_predictions(input_stem: str, xgboost_run_version: str) -> list:
    xgboost_dir = REPO_ROOT / "outputs" / "models" / input_stem / xgboost_run_version
    if not xgboost_dir.exists():
        raise FileNotFoundError(
            f"No saved results at {xgboost_dir}. Run run_xgboost.py for "
            f"'{xgboost_run_version}' first."
        )
    frames = []
    missing = []
    for fold_name in FOLD_NAMES:
        path = xgboost_dir / f"{fold_name}_predictions.csv"
        if not path.exists():
            missing.append(fold_name)
            continue
        df = pd.read_csv(path)
        if PRED_COL not in df.columns:
            raise ValueError(f"{path} has no '{PRED_COL}' column -- unexpected format, refusing to guess.")
        frames.append(df)
    if missing:
        raise FileNotFoundError(
            f"Missing predictions file(s) for fold(s) {missing} in {xgboost_dir} -- "
            f"the residual timeline must be built from ALL four contiguous folds, "
            f"not a partial subset (a gap would silently break the 'continuous "
            f"out-of-sample timeline' assumption this whole approach relies on)."
        )
    return frames


def main():
    cfg = load_config()
    logger = setup_logging(cfg["logging"]["level"])

    xgboost_run_version, output_run_version, window_days_override = resolve_run_args(sys.argv[1:])
    quantiles = cfg["uncertainty"]["quantiles"]
    config_window_days = cfg["uncertainty"]["residual_window_days"]
    window_days = window_days_override if window_days_override is not None else config_window_days
    min_periods_days = max(1, window_days // 4)  # require at least a quarter of the window before trusting it
    window_days_source = "explicit CLI argument" if window_days_override is not None else "config.yaml default"
    logger.info("window_days=%d (source: %s)", window_days, window_days_source)

    input_stem = "delu_features"  # matches the feature file this project's models are built from
    out_dir = REPO_ROOT / "outputs" / "uncertainty" / input_stem / output_run_version
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"{out_dir} already contains results. Runs are meant to be immutable "
            f"evidence -- pass a new output_run_version instead of overwriting it."
        )

    logger.info("Loading xgboost_full predictions from '%s' (all 4 folds)...", xgboost_run_version)
    fold_predictions = load_fold_predictions(input_stem, xgboost_run_version)

    logger.info(
        "Building the continuous out-of-sample residual series (contiguous folds, "
        "no gaps/overlaps -- asserted, not assumed)..."
    )
    residual_series = build_continuous_residual_series(
        fold_predictions, TARGET_COL, PRED_COL, fold_names=FOLD_NAMES
    )
    logger.info(
        "%d rows, %s -> %s", len(residual_series),
        residual_series["timestamp_utc"].min(), residual_series["timestamp_utc"].max(),
    )

    logger.info(
        "Computing rolling %d-day residual quantiles %s (min_periods=%d days)...",
        window_days, quantiles, min_periods_days,
    )
    quantile_forecasts = compute_rolling_residual_quantiles(
        residual_series, quantiles, window_days, min_periods_days
    )

    merged = residual_series.merge(quantile_forecasts.drop(columns=["prediction"]), on="timestamp_utc", how="left")

    print("\n" + "=" * 78)
    print(f"ROLLING RESIDUAL QUANTILE UNCERTAINTY: {output_run_version}")
    print(f"Built on top of xgboost_full predictions from '{xgboost_run_version}'")
    print("=" * 78)
    print(f"Quantiles: {quantiles}, window_days: {window_days}, min_periods_days: {min_periods_days}")
    n_warmup = merged[f"forecast_q{int(quantiles[0]*100)}"].isna().sum()
    print(f"Rows in warm-up period (no interval yet, correctly NaN): {n_warmup} / {len(merged)}")

    calibration_results = {}
    calibration_by_fold_df = None
    sorted_q = sorted(quantiles)
    if len(sorted_q) >= 2:
        lo_q, hi_q = sorted_q[0], sorted_q[-1]
        nominal_coverage = hi_q - lo_q
        lower_col, upper_col = f"forecast_q{int(lo_q*100)}", f"forecast_q{int(hi_q*100)}"
        calib = evaluate_interval_calibration(
            merged[TARGET_COL], merged[lower_col], merged[upper_col], nominal_coverage
        )
        calibration_results[f"[{lower_col}, {upper_col}]"] = calib
        print(f"\nPooled calibration check, nominal {nominal_coverage:.0%} interval "
              f"[{lower_col}, {upper_col}]:")
        print(f"  n = {calib['n']}")
        print(f"  empirical coverage = {calib['empirical_coverage']:.4f} (nominal {nominal_coverage:.4f})")
        print(f"  fraction below lower = {calib['frac_below_lower']:.4f} (nominal {(1-nominal_coverage)/2:.4f})")
        print(f"  fraction above upper = {calib['frac_above_upper']:.4f} (nominal {(1-nominal_coverage)/2:.4f})")

        # Per-fold breakdown: a pooled number can hide real regime
        # differences -- specifically whether regime_stress_test (the
        # newest regime, closest to the eventual holdout) is equally
        # well-calibrated, not just the average across all four folds.
        calibration_by_fold_df = evaluate_interval_calibration_by_fold(
            merged, TARGET_COL, lower_col, upper_col, nominal_coverage
        )
        print(f"\nPer-fold calibration breakdown (chronological order):")
        print(calibration_by_fold_df.to_string(index=False))

    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / "quantile_forecasts.csv", index=False)
    if calibration_by_fold_df is not None:
        calibration_by_fold_df.to_csv(out_dir / "calibration_by_fold.csv", index=False)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "xgboost_run_version": xgboost_run_version,
        "output_run_version": output_run_version,
        "quantiles": quantiles,
        "window_days": window_days,
        "window_days_source": window_days_source,
        "config_yaml_default_window_days": config_window_days,
        "min_periods_days": min_periods_days,
        "n_rows": len(merged),
        "residual_series_start": str(residual_series["timestamp_utc"].min()),
        "residual_series_end": str(residual_series["timestamp_utc"].max()),
        "calibration": calibration_results,
        "calibration_by_fold": calibration_by_fold_df.to_dict(orient="records") if calibration_by_fold_df is not None else None,
        "STANDING_CAVEAT": (
            "auction_sequence == 1 is a documented, still-open assumption pending external "
            "EPEX verification (see README Limitations). This uncertainty layer inherits "
            "that caveat unchanged."
        ),
    }
    with open(out_dir / "uncertainty_run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\nSaved quantile forecasts + manifest to {out_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
