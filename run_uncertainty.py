"""Run locally: python run_uncertainty.py <xgboost_run_version> <output_run_version>

Example:
  python run_uncertainty.py xgboost_v1_a03fix uncertainty_v1

Builds a DELIVERY-DAY-SAFE empirical-residual-quantile uncertainty
layer (config.yaml's uncertainty.quantiles=[0.1,0.5,0.9],
residual_window_days=180) on top of an already-run XGBoost evaluation's
saved fold predictions, and backtests its own calibration.

Uses compute_delivery_day_residual_quantiles(), not the older
compute_rolling_residual_quantiles() -- the latter computes each
hour's window per HOURLY timestamp, which can let an afternoon hour of
delivery day D see residuals from earlier hours of that SAME day D.
Safe for one-step-ahead hourly evaluation, not safe for this project's
actual decision problem: the whole delivery day D's schedule is
committed at D-1 11:45, before any of day D's prices exist, so no hour
of D may ever inform D's own uncertainty bound. Every hour of D now
receives one identical offset snapshot, built only from delivery days
strictly before D (see src/uncertainty.py's hostile tests for the
verified correctness guarantee).

Reads the four folds' xgboost_full predictions
(outputs/models/<input_stem>/<xgboost_run_version>/{fold}_predictions.csv)
-- these are already genuinely out-of-sample, walk-forward predictions
with contiguous validation windows (see src/uncertainty.py's module
docstring for why pooling them is safe), so this script does not
refit or re-predict anything; it only re-derives quantile intervals
from residuals that already exist on disk.

STANDING NOTE: auction_sequence == 1 was independently confirmed via
cross-check against SMARD (Germany's official market data platform)
across all 8,833 disagreeing corrected-data intervals -- 100%
consistent with sequence 1, 0% with sequence 2. See README "Auction
sequence" for the full result.
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
    compute_delivery_day_residual_quantiles,
    evaluate_interval_calibration,
    evaluate_interval_calibration_by_fold,
)
from src.strategy import assert_no_holdout_access
from src.utils import load_config, setup_logging, REPO_ROOT

FOLD_NAMES = ["fold_1", "fold_2", "fold_3", "regime_stress_test"]
PRED_COL = "xgboost_full_pred"


def resolve_run_args(args: list) -> tuple:
    """xgboost_run_version and output_run_version are always required.
    The 3-argument form's middle argument is a SENSITIVITY_RUN_VERSION
    (a string), not a raw window_days integer -- an earlier version of
    this script accepted a manually-typed integer override here, which
    left open the exact failure mode this replaces: nothing stopped a
    frozen, consequential run from silently using window_days=60 when
    the actual pre-registered sensitivity experiment selected 90,
    since the number was manually retyped rather than read from the
    experiment's own saved result. The 3-argument form now reads
    window_days, quantiles, and provenance directly from the named
    sensitivity run's manifest -- see load_sensitivity_winner().
    """
    if len(args) not in (2, 3):
        raise SystemExit(
            "Usage:\n"
            "  python run_uncertainty.py <xgboost_run_version> <output_run_version>\n"
            "  python run_uncertainty.py <xgboost_run_version> <sensitivity_run_version> <output_run_version>\n\n"
            "Examples:\n"
            "  python run_uncertainty.py xgboost_v1_a03fix uncertainty_v1\n"
            "  python run_uncertainty.py xgboost_v1_a03fix uncertainty_window_sensitivity_v3_dayorigin uncertainty_selected_v2_dayorigin\n\n"
            "The 2-argument form uses config.yaml's default window_days -- appropriate for "
            "exploratory runs only, not a frozen/consequential specification. The 3-argument "
            "form binds mechanically to a completed sensitivity experiment's own selected winner."
        )
    if len(args) == 2:
        return args[0], args[1], None
    return args[0], args[2], args[1]  # (xgboost_run_version, output_run_version, sensitivity_run_version)


def load_sensitivity_winner(input_stem: str, sensitivity_run_version: str, expected_xgboost_run_version: str) -> dict:
    """Reads the mechanically-selected winner (lowest common-row
    interval score) directly from a completed sensitivity run's own
    manifest -- never re-typed. Fails closed if the sensitivity run's
    own xgboost_run_version doesn't match what THIS run was asked to
    build on, or if it wasn't itself built with the corrected
    delivery-day-safe method.
    """
    manifest_path = REPO_ROOT / "outputs" / "uncertainty" / input_stem / sensitivity_run_version / "sensitivity_run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No sensitivity run found at {manifest_path}. Run run_uncertainty_sensitivity.py "
            f"for '{sensitivity_run_version}' first."
        )
    manifest = json.loads(manifest_path.read_text())
    for field in ("lowest_interval_score_window_days", "quantiles", "information_set_method", "xgboost_run_version"):
        if field not in manifest:
            raise ValueError(f"{manifest_path} is missing required field '{field}' -- unexpected format.")

    if manifest["xgboost_run_version"] != expected_xgboost_run_version:
        raise ValueError(
            f"Sensitivity run '{sensitivity_run_version}' was built on xgboost_run_version="
            f"{manifest['xgboost_run_version']!r}, but this run was asked to use "
            f"{expected_xgboost_run_version!r} -- refusing to mix predictions from different "
            f"XGBoost runs with a sensitivity experiment built on a different one."
        )
    if manifest["information_set_method"] != "delivery_day_safe":
        raise ValueError(
            f"Sensitivity run '{sensitivity_run_version}' has information_set_method="
            f"{manifest['information_set_method']!r}, expected 'delivery_day_safe' -- refusing "
            f"to bind a new economic uncertainty run to a stale, pre-correction sensitivity result."
        )

    return {
        "window_days": manifest["lowest_interval_score_window_days"],
        "quantiles": manifest["quantiles"],
        "information_set_method": manifest["information_set_method"],
        "xgboost_run_version": manifest["xgboost_run_version"],
        "source_manifest": str(manifest_path),
    }


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

    xgboost_run_version, output_run_version, sensitivity_run_version = resolve_run_args(sys.argv[1:])

    if sensitivity_run_version is not None:
        input_stem_for_lookup = "delu_features"
        winner = load_sensitivity_winner(input_stem_for_lookup, sensitivity_run_version, xgboost_run_version)
        window_days = winner["window_days"]
        quantiles = winner["quantiles"]
        window_days_source = f"mechanically selected by sensitivity run '{sensitivity_run_version}'"
        logger.info(
            "Binding to sensitivity winner from '%s': window_days=%d, quantiles=%s",
            winner["source_manifest"], window_days, quantiles,
        )
    else:
        quantiles = cfg["uncertainty"]["quantiles"]
        window_days = cfg["uncertainty"]["residual_window_days"]
        window_days_source = "config.yaml default (exploratory run, not a frozen specification)"
        winner = None

    min_periods_days = max(1, window_days // 4)  # require at least a quarter of the window before trusting it
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
    # Structural, not just conventional: a second independent lock on the
    # holdout, matching the pattern already established at the strategy
    # layer's own entry point (src/strategy.py::assert_no_holdout_access).
    assert_no_holdout_access(residual_series["timestamp_utc"])

    logger.info(
        "Computing delivery-day-safe residual quantiles %s over a %d-day trailing "
        "window (min_periods=%d delivery days) -- every hour of delivery day D uses "
        "residuals ONLY from delivery days strictly before D, never an earlier hour "
        "of D itself (see src/uncertainty.py for the information-set correction and "
        "its hostile tests)...",
        quantiles, window_days, min_periods_days,
    )
    quantile_forecasts = compute_delivery_day_residual_quantiles(
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
        "parent_sensitivity_run": sensitivity_run_version,
        "selected_window_days": window_days if winner is not None else None,
        "selection_rule": "mechanical: lowest common-row interval score, read from parent sensitivity manifest" if winner is not None else None,
        "min_periods_days": min_periods_days,
        "information_set_method": "delivery_day_safe",
        "n_rows": len(merged),
        "residual_series_start": str(residual_series["timestamp_utc"].min()),
        "residual_series_end": str(residual_series["timestamp_utc"].max()),
        "calibration": calibration_results,
        "calibration_by_fold": calibration_by_fold_df.to_dict(orient="records") if calibration_by_fold_df is not None else None,
        "STANDING_CAVEAT": (
            "auction_sequence == 1 was independently confirmed via cross-check against SMARD "
            "(Germany official market data platform) across all 8,833 disagreeing corrected-data "
            "intervals -- 100% consistent with sequence 1, 0% with sequence 2. See README  This uncertainty layer inherits "
            "that caveat unchanged."
        ),
    }
    with open(out_dir / "uncertainty_run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\nSaved quantile forecasts + manifest to {out_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
