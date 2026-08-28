"""Run locally: python run_uncertainty_sensitivity.py <xgboost_run_version> <output_run_version>

Example:
  python run_uncertainty_sensitivity.py xgboost_v1_a03fix uncertainty_window_sensitivity_v1

PRE-REGISTERED window sensitivity experiment -- built specifically
BECAUSE uncertainty_v2 (the frozen 180-day reference) showed unstable
per-fold calibration, and this project has a standing rule against
tuning a method's hyperparameter in direct response to having seen a
result (the same discipline as ElasticNet's baseline_v2 stopping rule
and XGBoost's no-retuning-after-xgboost_v1 rule). CANDIDATE_WINDOW_DAYS
below is fixed BEFORE this script has ever been run against real data,
specifically so "which window looks best" cannot retroactively shape
which windows were even considered.

Three dimensions are scored for every candidate, not just calibration
alone:
  1. Calibration -- pooled AND per-fold coverage, tail balance.
     Coverage alone is gameable: an absurdly wide interval "achieves"
     100% coverage while being useless.
  2. Sharpness -- mean/median interval width. Width alone is also
     gameable: a zero-width interval is maximally sharp and covers
     almost nothing.
  3. Interval/Winkler score -- the PRIMARY decision metric, because it
     cannot be gamed by either move alone (see
     src/uncertainty.py::compute_interval_score's docstring). Lower is
     better.

The decision is NOT fully automated. This script prints the full
comparison table and states the interval-score ranking, but explicitly
flags if the top-ranked candidate has any single fold whose coverage
deviates from nominal by more than FOLD_CALIBRATION_FLAG_THRESHOLD --
a human should look at that before accepting the "winner," not trust
one aggregate number silently, matching the same caution this project
applied when refusing to reopen frozen baselines automatically.

STANDING CAVEAT: inherits the same auction_sequence == 1 open
assumption as everything downstream of the price series -- see README
Limitations.
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
    evaluate_window_candidate,
    find_common_evaluation_start,
)
from src.utils import load_config, setup_logging, REPO_ROOT

FOLD_NAMES = ["fold_1", "fold_2", "fold_3", "regime_stress_test"]
PRED_COL = "xgboost_full_pred"

# ---------------------------------------------------------------------
# PRE-REGISTERED candidate windows -- fixed before this script has ever
# been run against real data. Do not add, remove, or reorder entries
# after seeing a result; that would defeat the entire point of
# pre-registration. min_periods_days keeps the same window/4 ratio
# uncertainty_v2 used, applied consistently across candidates rather
# than hand-tuned per candidate.
# ---------------------------------------------------------------------
CANDIDATE_WINDOW_DAYS = [60, 90, 120, 180, 365]


def min_periods_days_for(window_days: int) -> int:
    return max(1, window_days // 4)


def resolve_run_args(args: list) -> tuple:
    if len(args) != 2:
        raise SystemExit(
            "Usage:\n"
            "  python run_uncertainty_sensitivity.py <xgboost_run_version> <output_run_version>\n\n"
            "Example:\n"
            "  python run_uncertainty_sensitivity.py xgboost_v1_a03fix uncertainty_window_sensitivity_v1"
        )
    return args[0], args[1]


def load_fold_predictions(input_stem: str, xgboost_run_version: str) -> list:
    xgboost_dir = REPO_ROOT / "outputs" / "models" / input_stem / xgboost_run_version
    if not xgboost_dir.exists():
        raise FileNotFoundError(f"No saved results at {xgboost_dir}. Run run_xgboost.py first.")
    frames = []
    missing = []
    for fold_name in FOLD_NAMES:
        path = xgboost_dir / f"{fold_name}_predictions.csv"
        if not path.exists():
            missing.append(fold_name)
            continue
        df = pd.read_csv(path)
        if PRED_COL not in df.columns:
            raise ValueError(f"{path} has no '{PRED_COL}' column -- unexpected format.")
        frames.append(df)
    if missing:
        raise FileNotFoundError(f"Missing predictions file(s) for fold(s) {missing} in {xgboost_dir}.")
    return frames


def main():
    cfg = load_config()
    logger = setup_logging(cfg["logging"]["level"])

    xgboost_run_version, output_run_version = resolve_run_args(sys.argv[1:])
    quantiles = cfg["uncertainty"]["quantiles"]
    input_stem = "delu_features"

    out_dir = REPO_ROOT / "outputs" / "uncertainty" / input_stem / output_run_version
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"{out_dir} already contains results. Pass a new output_run_version instead of overwriting it."
        )

    logger.info("Loading xgboost_full predictions from '%s'...", xgboost_run_version)
    fold_predictions = load_fold_predictions(input_stem, xgboost_run_version)
    residual_series = build_continuous_residual_series(
        fold_predictions, TARGET_COL, PRED_COL, fold_names=FOLD_NAMES
    )
    logger.info(
        "%d rows, %s -> %s", len(residual_series),
        residual_series["timestamp_utc"].min(), residual_series["timestamp_utc"].max(),
    )

    print("\n" + "=" * 78)
    print(f"PRE-REGISTERED UNCERTAINTY WINDOW SENSITIVITY: {output_run_version}")
    print(f"Candidate window_days (fixed before this run): {CANDIDATE_WINDOW_DAYS}")
    print("=" * 78)

    candidate_configs = [(w, min_periods_days_for(w)) for w in CANDIDATE_WINDOW_DAYS]

    def _summarize(results_list):
        rows = []
        for r in results_list:
            rows.append({
                "window_days": r["window_days"],
                "n": r["pooled_calibration"]["n"],
                "n_warmup": r["n_warmup"],
                "pooled_coverage": r["pooled_calibration"]["empirical_coverage"],
                "pooled_frac_below": r["pooled_calibration"]["frac_below_lower"],
                "pooled_frac_above": r["pooled_calibration"]["frac_above_upper"],
                "mean_width": r["mean_width"],
                "median_width": r["median_width"],
                "interval_score": r["interval_score"],
            })
        return pd.DataFrame(rows).sort_values("interval_score")

    # --- Table 1: available-row, diagnostic only. Each candidate is
    # scored on however many rows IT can produce a forecast for -- a
    # shorter window is partly scored on early data a longer window
    # never gets evaluated on. Kept for transparency, NOT used to select.
    available_results = []
    for window_days, min_periods_days in candidate_configs:
        logger.info("Evaluating (available-row) window_days=%d (min_periods_days=%d)...", window_days, min_periods_days)
        available_results.append(
            evaluate_window_candidate(residual_series, quantiles, window_days, min_periods_days, target_col=TARGET_COL)
        )
    available_summary = _summarize(available_results)

    print("\n--- Table 1: AVAILABLE-ROW comparison (diagnostic only, NOT the selection table) ---")
    print("Each candidate scored on however many rows it can produce a forecast for -- different")
    print("candidates see different row counts (n differs), which confounds window choice with")
    print("which period of data got included. Kept for transparency, see Table 2 for selection.")
    print(available_summary.to_string(index=False))

    # --- Table 2: common-row, the ACTUAL selection table. Every
    # candidate scored on the identical row set (from the longest
    # warm-up's cutoff onward) -- same principle already enforced for
    # point-model comparison (baseline_v1's common_comparison_rows).
    common_start = find_common_evaluation_start(residual_series, quantiles, candidate_configs)
    logger.info("Common evaluation start (latest of all candidates' warm-ups): %s", common_start)

    common_results = []
    for window_days, min_periods_days in candidate_configs:
        logger.info("Evaluating (common-row) window_days=%d...", window_days)
        common_results.append(
            evaluate_window_candidate(
                residual_series, quantiles, window_days, min_periods_days,
                target_col=TARGET_COL, common_start=common_start,
            )
        )
    common_summary = _summarize(common_results)

    print(f"\n--- Table 2: COMMON-ROW comparison (SELECTION IS BASED ONLY ON THIS TABLE) ---")
    print(f"Every candidate scored on rows >= {common_start} (the longest warm-up's cutoff) --")
    print(f"identical n for every candidate, confirmed below.")
    print(common_summary.to_string(index=False))
    n_values = common_summary["n"].unique()
    assert len(n_values) == 1, f"common-row correction failed -- candidates still have different n: {n_values.tolist()}"
    print(f"\nConfirmed: all {len(CANDIDATE_WINDOW_DAYS)} candidates scored on identical n={n_values[0]}.")

    print("\nPer-fold calibration for every candidate (common-row basis):")
    for r in common_results:
        print(f"\n--- window_days = {r['window_days']} ---")
        if r["by_fold_calibration"] is not None:
            print(r["by_fold_calibration"].to_string(index=False))

    results = common_results
    summary = common_summary

    best = summary.iloc[0]
    best_window = int(best["window_days"])
    best_result = next(r for r in results if r["window_days"] == best_window)

    print("\n" + "=" * 78)
    print(f"LOWEST INTERVAL SCORE (common-row basis): window_days={best_window} (score={best['interval_score']:.4f})")
    print("=" * 78)

    FOLD_CALIBRATION_FLAG_THRESHOLD = 0.03  # 3 percentage points off nominal
    flagged_folds = []
    if best_result["by_fold_calibration"] is not None:
        for _, row in best_result["by_fold_calibration"].iterrows():
            deviation = abs(row["empirical_coverage"] - row["nominal_coverage"])
            if deviation > FOLD_CALIBRATION_FLAG_THRESHOLD:
                flagged_folds.append((row["fold"], row["empirical_coverage"], deviation))

    if flagged_folds:
        print(f"\nCAUTION: the lowest-interval-score candidate (window_days={best_window}) still has "
              f"fold(s) with coverage more than {FOLD_CALIBRATION_FLAG_THRESHOLD*100:.0f} points off "
              f"nominal:")
        for fold_name, coverage, deviation in flagged_folds:
            print(f"  {fold_name}: coverage={coverage:.4f} (off by {deviation*100:.2f} points)")
        print("Do not treat the lowest interval score as a complete answer on its own -- review "
              "these folds specifically before adopting this window as the new reference.")
    else:
        print(f"\nNo fold's coverage deviates from nominal by more than "
              f"{FOLD_CALIBRATION_FLAG_THRESHOLD*100:.0f} points for this candidate.")

    out_dir.mkdir(parents=True, exist_ok=True)
    available_summary.to_csv(out_dir / "available_row_comparison_DIAGNOSTIC_ONLY.csv", index=False)
    summary.to_csv(out_dir / "common_row_comparison_SELECTION_TABLE.csv", index=False)
    for r in results:
        if r["by_fold_calibration"] is not None:
            r["by_fold_calibration"].to_csv(
                out_dir / f"by_fold_calibration_window{r['window_days']}.csv", index=False
            )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "xgboost_run_version": xgboost_run_version,
        "output_run_version": output_run_version,
        "pre_registered_candidate_window_days": CANDIDATE_WINDOW_DAYS,
        "quantiles": quantiles,
        "primary_decision_metric": "interval_score (lower is better), computed on the COMMON-ROW table",
        "common_evaluation_start": str(common_start),
        "common_n": int(n_values[0]),
        "lowest_interval_score_window_days": best_window,
        "flagged_folds_for_lowest_score_candidate": [
            {"fold": f, "coverage": c, "deviation_from_nominal": d} for f, c, d in flagged_folds
        ],
        "fold_calibration_flag_threshold": FOLD_CALIBRATION_FLAG_THRESHOLD,
        "STANDING_CAVEAT": (
            "auction_sequence == 1 is a documented, still-open assumption pending external "
            "EPEX verification (see README Limitations)."
        ),
    }
    with open(out_dir / "sensitivity_run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\nSaved comparison + manifest to {out_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
