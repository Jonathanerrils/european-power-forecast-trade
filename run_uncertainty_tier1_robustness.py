"""Run locally: python run_uncertainty_tier1_robustness.py <xgboost_run_version> <uncertainty_run_version> <output_run_version>

Example:
  python run_uncertainty_tier1_robustness.py xgboost_v1_a03fix uncertainty_selected_v2_dayorigin uncertainty_tier1_robustness_v3_dayorigin

ROBUSTNESS CHECK, not a new model-selection exercise. Applies the
ALREADY-FROZEN uncertainty specification from an explicitly-named
parent uncertainty run (window_days, min_periods_days, quantiles --
read directly from that run's own saved manifest, never re-hardcoded
here) to XGBoost Tier-1's predictions instead of XGBoost-full's, and
reports the two side by side.

uncertainty_run_version is a REQUIRED positional argument, not a
default -- an earlier version of this script silently defaulted to
"uncertainty_selected_v1" if not given, which became genuinely
dangerous after that specification was superseded by a delivery-day-
information-set correction: a caller who forgot to pass the new
parent explicitly would silently keep reading the OLD, pre-correction
specification with no error. Requiring it explicitly, every time,
removes that failure mode entirely rather than relying on anyone
remembering to override a default. The saved manifest also now
records the ACTUAL parent passed, not a hardcoded string literal --
a prior version of this script's manifest always wrote
"uncertainty_selected_v1" regardless of what was actually used, which
would have made a future override untraceable even after the fact.

Also verifies the parent uncertainty run was itself built with the
delivery-day-safe method (information_set_method == "delivery_day_safe"
in its manifest) before proceeding -- refuses to build Tier-1
robustness evidence on top of a stale, pre-correction Full
specification, even if one is passed by name.

Deliberately does NOT:
  - run the [60, 90, 120, 180, 365] sensitivity competition for Tier-1
  - retune window_days, min_periods_days, or quantiles in any way
  - select a different specification even if Tier-1 calibrates worse

The question this answers is narrow and specific: does the SAME
uncertainty specification, selected using the full-information model,
remain reasonably calibrated on the more defensible (but less
accurate) Tier-1 information set? Not "what's the best uncertainty
spec for Tier-1" -- that would be a second, separate, currently
out-of-scope experiment.

STANDING NOTE: auction_sequence == 1 was independently confirmed via
cross-check against SMARD (Germany's official market data platform)
across all 8,833 disagreeing corrected-data intervals -- 100%
consistent with sequence 1, 0% with sequence 2. See README "Auction
sequence" for the full result. Never touches 2026 (built from the same four contiguous
development folds as everything else in this project).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from src.models import TARGET_COL
from src.uncertainty import build_continuous_residual_series, evaluate_window_candidate
from src.strategy import assert_no_holdout_access
from src.utils import load_config, setup_logging, REPO_ROOT

FOLD_NAMES = ["fold_1", "fold_2", "fold_3", "regime_stress_test"]
FULL_PRED_COL = "xgboost_full_pred"
TIER1_PRED_COL = "xgboost_tier1_pred"
REQUIRED_INFORMATION_SET_METHOD = "delivery_day_safe"


def resolve_run_args(args: list) -> tuple:
    if len(args) != 3:
        raise SystemExit(
            "Usage:\n"
            "  python run_uncertainty_tier1_robustness.py <xgboost_run_version> <uncertainty_run_version> <output_run_version>\n\n"
            "Example:\n"
            "  python run_uncertainty_tier1_robustness.py xgboost_v1_a03fix uncertainty_selected_v2_dayorigin uncertainty_tier1_robustness_v3_dayorigin\n\n"
            "uncertainty_run_version is REQUIRED and has no default -- pass the exact parent "
            "uncertainty run explicitly every time."
        )
    return args[0], args[1], args[2]


def load_frozen_uncertainty_spec(input_stem: str, uncertainty_run_version: str, expected_xgboost_run_version: str) -> dict:
    """Reads window_days/min_periods_days/quantiles from the named
    parent uncertainty run's OWN saved manifest -- never re-hardcoded
    in this script. uncertainty_run_version has no default; the caller
    must name the parent explicitly every time (see module docstring
    for why this used to be dangerous). Also fails closed if the
    parent was not itself built with the corrected delivery-day-safe
    method, or if it was built on a DIFFERENT xgboost_run_version than
    the one THIS run's Tier-1/Full predictions are being read from --
    the earlier version of this check verified the method but never
    verified the parent's own xgboost_run_version, leaving room to
    silently combine e.g. Full's corrected uncertainty bounds (built
    from one XGBoost run) with Tier-1 predictions from a DIFFERENT,
    unrelated XGBoost run.
    """
    manifest_path = REPO_ROOT / "outputs" / "uncertainty" / input_stem / uncertainty_run_version / "uncertainty_run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No frozen uncertainty specification found at {manifest_path}. Run "
            f"'python run_uncertainty.py <xgboost_run_version> {uncertainty_run_version} <window_days>' first."
        )
    manifest = json.loads(manifest_path.read_text())
    for field in ("window_days", "min_periods_days", "quantiles", "xgboost_run_version"):
        if field not in manifest:
            raise ValueError(f"{manifest_path} is missing required field '{field}' -- unexpected format.")

    method = manifest.get("information_set_method")
    if method != REQUIRED_INFORMATION_SET_METHOD:
        raise ValueError(
            f"Parent uncertainty run '{uncertainty_run_version}' has information_set_method="
            f"{method!r}, expected {REQUIRED_INFORMATION_SET_METHOD!r}. Refusing to build Tier-1 "
            f"robustness evidence on top of a stale, pre-correction specification -- this run "
            f"predates the delivery-day information-set fix and must not be used as a parent for "
            f"new economic-uncertainty work. (Historical runs like uncertainty_selected_v1 remain "
            f"valid evidence of what the OLD method produced; they are simply not valid PARENTS "
            f"for new work going forward.)"
        )

    if manifest["xgboost_run_version"] != expected_xgboost_run_version:
        raise ValueError(
            f"Parent uncertainty run '{uncertainty_run_version}' was built on xgboost_run_version="
            f"{manifest['xgboost_run_version']!r}, but this run was asked to use "
            f"{expected_xgboost_run_version!r} -- refusing to build Tier-1 robustness evidence by "
            f"mixing a parent specification with predictions from a different XGBoost run."
        )

    return {
        "window_days": manifest["window_days"],
        "min_periods_days": manifest["min_periods_days"],
        "quantiles": manifest["quantiles"],
        "xgboost_run_version": manifest["xgboost_run_version"],
        "information_set_method": manifest["information_set_method"],
        "source_manifest": str(manifest_path),
    }


def load_fold_predictions(input_stem: str, xgboost_run_version: str, pred_col: str) -> list:
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
        if pred_col not in df.columns:
            raise ValueError(f"{path} has no '{pred_col}' column -- unexpected format, refusing to guess.")
        frames.append(df)
    if missing:
        raise FileNotFoundError(
            f"Missing predictions file(s) for fold(s) {missing} in {xgboost_dir} -- the residual "
            f"timeline must be built from ALL four contiguous folds, not a partial subset."
        )
    return frames


def main():
    cfg = load_config()
    logger = setup_logging(cfg["logging"]["level"])

    xgboost_run_version, uncertainty_run_version, output_run_version = resolve_run_args(sys.argv[1:])
    input_stem = "delu_features"

    out_dir = REPO_ROOT / "outputs" / "uncertainty" / input_stem / output_run_version
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"{out_dir} already contains results. Pass a new output_run_version instead of overwriting it."
        )

    spec = load_frozen_uncertainty_spec(input_stem, uncertainty_run_version, xgboost_run_version)
    logger.info(
        "Using FROZEN specification from '%s': window_days=%d, min_periods_days=%d, quantiles=%s",
        spec["source_manifest"], spec["window_days"], spec["min_periods_days"], spec["quantiles"],
    )

    print("\n" + "=" * 78)
    print(f"TIER-1 UNCERTAINTY ROBUSTNESS CHECK: {output_run_version}")
    print(f"Reusing the FROZEN '{uncertainty_run_version}' specification -- no retuning.")
    print("=" * 78)
    print(f"window_days={spec['window_days']}, min_periods_days={spec['min_periods_days']}, "
          f"quantiles={spec['quantiles']}")

    results = {}
    for label, pred_col in [("full", FULL_PRED_COL), ("tier1", TIER1_PRED_COL)]:
        logger.info("Building residual series for xgboost_%s...", label)
        fold_predictions = load_fold_predictions(input_stem, xgboost_run_version, pred_col)
        residual_series = build_continuous_residual_series(
            fold_predictions, TARGET_COL, pred_col, fold_names=FOLD_NAMES
        )
        assert_no_holdout_access(residual_series["timestamp_utc"])
        results[label] = evaluate_window_candidate(
            residual_series, spec["quantiles"], spec["window_days"], spec["min_periods_days"],
            target_col=TARGET_COL, return_per_row=True,
        )

    comparison_rows = []
    for label in ("full", "tier1"):
        r = results[label]
        comparison_rows.append({
            "model": f"xgboost_{label}",
            "n": r["pooled_calibration"]["n"],
            "pooled_coverage": r["pooled_calibration"]["empirical_coverage"],
            "pooled_frac_below": r["pooled_calibration"]["frac_below_lower"],
            "pooled_frac_above": r["pooled_calibration"]["frac_above_upper"],
            "mean_width": r["mean_width"],
            "median_width": r["median_width"],
            "interval_score": r["interval_score"],
        })
    comparison = pd.DataFrame(comparison_rows)

    print(
        f"\nFull vs Tier-1, pooled comparison "
        f"(same frozen {spec['window_days']}-day specification for both):"
    )
    print(comparison.to_string(index=False))

    print("\nPer-fold calibration, Full vs Tier-1:")
    for label in ("full", "tier1"):
        print(f"\n--- xgboost_{label} ---")
        by_fold = results[label]["by_fold_calibration"]
        if by_fold is not None:
            print(by_fold.to_string(index=False))

    stress_full = results["full"]["by_fold_calibration"]
    stress_tier1 = results["tier1"]["by_fold_calibration"]
    if stress_full is not None and stress_tier1 is not None:
        sf = stress_full[stress_full["fold"] == "regime_stress_test"].iloc[0]
        st = stress_tier1[stress_tier1["fold"] == "regime_stress_test"].iloc[0]
        print(f"\nregime_stress_test specifically -- Full coverage={sf['empirical_coverage']:.4f}, "
              f"Tier-1 coverage={st['empirical_coverage']:.4f}, "
              f"difference={st['empirical_coverage']-sf['empirical_coverage']:+.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(out_dir / "full_vs_tier1_comparison.csv", index=False)
    for label in ("full", "tier1"):
        by_fold = results[label]["by_fold_calibration"]
        if by_fold is not None:
            by_fold.to_csv(out_dir / f"by_fold_calibration_{label}.csv", index=False)
        # Per-timestamp quantile bounds -- previously computed internally
        # and silently discarded. Needed so a later strategy backtest can
        # read Tier-1's frozen OOS bounds directly (docs/economic_contract_v1.md's
        # provenance rule: S5 must read a saved artifact, not recompute one).
        # Saved for "full" here too, for a self-contained pair, even though
        # uncertainty_selected_v1 already saves Full's bounds separately.
        results[label]["per_row"].to_csv(out_dir / f"quantile_forecasts_{label}.csv", index=False)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "xgboost_run_version": xgboost_run_version,
        "output_run_version": output_run_version,
        "parent_uncertainty_spec": uncertainty_run_version,
        "parent_uncertainty_spec_source": spec["source_manifest"],
        "window_days": spec["window_days"],
        "min_periods_days": spec["min_periods_days"],
        "quantiles": spec["quantiles"],
        "tuning_performed": False,
        "information_set_method": "delivery_day_safe",
        "holdout_used": False,
        "full_pooled_coverage": results["full"]["pooled_calibration"]["empirical_coverage"],
        "tier1_pooled_coverage": results["tier1"]["pooled_calibration"]["empirical_coverage"],
        "full_interval_score": results["full"]["interval_score"],
        "tier1_interval_score": results["tier1"]["interval_score"],
        "STANDING_CAVEAT": (
            "auction_sequence == 1 was independently confirmed via cross-check against SMARD "
            "(Germany official market data platform) across all 8,833 disagreeing corrected-data "
            "intervals -- 100% consistent with sequence 1, 0% with sequence 2. See README "
        ),
    }
    with open(out_dir / "tier1_robustness_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\nSaved comparison + manifest to {out_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
