"""Run locally: python run_uncertainty_tier1_regime_diagnostic.py <xgboost_run_version> <output_run_version>

Example:
  python run_uncertainty_tier1_regime_diagnostic.py xgboost_v1_a03fix uncertainty_tier1_regime_diagnostic_v1

DESCRIPTIVE-ONLY diagnostic, explicitly NOT a model-selection exercise
and produces NO promotion decision. Answers exactly one question:
Tier-1's regime_stress_test lower-tail miss rate is 45% lower than
Full's -- is that because Tier-1's uncertainty envelope is simply 55%
wider (mechanical), or because Tier-1's actual signed forecast errors
are less severely skewed downward in this specific regime
(structural)? Those require different interpretations and this script
does not assume either answer in advance.

The frozen 60-day uncertainty_selected_v1 specification is NOT
retuned, replaced, or reconsidered here regardless of what this
diagnostic finds -- see README "Uncertainty quantification" for why
that decision is already closed.

Two pieces, both restricted to regime_stress_test's exact timestamps
(the fold this whole diagnostic is about):

1. Signed residual distribution, Full vs Tier-1 -- are Tier-1's
   negative-tail errors genuinely less severe, or just as severe but
   swallowed by a wider interval?
2. The point-forecast/envelope CROSS: four combinations of
   {Full, Tier-1} point forecast x {Full, Tier-1} residual-quantile
   envelope width, all evaluated at the SAME q10 quantile. This
   isolates whether the lower-tail improvement comes from the point
   forecast location, the envelope width, or both -- the mechanical
   vs. structural question directly, not inferred from a proxy.

STANDING CAVEAT: inherits the same auction_sequence == 1 open
assumption as everything downstream of the price series -- see README
Limitations. Never touches 2026.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from src.models import TARGET_COL
from src.uncertainty import build_continuous_residual_series, compute_residual_quantile_offset
from src.utils import load_config, setup_logging, REPO_ROOT
from run_uncertainty_tier1_robustness import load_fold_predictions, load_frozen_uncertainty_spec, FOLD_NAMES, FULL_PRED_COL, TIER1_PRED_COL

REGIME_FOLD_NAME = "regime_stress_test"


def resolve_run_args(args: list) -> tuple:
    if len(args) != 2:
        raise SystemExit(
            "Usage:\n"
            "  python run_uncertainty_tier1_regime_diagnostic.py <xgboost_run_version> <output_run_version>\n\n"
            "Example:\n"
            "  python run_uncertainty_tier1_regime_diagnostic.py xgboost_v1_a03fix uncertainty_tier1_regime_diagnostic_v1"
        )
    return args[0], args[1]


def signed_residual_summary(residuals: pd.Series) -> dict:
    q = residuals.quantile([0.01, 0.05, 0.10, 0.25, 0.75, 0.90, 0.95, 0.99])
    worst_10pct = residuals.nsmallest(max(1, int(len(residuals) * 0.10)))
    worst_5pct = residuals.nsmallest(max(1, int(len(residuals) * 0.05)))
    return {
        "n": int(residuals.notna().sum()),
        "mean": float(residuals.mean()),
        "median": float(residuals.median()),
        "std": float(residuals.std()),
        "q01": float(q.loc[0.01]), "q05": float(q.loc[0.05]), "q10": float(q.loc[0.10]),
        "q25": float(q.loc[0.25]), "q75": float(q.loc[0.75]), "q90": float(q.loc[0.90]),
        "q95": float(q.loc[0.95]), "q99": float(q.loc[0.99]),
        "mean_worst_10pct": float(worst_10pct.mean()),
        "mean_worst_5pct": float(worst_5pct.mean()),
    }


def main():
    cfg = load_config()
    logger = setup_logging(cfg["logging"]["level"])

    xgboost_run_version, output_run_version = resolve_run_args(sys.argv[1:])
    input_stem = "delu_features"

    out_dir = REPO_ROOT / "outputs" / "uncertainty" / input_stem / output_run_version
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} already contains results. Pass a new output_run_version.")

    spec = load_frozen_uncertainty_spec(input_stem)
    logger.info("Using FROZEN specification: window_days=%d, min_periods_days=%d (no retuning)",
                spec["window_days"], spec["min_periods_days"])

    print("\n" + "=" * 78)
    print(f"TIER-1 REGIME DIAGNOSTIC (DESCRIPTIVE ONLY -- NO PROMOTION DECISION): {output_run_version}")
    print("=" * 78)

    residual_series = {}
    for label, pred_col in [("full", FULL_PRED_COL), ("tier1", TIER1_PRED_COL)]:
        fold_predictions = load_fold_predictions(input_stem, xgboost_run_version, pred_col)
        residual_series[label] = build_continuous_residual_series(
            fold_predictions, TARGET_COL, pred_col, fold_names=FOLD_NAMES
        )

    # Restrict to the exact regime_stress_test timestamps for both models.
    stress = {
        label: rs[rs["fold"] == REGIME_FOLD_NAME].sort_values("timestamp_utc").reset_index(drop=True)
        for label, rs in residual_series.items()
    }
    merged_stress = stress["full"][["timestamp_utc", TARGET_COL, "prediction", "residual"]].rename(
        columns={"prediction": "full_prediction", "residual": "full_residual"}
    ).merge(
        stress["tier1"][["timestamp_utc", "prediction", "residual"]].rename(
            columns={"prediction": "tier1_prediction", "residual": "tier1_residual"}
        ),
        on="timestamp_utc", how="inner",
    )
    logger.info("regime_stress_test common timestamps: %d", len(merged_stress))

    # --- 1. Signed residual distribution ---
    print("\n--- Signed residual distribution, regime_stress_test, Full vs Tier-1 ---")
    print("(residual = actual - prediction; more negative = model predicted too HIGH)")
    full_summary = signed_residual_summary(merged_stress["full_residual"])
    tier1_summary = signed_residual_summary(merged_stress["tier1_residual"])
    residual_comparison = pd.DataFrame([full_summary, tier1_summary], index=["full", "tier1"])
    print(residual_comparison.T.to_string())

    # --- 2. Point-forecast delta ---
    merged_stress["point_delta"] = merged_stress["tier1_prediction"] - merged_stress["full_prediction"]
    print(f"\n--- Point forecast delta (Tier-1 - Full), regime_stress_test ---")
    print(f"  mean={merged_stress['point_delta'].mean():.4f}, median={merged_stress['point_delta'].median():.4f}")

    full_lower_col_offset = compute_residual_quantile_offset(
        residual_series["full"], 0.1, spec["window_days"], spec["min_periods_days"]
    )
    full_lower_bound = merged_stress.merge(full_lower_col_offset, on="timestamp_utc", how="left")
    full_downside_misses = full_lower_bound[
        full_lower_bound[TARGET_COL] < (full_lower_bound["full_prediction"] + full_lower_bound["offset"])
    ]
    if len(full_downside_misses) > 0:
        on_miss_delta = full_downside_misses["point_delta"].mean()
        print(f"\n  On Full's own {len(full_downside_misses)} downside-miss timestamps specifically:")
        print(f"  mean point delta (Tier-1 - Full) = {on_miss_delta:.4f} "
              f"({'Tier-1 predicts LOWER' if on_miss_delta < 0 else 'Tier-1 predicts HIGHER'} than Full here)")

    # --- 3. The point x envelope cross ---
    print("\n--- Point forecast x envelope cross (all at q10), regime_stress_test ---")
    offsets = {}
    for label, rs in residual_series.items():
        offsets[label] = compute_residual_quantile_offset(rs, 0.1, spec["window_days"], spec["min_periods_days"])

    cross_results = []
    for point_label in ("full", "tier1"):
        for envelope_label in ("full", "tier1"):
            point_col = f"{point_label}_prediction"
            combo = merged_stress[["timestamp_utc", TARGET_COL, point_col]].merge(
                offsets[envelope_label], on="timestamp_utc", how="left"
            )
            lower_bound = combo[point_col] + combo["offset"]
            valid = combo[TARGET_COL].notna() & lower_bound.notna()
            miss_rate = float((combo[TARGET_COL][valid] < lower_bound[valid]).mean())
            cross_results.append({
                "point_forecast": point_label, "envelope": envelope_label,
                "n": int(valid.sum()), "lower_tail_miss_rate": miss_rate,
            })
    cross_df = pd.DataFrame(cross_results)
    print(cross_df.to_string(index=False))

    aa = cross_df[(cross_df.point_forecast == "full") & (cross_df.envelope == "full")]["lower_tail_miss_rate"].iloc[0]
    dd = cross_df[(cross_df.point_forecast == "tier1") & (cross_df.envelope == "tier1")]["lower_tail_miss_rate"].iloc[0]
    ab = cross_df[(cross_df.point_forecast == "full") & (cross_df.envelope == "tier1")]["lower_tail_miss_rate"].iloc[0]
    ba = cross_df[(cross_df.point_forecast == "tier1") & (cross_df.envelope == "full")]["lower_tail_miss_rate"].iloc[0]

    envelope_effect = aa - ab  # holding point forecast fixed at Full, swapping envelope width
    point_effect = aa - ba     # holding envelope fixed at Full, swapping point forecast

    print(f"\nDecomposition (both measured as reduction from A={aa:.4f}):")
    print(f"  Envelope-width effect alone (A -> B, point forecast held at Full): {envelope_effect:+.4f}")
    print(f"  Point-forecast effect alone (A -> C, envelope held at Full):       {point_effect:+.4f}")
    print(f"  Combined observed effect (A -> D, the actual robustness result):  {aa-dd:+.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    residual_comparison.T.to_csv(out_dir / "signed_residual_comparison.csv")
    cross_df.to_csv(out_dir / "point_envelope_cross.csv", index=False)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "xgboost_run_version": xgboost_run_version,
        "output_run_version": output_run_version,
        "purpose": "descriptive diagnostic only -- explains the Tier-1 robustness result, does not select or tune anything",
        "promotion_decision": None,
        "tuning_performed": False,
        "holdout_used": False,
        "window_days": spec["window_days"],
        "envelope_effect_on_lower_tail_miss_rate": envelope_effect,
        "point_forecast_effect_on_lower_tail_miss_rate": point_effect,
        "STANDING_CAVEAT": (
            "auction_sequence == 1 is a documented, still-open assumption pending external "
            "EPEX verification (see README Limitations)."
        ),
    }
    with open(out_dir / "regime_diagnostic_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\nSaved diagnostic outputs to {out_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
