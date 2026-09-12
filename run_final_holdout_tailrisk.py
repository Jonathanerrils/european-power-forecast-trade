"""Run locally: python run_final_holdout_tailrisk.py <output_run_version>

Example:
  python run_final_holdout_tailrisk.py holdout_tailrisk_v1

READ-ONLY 2026 holdout tail-risk layer.

Reads ONLY the already-saved primary-cell holdout_v1 per_day_results.csv
and holdout_manifest.json. It never recomputes a strategy decision and
never reloads the raw 2026 feature parquet.

Loss is the same frozen strategy-loss definition used in development:

    L_D,s = -Pi_D,s

Confidence levels and method are unchanged:
  - empirical/historical VaR and ES
  - 95% and 99%
  - exact worst-m observation-equivalent ES
  - LOW PRECISION flag when effective tail size < 20 observations

The small 2026 holdout sample makes the precision warning particularly
important: 99% ES will necessarily rest on only a few observation
equivalents.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from src.utils import REPO_ROOT, load_config, setup_logging
from run_strategy_backtest import STRATEGIES
from run_strategy_tailrisk import (
    CONFIDENCE_LEVELS,
    LOW_PRECISION_TAIL_THRESHOLD,
    compute_loss_series,
    compute_empirical_var_es,
)

INPUT_STEM = "delu_features"
HOLDOUT_RUN_VERSION = "holdout_v1"

PROTOCOL_TAG = "pre-holdout-protocol-v2"
PROTOCOL_COMMIT = "a2d1f9a79022ca1e2731d52803061825038c69f5"
FINAL_MODEL_FIT_TAG = "final-model-fit-v1"
FINAL_MODEL_FIT_COMMIT = "54a7c2e77122134eca31a09798cc1614df1c634c"

HOLDOUT_START_LOCAL = "2026-01-01"
HOLDOUT_END_LOCAL_EXCLUSIVE = "2026-08-01"
PRIMARY_ETA_RT = 0.85
PRIMARY_C = 10.0


def resolve_run_args(args: list) -> str:
    if len(args) != 1:
        raise SystemExit(
            "Usage:\n"
            "  python run_final_holdout_tailrisk.py <output_run_version>\n\n"
            "Example:\n"
            "  python run_final_holdout_tailrisk.py holdout_tailrisk_v1\n\n"
            "Source is frozen to holdout_v1."
        )
    return args[0]


def load_holdout_pnl() -> tuple:
    holdout_dir = (
        REPO_ROOT / "outputs" / "holdout" / INPUT_STEM / HOLDOUT_RUN_VERSION
    )
    per_day_path = holdout_dir / "per_day_results.csv"
    manifest_path = holdout_dir / "holdout_manifest.json"

    if not per_day_path.exists():
        raise FileNotFoundError(f"Missing {per_day_path}.")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {manifest_path}.")

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    expected = {
        "output_run_version": HOLDOUT_RUN_VERSION,
        "holdout_used": True,
        "classification_complete": False,
        "holdout_start_local": HOLDOUT_START_LOCAL,
        "holdout_end_local_exclusive": HOLDOUT_END_LOCAL_EXCLUSIVE,
        "eta_rt_primary": PRIMARY_ETA_RT,
        "c_primary": PRIMARY_C,
        "strategies": list(STRATEGIES),
        "structural_invariants": "PASSED",
        "exact_raw_holdout_coverage": "PASSED",
        "protocol_tag": PROTOCOL_TAG,
        "protocol_commit": PROTOCOL_COMMIT,
        "final_model_fit_tag": FINAL_MODEL_FIT_TAG,
        "final_model_fit_commit": FINAL_MODEL_FIT_COMMIT,
    }
    problems = []
    for field, expected_value in expected.items():
        if manifest.get(field) != expected_value:
            problems.append(
                f"{field}={manifest.get(field)!r}, "
                f"expected {expected_value!r}"
            )
    if problems:
        raise ValueError(
            "HOLDOUT TAIL-RISK SOURCE CHECK FAILED:\n  - "
            + "\n  - ".join(problems)
        )

    per_day = pd.read_csv(per_day_path)
    return per_day, manifest, holdout_dir


def validate_holdout_pnl(
    per_day: pd.DataFrame,
    manifest: dict,
) -> None:
    required = ["delivery_date", "n_hours"] + [
        f"{s}_net_pnl" for s in STRATEGIES
    ]
    missing = [c for c in required if c not in per_day.columns]
    if missing:
        raise ValueError(
            f"per_day_results.csv missing required columns: {missing}"
        )

    problems = []

    if per_day["delivery_date"].isna().any():
        problems.append("delivery_date contains missing value(s)")

    dates = pd.to_datetime(
        per_day["delivery_date"], errors="coerce"
    )
    if dates.isna().any():
        problems.append("delivery_date contains invalid value(s)")
    if per_day["delivery_date"].astype(str).duplicated().any():
        problems.append("delivery_date contains duplicate day(s)")

    start = pd.Timestamp(HOLDOUT_START_LOCAL)
    end = pd.Timestamp(HOLDOUT_END_LOCAL_EXCLUSIVE)
    if not dates.isna().any():
        outside = (dates < start) | (dates >= end)
        if outside.any():
            problems.append(
                f"{int(outside.sum())} delivery date(s) outside frozen "
                "holdout window"
            )

    n_hours = pd.to_numeric(
        per_day["n_hours"], errors="coerce"
    )
    bad_hours = (
        ~np.isfinite(n_hours)
        | (n_hours % 1 != 0)
        | ~n_hours.isin([23, 24, 25])
    )
    if bad_hours.any():
        problems.append(
            f"n_hours has {int(bad_hours.sum())} invalid value(s)"
        )

    for s in STRATEGIES:
        pnl = pd.to_numeric(
            per_day[f"{s}_net_pnl"], errors="coerce"
        )
        if not np.isfinite(pnl).all():
            problems.append(
                f"{s}_net_pnl contains non-finite/non-numeric value(s)"
            )

    s0 = pd.to_numeric(
        per_day["S0_net_pnl"], errors="coerce"
    )
    if np.isfinite(s0).all() and (s0 != 0).any():
        problems.append("S0_net_pnl must be exactly zero on every day")

    if len(per_day) != int(manifest.get("n_common_holdout_days", -1)):
        problems.append(
            "per_day row count differs from manifest n_common_holdout_days"
        )

    if problems:
        raise ValueError(
            "HOLDOUT P&L VALIDATION FAILED:\n  - "
            + "\n  - ".join(problems)
        )


def main():
    cfg = load_config()
    logger = setup_logging(cfg["logging"]["level"])
    output_run_version = resolve_run_args(sys.argv[1:])

    out_dir = (
        REPO_ROOT / "outputs" / "risk" / INPUT_STEM / output_run_version
    )
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"{out_dir} already contains results. Use a new run version."
        )

    logger.info("Loading frozen holdout_v1 primary daily P&L...")
    per_day, manifest, _ = load_holdout_pnl()

    logger.info("Validating holdout P&L before VaR/ES...")
    validate_holdout_pnl(per_day, manifest)

    rows = []
    descriptive = []

    for strategy in STRATEGIES:
        loss = compute_loss_series(per_day, strategy).astype(float)
        descriptive.append(
            {
                "strategy": strategy,
                "n_days": int(len(loss)),
                "mean_daily_loss": float(loss.mean()),
                "median_daily_loss": float(loss.median()),
                "worst_single_day_loss": float(loss.max()),
                "best_single_day_loss": float(loss.min()),
            }
        )

        for confidence_level in CONFIDENCE_LEVELS:
            result = compute_empirical_var_es(
                loss, confidence_level
            )
            rows.append({"strategy": strategy, **result})

    results_df = pd.DataFrame(rows)
    descriptive_df = pd.DataFrame(descriptive)

    out_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(
        out_dir / "var_es_results.csv", index=False
    )
    descriptive_df.to_csv(
        out_dir / "tailrisk_descriptive.csv", index=False
    )

    manifest_out = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_run_version": output_run_version,
        "source_holdout_run_version": HOLDOUT_RUN_VERSION,
        "source_holdout_used": True,
        "source_holdout_structural_invariants": "PASSED",
        "protocol_commit": PROTOCOL_COMMIT,
        "final_model_fit_commit": FINAL_MODEL_FIT_COMMIT,
        "confidence_levels": list(CONFIDENCE_LEVELS),
        "method": "empirical_historical",
        "loss_definition": "L_D,s = -Pi_D,s",
        "n_common_holdout_days": int(len(per_day)),
        "low_precision_tail_threshold": LOW_PRECISION_TAIL_THRESHOLD,
        "holdout_used": True,
        "decision_recomputation_performed": False,
        "forecast_recomputation_performed": False,
    }
    with open(
        out_dir / "tailrisk_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(manifest_out, f, indent=2, default=str)

    print("\n" + "=" * 78)
    print(
        f"FINAL 2026 HOLDOUT TAIL RISK: {output_run_version} "
        f"(source: {HOLDOUT_RUN_VERSION})"
    )
    print(
        "Loss = -strategy net P&L. Method = empirical/historical. "
        f"n={len(per_day)} common days."
    )
    print("=" * 78)

    for strategy in STRATEGIES:
        print(f"\n--- {strategy} ---")
        d = descriptive_df[
            descriptive_df["strategy"] == strategy
        ].iloc[0]
        print(
            f"Mean daily loss: EUR {d['mean_daily_loss']:.2f}; "
            f"worst single-day loss: EUR "
            f"{d['worst_single_day_loss']:.2f}"
        )
        subset = results_df[
            results_df["strategy"] == strategy
        ]
        for _, row in subset.iterrows():
            flag = (
                " [LOW PRECISION -- effective tail size < 20]"
                if bool(row["low_precision"])
                else ""
            )
            print(
                f"VaR_{row['confidence_level']*100:.0f}: "
                f"EUR {row['var']:.2f}; "
                f"ES_{row['confidence_level']*100:.0f}: "
                f"EUR {row['es']:.2f}; "
                f"effective tail={row['n_tail']:.2f}/"
                f"{int(row['n_total'])}{flag}"
            )

    print("\nLIMITATIONS:")
    print(
        "- Empirical/historical point estimates only; no new conditional "
        "risk model or post-holdout model selection."
    )
    print(
        "- 99% estimates are expected to be low precision because the "
        "holdout contains only about seven months of common delivery days."
    )
    print(
        "- Daily losses may be serially clustered; no i.i.d. significance "
        "claim is made."
    )
    print(
        "- This is strategy loss risk, not forecast-residual uncertainty."
    )
    print("=" * 78)
    print(f"Saved holdout VaR/ES evidence to {out_dir}")


if __name__ == "__main__":
    main()
