"""Run locally: python run_final_holdout_report.py <output_run_version>

Example:
  python run_final_holdout_report.py holdout_report_v1

READ-ONLY final holdout reporting layer.

Consumes ONLY already-saved frozen holdout artifacts:
  outputs/holdout/delu_features/holdout_v1/
      hourly_holdout_inputs.csv
      per_day_results.csv
      holdout_manifest.json
  outputs/holdout/delu_features/holdout_sensitivity_v1/
      sensitivity_grid.csv
      per_day_sensitivity_results.csv
      sensitivity_manifest.json

It never reloads the processed 2026 feature parquet, never regenerates a
forecast or uncertainty interval, and never re-runs strategy selection.

Report sections:
  1. provenance/evaluation population;
  2. preregistered point-forecast metrics (MAE/RMSE/MedianAE) for lag24,
     lag168, XGBoost Full and Tier-1 on one identical finite row set;
  3. primary strategy table;
  4. four preregistered economic deltas;
  5. frozen 3x3 sensitivity grid + mechanical Section 8 verdict;
  6. uncertainty decomposition for S2<->S3 and S4<->S5, using the
     already-saved primary per-day decisions only;
  7. drawdown/downside;
  8. oracle/value capture;
  9. P&L concentration;
 10. market-regime/stress-day diagnostics from the saved hourly holdout
     actuals;
 11. explicit limitations.

No Sharpe ratio, annualized return, ROI, or capital-efficiency metric is
reported because the frozen study still has no capital/exposure model.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from src.models import TARGET_COL, compute_metrics
from src.utils import REPO_ROOT, load_config, setup_logging
from run_strategy_backtest import STRATEGIES
from run_strategy_decomposition import validate_backtest_results, decompose
from run_final_holdout_sensitivity import classify_section_8
from run_strategy_report import (
    CUTOVER_LOCAL_DATE,
    NAMED_DELTAS,
    compute_primary_table,
    compute_named_deltas,
    compute_drawdown,
    compute_concentration,
    compute_oracle_value_capture,
    compute_split,
)

INPUT_STEM = "delu_features"
HOLDOUT_RUN_VERSION = "holdout_v1"
SENSITIVITY_RUN_VERSION = "holdout_sensitivity_v1"

PROTOCOL_TAG = "pre-holdout-protocol-v2"
PROTOCOL_COMMIT = "a2d1f9a79022ca1e2731d52803061825038c69f5"
DEVELOPMENT_TAG = "corrected-development-v2-dayorigin"
DEVELOPMENT_COMMIT = "704cc933b99bb32c21cd358e613d1cd667d71488"
FINAL_MODEL_FIT_TAG = "final-model-fit-v1"
FINAL_MODEL_FIT_COMMIT = "54a7c2e77122134eca31a09798cc1614df1c634c"

HOLDOUT_START_LOCAL = "2026-01-01"
HOLDOUT_END_LOCAL_EXCLUSIVE = "2026-08-01"
PRIMARY_ETA_RT = 0.85
PRIMARY_C = 10.0
FLOAT_TOL = 1e-8

POINT_MODELS = {
    "lag24": "lag_24_pred",
    "lag168": "lag_168_pred",
    "xgboost_full": "xgboost_full_pred",
    "xgboost_tier1": "xgboost_tier1_pred",
}


def resolve_run_args(args: list) -> str:
    if len(args) != 1:
        raise SystemExit(
            "Usage:\n"
            "  python run_final_holdout_report.py <output_run_version>\n\n"
            "Example:\n"
            "  python run_final_holdout_report.py holdout_report_v1\n\n"
            "Source runs are frozen to holdout_v1 and "
            "holdout_sensitivity_v1."
        )
    return args[0]


def _coerce_common_flag(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    bad = sorted(set(normalized) - set(mapping))
    if bad:
        raise ValueError(
            f"Unrecognized common_evaluation_day value(s): {bad[:10]}"
        )
    return normalized.map(mapping).astype(bool)


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_frozen_artifacts() -> dict:
    holdout_dir = (
        REPO_ROOT / "outputs" / "holdout" / INPUT_STEM / HOLDOUT_RUN_VERSION
    )
    sensitivity_dir = (
        REPO_ROOT / "outputs" / "holdout" / INPUT_STEM / SENSITIVITY_RUN_VERSION
    )

    hourly_path = holdout_dir / "hourly_holdout_inputs.csv"
    per_day_path = holdout_dir / "per_day_results.csv"
    holdout_manifest_path = holdout_dir / "holdout_manifest.json"
    grid_path = sensitivity_dir / "sensitivity_grid.csv"
    sensitivity_manifest_path = sensitivity_dir / "sensitivity_manifest.json"

    for path in (
        hourly_path,
        per_day_path,
        holdout_manifest_path,
        grid_path,
        sensitivity_manifest_path,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}.")

    holdout_manifest = _read_json(holdout_manifest_path)
    sensitivity_manifest = _read_json(sensitivity_manifest_path)

    holdout_expected = {
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
        "development_evidence_tag": DEVELOPMENT_TAG,
        "development_evidence_commit": DEVELOPMENT_COMMIT,
        "final_model_fit_tag": FINAL_MODEL_FIT_TAG,
        "final_model_fit_commit": FINAL_MODEL_FIT_COMMIT,
    }
    problems = []
    for field, expected in holdout_expected.items():
        if holdout_manifest.get(field) != expected:
            problems.append(
                f"holdout {field}={holdout_manifest.get(field)!r}, "
                f"expected {expected!r}"
            )

    sensitivity_expected = {
        "source_holdout_run_version": HOLDOUT_RUN_VERSION,
        "base_case_reproduction": "PASSED",
        "classification_complete": True,
        "holdout_used": True,
        "n_grid_cells": 9,
        "source_holdout_protocol_commit": PROTOCOL_COMMIT,
        "source_final_model_fit_commit": FINAL_MODEL_FIT_COMMIT,
    }
    for field, expected in sensitivity_expected.items():
        if sensitivity_manifest.get(field) != expected:
            problems.append(
                f"sensitivity {field}={sensitivity_manifest.get(field)!r}, "
                f"expected {expected!r}"
            )

    verdict = sensitivity_manifest.get("section_8_verdict")
    if verdict not in {"CONFIRMATION", "PARTIAL CONFIRMATION", "FAILURE"}:
        problems.append(
            f"sensitivity section_8_verdict={verdict!r} is invalid"
        )

    if problems:
        raise ValueError(
            "FINAL HOLDOUT REPORT LINEAGE CHECK FAILED:\n  - "
            + "\n  - ".join(problems)
        )

    hourly = pd.read_csv(hourly_path)
    hourly["timestamp_utc"] = pd.to_datetime(
        hourly["timestamp_utc"], utc=True
    )
    required_hourly = [
        "timestamp_utc",
        TARGET_COL,
        "lag_24_pred",
        "lag_168_pred",
        "xgboost_full_pred",
        "xgboost_tier1_pred",
        "full_L",
        "full_U",
        "tier1_L",
        "tier1_U",
        "common_evaluation_day",
    ]
    missing = [c for c in required_hourly if c not in hourly.columns]
    if missing:
        raise ValueError(
            f"hourly_holdout_inputs.csv missing columns: {missing}"
        )
    hourly["common_evaluation_day"] = _coerce_common_flag(
        hourly["common_evaluation_day"]
    )

    per_day = pd.read_csv(per_day_path)
    per_day["delivery_date"] = per_day["delivery_date"].astype(str)

    grid = pd.read_csv(grid_path)
    if len(grid) != 9 or grid[["eta_rt", "c"]].duplicated().any():
        raise ValueError("Sensitivity grid is not exactly 9 unique cells.")

    # Recompute the mechanical Section 8 classification from the persisted
    # grid rather than trusting the sensitivity manifest's label.
    recomputed_verdict, recomputed_n_positive, recomputed_primary_delta = (
        classify_section_8(grid)
    )
    if sensitivity_manifest.get("section_8_verdict") != recomputed_verdict:
        raise ValueError(
            "Sensitivity manifest verdict does not reproduce the mechanical "
            f"classification from sensitivity_grid.csv: "
            f"{sensitivity_manifest.get('section_8_verdict')!r} != "
            f"{recomputed_verdict!r}."
        )
    if int(sensitivity_manifest.get("n_positive_delta_forecast", -1)) != int(
        recomputed_n_positive
    ):
        raise ValueError(
            "Sensitivity manifest positive-cell count does not reproduce "
            "sensitivity_grid.csv."
        )
    if not np.isclose(
        float(sensitivity_manifest.get("primary_delta_forecast", np.nan)),
        float(recomputed_primary_delta),
        rtol=0.0,
        atol=FLOAT_TOL,
    ):
        raise ValueError(
            "Sensitivity manifest primary_delta_forecast does not reproduce "
            "sensitivity_grid.csv."
        )

    if len(per_day) != int(holdout_manifest["n_common_holdout_days"]):
        raise ValueError(
            "per_day_results row count differs from holdout manifest "
            "n_common_holdout_days."
        )

    common_days_hourly = sorted(
        str(d)
        for d in pd.to_datetime(
            hourly.loc[
                hourly["common_evaluation_day"], "timestamp_utc"
            ],
            utc=True,
        )
        .dt.tz_convert("Europe/Berlin")
        .dt.date.unique()
    )
    saved_days = sorted(per_day["delivery_date"].tolist())
    if common_days_hourly != saved_days:
        raise ValueError(
            "Hourly common-day mask differs from per_day_results.csv."
        )

    primary_grid = grid[
        (np.isclose(grid["eta_rt"], PRIMARY_ETA_RT))
        & (np.isclose(grid["c"], PRIMARY_C))
    ]
    if len(primary_grid) != 1:
        raise ValueError(
            "Sensitivity grid must contain exactly one primary cell."
        )

    per_day_totals = {
        s: float(per_day[f"{s}_net_pnl"].sum()) for s in STRATEGIES
    }
    primary = primary_grid.iloc[0]
    for s in STRATEGIES:
        col = f"{s}_total_net_pnl"
        if col not in grid.columns:
            raise ValueError(f"Sensitivity grid is missing {col}.")
        if not np.isclose(
            float(primary[col]),
            per_day_totals[s],
            rtol=0.0,
            atol=FLOAT_TOL,
        ):
            raise ValueError(
                f"Primary sensitivity {col} does not reproduce "
                f"holdout per-day total."
            )

    return {
        "hourly": hourly,
        "per_day": per_day,
        "grid": grid,
        "holdout_manifest": holdout_manifest,
        "sensitivity_manifest": sensitivity_manifest,
        "holdout_dir": holdout_dir,
        "sensitivity_dir": sensitivity_dir,
    }


def compute_point_forecast_metrics(hourly: pd.DataFrame) -> tuple:
    required = [TARGET_COL] + list(POINT_MODELS.values())
    numeric = hourly[required].apply(pd.to_numeric, errors="coerce")
    finite_mask = np.isfinite(numeric.to_numpy()).all(axis=1)
    point_df = hourly.loc[finite_mask].copy()

    rows = []
    for model_name, pred_col in POINT_MODELS.items():
        metrics = compute_metrics(
            point_df[TARGET_COL].astype(float),
            point_df[pred_col].astype(float),
        )
        rows.append({"model": model_name, **metrics})

    metrics_df = pd.DataFrame(rows)
    if metrics_df["n"].nunique() != 1:
        raise AssertionError(
            "Point models were not evaluated on an identical row set."
        )
    return metrics_df, point_df


def attach_market_flags(
    per_day: pd.DataFrame,
    hourly: pd.DataFrame,
) -> pd.DataFrame:
    common = hourly[hourly["common_evaluation_day"]].copy()
    local = pd.to_datetime(common["timestamp_utc"], utc=True).dt.tz_convert(
        "Europe/Berlin"
    )
    common["delivery_date"] = local.dt.date.astype(str)

    stress = (
        common.groupby("delivery_date")[TARGET_COL]
        .agg(
            is_negative_price_day=lambda s: bool((s < 0).any()),
            is_gt200_day=lambda s: bool((s > 200).any()),
            is_gt500_day=lambda s: bool((s > 500).any()),
        )
        .reset_index()
    )

    out = per_day.merge(stress, on="delivery_date", how="left")
    if out[
        ["is_negative_price_day", "is_gt200_day", "is_gt500_day"]
    ].isna().any().any():
        raise ValueError(
            "Could not attach stress flags to every common holdout day."
        )
    out["is_post_cutover"] = out["delivery_date"] >= CUTOVER_LOCAL_DATE
    return out


def _fmt_eur(value) -> str:
    return f"EUR {float(value):,.2f}"


def _fmt_pct(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.1f}%"


def main():
    cfg = load_config()
    logger = setup_logging(cfg["logging"]["level"])
    output_run_version = resolve_run_args(sys.argv[1:])

    out_dir = (
        REPO_ROOT / "outputs" / "holdout" / INPUT_STEM / output_run_version
    )
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"{out_dir} already contains results. Use a new run version."
        )

    logger.info("Loading frozen holdout and sensitivity artifacts...")
    artifacts = load_frozen_artifacts()

    per_day = attach_market_flags(
        artifacts["per_day"], artifacts["hourly"]
    )
    grid = artifacts["grid"]

    logger.info("Computing point-forecast metrics on identical rows...")
    point_metrics, point_rows = compute_point_forecast_metrics(
        artifacts["hourly"]
    )

    logger.info("Validating and decomposing saved primary decisions...")
    validate_backtest_results(artifacts["per_day"])
    decomposition = {
        "S2_vs_S3": decompose(artifacts["per_day"], "S2", "S3"),
        "S4_vs_S5": decompose(artifacts["per_day"], "S4", "S5"),
    }

    primary_table = compute_primary_table(per_day)
    deltas = compute_named_deltas(per_day)

    drawdown = {
        s: compute_drawdown(per_day[f"{s}_net_pnl"])
        for s in STRATEGIES
    }
    concentration = {
        s: compute_concentration(per_day[f"{s}_net_pnl"])
        for s in STRATEGIES
    }
    oracle_capture = compute_oracle_value_capture(per_day)

    regime_split = compute_split(per_day, "is_post_cutover")
    stress_splits = {
        "negative_price_day": compute_split(
            per_day, "is_negative_price_day"
        ),
        "gt200_day": compute_split(per_day, "is_gt200_day"),
        "gt500_day": compute_split(per_day, "is_gt500_day"),
    }

    verdict = artifacts["sensitivity_manifest"]["section_8_verdict"]
    n_positive = int(
        artifacts["sensitivity_manifest"]["n_positive_delta_forecast"]
    )

    lines = []

    def emit(line=""):
        lines.append(str(line))

    emit("=" * 78)
    emit(f"FINAL 2026 HOLDOUT REPORT: {output_run_version}")
    emit(
        f"Sources: {HOLDOUT_RUN_VERSION} / {SENSITIVITY_RUN_VERSION}"
    )
    emit("=" * 78)

    emit("\n--- 1. PROVENANCE AND EVALUATION POPULATION ---")
    emit(
        f"Frozen holdout: {HOLDOUT_START_LOCAL} inclusive -> "
        f"{HOLDOUT_END_LOCAL_EXCLUSIVE} exclusive"
    )
    emit(
        f"Raw holdout delivery days: "
        f"{artifacts['holdout_manifest']['raw_holdout_delivery_days']}"
    )
    emit(f"Common economic evaluation days: {len(per_day)}")
    emit(f"Identical point-evaluation rows: {len(point_rows)}")
    emit(
        f"Primary economics: eta_rt={PRIMARY_ETA_RT}, c=EUR {PRIMARY_C:.0f}"
    )
    emit(
        f"Protocol commit: {PROTOCOL_COMMIT}; "
        f"final model fit commit: {FINAL_MODEL_FIT_COMMIT}"
    )

    emit("\n--- 2. POINT-FORECAST HOLDOUT METRICS ---")
    emit(
        point_metrics[
            ["model", "mae", "rmse", "median_ae", "n"]
        ].to_string(index=False)
    )

    emit("\n--- 3. PRIMARY STRATEGY TABLE ---")
    display = primary_table.copy()
    for col in (
        "gross_pnl",
        "net_pnl",
        "mean_daily_pnl",
        "median_daily_pnl",
        "worst_day",
    ):
        display[col] = display[col].map(lambda x: f"{x:,.2f}")
    display["profitable_day_rate"] = display[
        "profitable_day_rate"
    ].map(_fmt_pct)
    display["trade_hit_rate"] = display["trade_hit_rate"].map(_fmt_pct)
    emit(display.to_string(index=False))

    emit("\n--- 4. FOUR PREREGISTERED ECONOMIC DELTAS ---")
    for name, _, _, label in NAMED_DELTAS:
        emit(f"{label}: {_fmt_eur(deltas[name])}")

    emit("\n--- 5. FROZEN 3x3 SENSITIVITY / SECTION 8 VERDICT ---")
    emit(
        grid[
            [
                "eta_rt",
                "c",
                "delta_forecast",
                "delta_uncertainty",
                "delta_tier1",
                "delta_tier1_u",
            ]
        ].to_string(index=False)
    )
    emit(f"delta_forecast positive cells: {n_positive}/9")
    emit(f"MECHANICAL SECTION 8 VERDICT: {verdict}")

    emit("\n--- 6. UNCERTAINTY DECOMPOSITION ---")
    for label, result in decomposition.items():
        emit(f"{label}:")
        for key in (
            "point_strategy",
            "uncertainty_strategy",
            "profits_forgone",
            "losses_avoided",
            "pair_selection_effect",
            "reverse_category_effect",
            "same_pair_difference",
            "actual_gap",
            "reconciled_gap",
        ):
            if key in result:
                emit(f"  {key}: {result[key]}")
        if "category_counts" in result:
            emit(f"  category_counts: {result['category_counts']}")

    emit("\n--- 7. DRAWDOWN / DOWNSIDE ---")
    for s in STRATEGIES:
        d = drawdown[s]
        emit(
            f"{s}: worst_day={_fmt_eur(d['worst_day'])}; "
            f"worst_rolling_5day="
            f"{_fmt_eur(d['worst_rolling_5day']) if d['worst_rolling_5day'] is not None else 'N/A'}; "
            f"max_drawdown={_fmt_eur(d['max_drawdown'])}"
        )

    emit("\n--- 8. ORACLE / VALUE CAPTURE ---")
    emit(f"Oracle total: {_fmt_eur(oracle_capture['oracle_total'])}")
    for s in STRATEGIES:
        emit(f"{s} value-capture ratio: {_fmt_pct(oracle_capture[f'{s}_vcr'])}")

    emit("\n--- 9. P&L CONCENTRATION ---")
    for s in STRATEGIES:
        c = concentration[s]
        emit(
            f"{s}: top1={_fmt_pct(c['top_1pct_pct_of_total'])}, "
            f"top5={_fmt_pct(c['top_5pct_pct_of_total'])}, "
            f"top10={_fmt_pct(c['top_10pct_pct_of_total'])}"
        )

    emit("\n--- 10. REGIME / STRESS-DAY DIAGNOSTICS ---")
    emit(f"Post-15min cutover split: {regime_split}")
    for name, split in stress_splits.items():
        emit(f"{name}: {split}")

    emit("\n--- 11. LIMITATIONS ---")
    emit(
        "- Price-taking, stylized single-cycle day-ahead storage scheduling; "
        "no market impact."
    )
    emit(
        "- Market transaction costs remain excluded (Cmarket=0); degradation "
        "cost follows the frozen contract."
    )
    emit(
        "- Tier-2 wind/solar point-in-time public availability by 11:45 D-1 "
        "remains unresolved; Tier-1 is the robustness information set."
    )
    emit(
        "- No capital/exposure model: no ROI, Sharpe, annualized return, or "
        "capital-efficiency claim."
    )
    emit(
        "- Holdout currently covers 2026-01-01 through 2026-07-31 delivery "
        "dates only, per the frozen structural boundary."
    )
    emit(
        "- Tail-risk precision, especially at 99%, is reported separately "
        "by run_final_holdout_tailrisk.py."
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    point_metrics.to_csv(
        out_dir / "point_forecast_metrics.csv", index=False
    )
    primary_table.to_csv(
        out_dir / "primary_strategy_table.csv", index=False
    )
    grid.to_csv(out_dir / "sensitivity_grid.csv", index=False)

    with open(
        out_dir / "decomposition_results.json", "w", encoding="utf-8"
    ) as f:
        json.dump(decomposition, f, indent=2, default=str)

    diagnostics = {
        "named_deltas": deltas,
        "drawdown": drawdown,
        "concentration": concentration,
        "oracle_value_capture": oracle_capture,
        "regime_split": regime_split,
        "stress_splits": stress_splits,
    }
    with open(
        out_dir / "holdout_diagnostics.json", "w", encoding="utf-8"
    ) as f:
        json.dump(diagnostics, f, indent=2, default=str)

    report_text = "\n".join(lines) + "\n"
    (out_dir / "holdout_report.txt").write_text(
        report_text, encoding="utf-8"
    )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_run_version": output_run_version,
        "source_holdout_run_version": HOLDOUT_RUN_VERSION,
        "source_sensitivity_run_version": SENSITIVITY_RUN_VERSION,
        "source_holdout_used": True,
        "source_holdout_structural_invariants": "PASSED",
        "source_sensitivity_base_case_reproduction": "PASSED",
        "source_sensitivity_classification_complete": True,
        "protocol_commit": PROTOCOL_COMMIT,
        "final_model_fit_commit": FINAL_MODEL_FIT_COMMIT,
        "n_common_economic_days": int(len(per_day)),
        "n_point_evaluation_rows": int(len(point_rows)),
        "section_8_verdict": verdict,
        "n_positive_delta_forecast_cells": n_positive,
        "holdout_used": True,
        "decision_recomputation_performed": False,
        "forecast_recomputation_performed": False,
    }
    with open(
        out_dir / "holdout_report_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(manifest, f, indent=2, default=str)

    print(report_text, end="")
    print(f"\nSaved final holdout report to {out_dir}")


if __name__ == "__main__":
    main()
