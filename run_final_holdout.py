"""Run locally: python run_final_holdout.py <output_run_version>

Example:
  python run_final_holdout.py holdout_v1

FROZEN holdout-exposure runner for docs/holdout_protocol_v2.md.

Only output_run_version is user-supplied. The dataset, development OOS
residual source, final model artifacts, holdout boundary, uncertainty
specification, and primary economics are exact frozen constants.

For each raw holdout delivery day D, in chronological order:
  1. generate Full/Tier-1 point forecasts from frozen final models;
  2. compute one delivery-day-safe q10/q50/q90 residual snapshot using
     only residuals from delivery dates strictly before D;
  3. materialize D's point forecasts and bounds;
  4. only then construct D realized residuals;
  5. append D residuals for D+1 onward.

This sequential loop processes EVERY raw holdout day before the common
strategy-evaluation mask is applied. A day later excluded from economic
evaluation still legitimately contributes every finite revealed
final-model residual to later uncertainty snapshots.

The development function build_common_evaluation_days() is then reused
verbatim. Economic P&L is calculated only on common days. No 2026
economic figure is printed or saved until provenance, exact raw coverage,
common-mask construction, and structural economic invariants have passed.

hourly_holdout_inputs.csv is deliberately saved: it is the sole 2026
input for the later frozen report / 3x3 sensitivity step, so later scripts
need not reload the raw 2026 feature parquet or regenerate forecasts.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.clean import add_local_time_columns, local_delivery_date_to_utc
from src.models import TARGET_COL, baseline_predictions
from src.splits import get_holdout_window, slice_window
from src.strategy import assert_no_holdout_access
from src.uncertainty import (
    build_continuous_residual_series,
    residual_quantile_offsets_for_delivery_day,
)
from src.utils import REPO_ROOT, load_config, setup_logging
from src.xgboost_model import predict_xgboost
from run_uncertainty_tier1_robustness import (
    FOLD_NAMES,
    FULL_PRED_COL,
    TIER1_PRED_COL,
    load_fold_predictions,
)
from run_strategy_backtest import (
    C_PRIMARY,
    ETA_RT_PRIMARY,
    STRATEGIES,
    build_common_evaluation_days,
    run_backtest_for_day,
    verify_structural_invariants,
)

INPUT_FILENAME = "delu_features.parquet"
INPUT_STEM = "delu_features"

XGBOOST_RUN_VERSION = "xgboost_v1_a03fix"
UNCERTAINTY_SELECTED_RUN_VERSION = "uncertainty_selected_v2_dayorigin"
TIER1_UNCERTAINTY_RUN_VERSION = "uncertainty_tier1_robustness_v3_dayorigin"
FINAL_MODEL_FIT_RUN_VERSION = "final_model_fit_v1"

HOLDOUT_START_LOCAL = "2026-01-01"
HOLDOUT_END_LOCAL_EXCLUSIVE = "2026-08-01"

WINDOW_DAYS = 60
MIN_PERIODS_DAYS = 15
QUANTILES = [0.1, 0.5, 0.9]
INFORMATION_SET_METHOD = "delivery_day_safe"

PROTOCOL_TAG = "pre-holdout-protocol-v2"
PROTOCOL_COMMIT = "a2d1f9a79022ca1e2731d52803061825038c69f5"
DEVELOPMENT_TAG = "corrected-development-v2-dayorigin"
DEVELOPMENT_COMMIT = "704cc933b99bb32c21cd358e613d1cd667d71488"
FINAL_MODEL_FIT_IMPLEMENTATION_TAG = "final-model-fit-implementation-v1"
FINAL_MODEL_FIT_IMPLEMENTATION_COMMIT = "9f9db6bef9a7c8f4dd04ec7547f7d43684c6627f"
FINAL_MODEL_FIT_TAG = "final-model-fit-v1"
FINAL_MODEL_FIT_COMMIT = "54a7c2e77122134eca31a09798cc1614df1c634c"

EXPECTED_FINAL_FIT_SHA256 = {
    "final_model_fit_manifest.json": "EF28341A90B4427FE72DA15F06C1F851B8840FADF89FD83D4CDB4280A218EEBD",
    "full_hyperparameters.json": "BF1C02426BED6B49E1B4EF0EBD444B77E496E202B3E7230FFF0CC1ED31E26987",
    "full_model.json": "1AC9F94FC60DF835CB53D66966847F464D22FFD0815B9757EB8A2C281893F666",
    "full_predictor_cols.json": "75C7ECAA88F130096358BE14675211E371845CBC7394F820949C2EBFE639A843",
    "tier1_hyperparameters.json": "990731D56663EF704231BD1E255ED831F824D321165D7AADEEF062343EB14A69",
    "tier1_model.json": "B4C9D22A7075CCF01F58A0CCAF3946F898E8D19EA3C3B3AA9A8AE43EFD1ED025",
    "tier1_predictor_cols.json": "4D3531BB1A760FB538BAE54C26AAC72A110A83FBABE24D0568D6218B71C6E339",
}


def resolve_run_args(args: list) -> str:
    if len(args) != 1:
        raise SystemExit(
            "Usage:\n"
            "  python run_final_holdout.py <output_run_version>\n\n"
            "Example:\n"
            "  python run_final_holdout.py holdout_v1\n\n"
            "All scientifically consequential inputs are frozen constants."
        )
    return args[0]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def assert_frozen_economic_contract() -> None:
    if ETA_RT_PRIMARY != 0.85:
        raise AssertionError(f"ETA_RT_PRIMARY drifted: {ETA_RT_PRIMARY!r}.")
    if C_PRIMARY != 10.0:
        raise AssertionError(f"C_PRIMARY drifted: {C_PRIMARY!r}.")
    if list(STRATEGIES) != ["S0", "S1", "S2", "S3", "S4", "S5"]:
        raise AssertionError(f"Frozen strategy set drifted: {STRATEGIES!r}.")


def _parse_sha256sums(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen checksum record {path}.")
    parsed = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            digest, filename = line.split(maxsplit=1)
        except ValueError as exc:
            raise ValueError(
                f"Malformed checksum line in {path}: {raw_line!r}"
            ) from exc
        parsed[filename.strip()] = digest.upper()
    return parsed


def load_final_models() -> dict:
    fit_dir = (
        REPO_ROOT
        / "outputs"
        / "final_model_fit"
        / INPUT_STEM
        / FINAL_MODEL_FIT_RUN_VERSION
    )
    manifest_path = fit_dir / "final_model_fit_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {manifest_path}.")

    checksum_record = _parse_sha256sums(fit_dir / "SHA256SUMS.txt")
    if checksum_record != EXPECTED_FINAL_FIT_SHA256:
        raise ValueError(
            "SHA256SUMS.txt does not match the exact frozen final-model-fit "
            "artifact set."
        )

    for filename, expected in EXPECTED_FINAL_FIT_SHA256.items():
        path = fit_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing frozen final-fit artifact {path}.")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"Frozen final-fit artifact hash mismatch for {filename}: "
                f"{actual} != {expected}."
            )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_exact = {
        "input_filename": INPUT_FILENAME,
        "output_run_version": FINAL_MODEL_FIT_RUN_VERSION,
        "train_start_local_date": "2019-01-01",
        "train_end_local_date_exclusive": HOLDOUT_START_LOCAL,
        "target_col": TARGET_COL,
        "param_grid_n_combinations": 36,
        "inner_cv_splits": 3,
        "day_aligned_cv": True,
        "scoring": "neg_mean_absolute_error",
        "random_seed": 42,
        "xgboost_objective": "reg:absoluteerror",
        "xgboost_tree_method": "hist",
        "selection_independent_per_information_set": True,
        "holdout_used": False,
        "protocol_tag": PROTOCOL_TAG,
        "protocol_commit": PROTOCOL_COMMIT,
        "development_evidence_tag": DEVELOPMENT_TAG,
        "development_evidence_commit": DEVELOPMENT_COMMIT,
        "xgboost_development_run": XGBOOST_RUN_VERSION,
    }
    problems = []
    for field, expected in required_exact.items():
        actual = manifest.get(field)
        if actual != expected:
            problems.append(f"{field}={actual!r}, expected {expected!r}")
    if problems:
        raise ValueError(
            "Frozen final-model-fit manifest lineage/contract check failed:\n  - "
            + "\n  - ".join(problems)
        )

    models = {}
    for label in ("full", "tier1"):
        model = XGBRegressor()
        model.load_model(str(fit_dir / f"{label}_model.json"))
        predictor_cols = json.loads(
            (fit_dir / f"{label}_predictor_cols.json").read_text(
                encoding="utf-8"
            )
        )
        if (
            predictor_cols != manifest.get(f"{label}_predictor_cols")
            or predictor_cols
            != manifest.get(f"frozen_{label}_predictor_cols")
        ):
            raise ValueError(
                f"{label} predictor-list artifact does not match the "
                "frozen fit manifest."
            )
        models[label] = {
            "model": model,
            "predictor_cols": predictor_cols,
        }

    return {"models": models, "manifest": manifest, "fit_dir": fit_dir}


def verify_development_lineage(input_path: Path) -> dict:
    model_manifest_path = (
        REPO_ROOT
        / "outputs"
        / "models"
        / INPUT_STEM
        / XGBOOST_RUN_VERSION
        / "model_run_manifest.json"
    )
    if not model_manifest_path.exists():
        raise FileNotFoundError(f"Missing {model_manifest_path}.")
    model_manifest = json.loads(
        model_manifest_path.read_text(encoding="utf-8")
    )

    current_input_sha = sha256_file(input_path)
    model_input_sha = str(model_manifest.get("input_sha256", "")).upper()

    problems = []
    for field, expected in {
        "run_version": XGBOOST_RUN_VERSION,
        "target": TARGET_COL,
        "xgboost_day_aligned_cv": True,
        "holdout_used": False,
    }.items():
        if model_manifest.get(field) != expected:
            problems.append(
                f"{field}={model_manifest.get(field)!r}, expected {expected!r}"
            )
    if current_input_sha != model_input_sha:
        problems.append(
            "current delu_features.parquet SHA256 differs from the exact "
            "input SHA256 recorded by xgboost_v1_a03fix"
        )
    if problems:
        raise ValueError(
            "Development XGBoost lineage check failed:\n  - "
            + "\n  - ".join(problems)
        )

    full_manifest_path = (
        REPO_ROOT
        / "outputs"
        / "uncertainty"
        / INPUT_STEM
        / UNCERTAINTY_SELECTED_RUN_VERSION
        / "uncertainty_run_manifest.json"
    )
    if not full_manifest_path.exists():
        raise FileNotFoundError(f"Missing {full_manifest_path}.")
    full_manifest = json.loads(
        full_manifest_path.read_text(encoding="utf-8")
    )

    tier1_manifest_path = (
        REPO_ROOT
        / "outputs"
        / "uncertainty"
        / INPUT_STEM
        / TIER1_UNCERTAINTY_RUN_VERSION
        / "tier1_robustness_manifest.json"
    )
    if not tier1_manifest_path.exists():
        raise FileNotFoundError(f"Missing {tier1_manifest_path}.")
    tier1_manifest = json.loads(
        tier1_manifest_path.read_text(encoding="utf-8")
    )

    u_problems = []
    for field, expected in {
        "xgboost_run_version": XGBOOST_RUN_VERSION,
        "window_days": WINDOW_DAYS,
        "min_periods_days": MIN_PERIODS_DAYS,
        "quantiles": QUANTILES,
        "information_set_method": INFORMATION_SET_METHOD,
    }.items():
        if full_manifest.get(field) != expected:
            u_problems.append(
                f"Full uncertainty {field}={full_manifest.get(field)!r}, "
                f"expected {expected!r}"
            )

    # The genuine frozen Full uncertainty manifest predates the
    # holdout_used field convention. It instead records the exact residual
    # series end; require that end to be strictly before the holdout boundary.
    residual_end = pd.to_datetime(
        full_manifest.get("residual_series_end"), utc=True, errors="coerce"
    )
    holdout_start_utc = pd.Timestamp(
        local_delivery_date_to_utc(HOLDOUT_START_LOCAL)
    )
    if pd.isna(residual_end) or residual_end >= holdout_start_utc:
        u_problems.append(
            "Full uncertainty residual_series_end is missing/invalid or "
            "not strictly before the frozen holdout boundary"
        )

    for field, expected in {
        "xgboost_run_version": XGBOOST_RUN_VERSION,
        "parent_uncertainty_spec": UNCERTAINTY_SELECTED_RUN_VERSION,
        "window_days": WINDOW_DAYS,
        "min_periods_days": MIN_PERIODS_DAYS,
        "quantiles": QUANTILES,
        "information_set_method": INFORMATION_SET_METHOD,
        "holdout_used": False,
    }.items():
        if tier1_manifest.get(field) != expected:
            u_problems.append(
                f"Tier-1 uncertainty {field}={tier1_manifest.get(field)!r}, "
                f"expected {expected!r}"
            )
    if u_problems:
        raise ValueError(
            "Frozen uncertainty lineage check failed:\n  - "
            + "\n  - ".join(u_problems)
        )

    return {
        "current_input_sha256": current_input_sha,
        "development_model_manifest": model_manifest,
        "full_uncertainty_manifest": full_manifest,
        "tier1_uncertainty_manifest": tier1_manifest,
    }


def build_initial_residual_history(pred_col: str) -> pd.DataFrame:
    fold_predictions = load_fold_predictions(
        INPUT_STEM, XGBOOST_RUN_VERSION, pred_col
    )
    history = build_continuous_residual_series(
        fold_predictions,
        TARGET_COL,
        pred_col,
        fold_names=FOLD_NAMES,
    )
    assert_no_holdout_access(history["timestamp_utc"])
    return history


def assert_exact_holdout_coverage(holdout_df: pd.DataFrame) -> dict:
    if "timestamp_utc" not in holdout_df.columns:
        raise ValueError("Holdout dataframe has no timestamp_utc column.")
    if TARGET_COL not in holdout_df.columns:
        raise ValueError(f"Holdout dataframe has no {TARGET_COL!r} column.")

    ts = pd.to_datetime(holdout_df["timestamp_utc"], utc=True)
    if ts.duplicated().any():
        raise ValueError("Holdout data contains duplicate timestamps.")

    start = pd.Timestamp(local_delivery_date_to_utc(HOLDOUT_START_LOCAL))
    end = pd.Timestamp(
        local_delivery_date_to_utc(HOLDOUT_END_LOCAL_EXCLUSIVE)
    )
    expected = pd.date_range(start, end, freq="1h", inclusive="left")
    observed = pd.DatetimeIndex(ts.sort_values().reset_index(drop=True))

    if not observed.equals(expected):
        expected_set = set(expected)
        observed_set = set(observed)
        missing = expected_set - observed_set
        extra = observed_set - expected_set
        raise ValueError(
            "Holdout timestamp coverage does not exactly match the frozen "
            f"window. expected={len(expected)}, observed={len(observed)}, "
            f"missing={len(missing)}, extra={len(extra)}."
        )

    target = pd.to_numeric(holdout_df[TARGET_COL], errors="coerce")
    if target.isna().any() or not np.isfinite(target.to_numpy()).all():
        raise ValueError(
            "Holdout target contains missing/non-finite values."
        )

    local = add_local_time_columns(
        holdout_df[["timestamp_utc"]].copy()
    )
    raw_days = sorted(local["delivery_date"].unique())
    expected_days = [
        d.date()
        for d in pd.date_range(
            HOLDOUT_START_LOCAL,
            pd.Timestamp(HOLDOUT_END_LOCAL_EXCLUSIVE)
            - pd.Timedelta(days=1),
            freq="D",
        )
    ]
    if raw_days != expected_days:
        raise ValueError(
            "Holdout local delivery-day coverage differs from the frozen "
            f"calendar window: observed={len(raw_days)}, "
            f"expected={len(expected_days)}."
        )

    return {
        "expected_hourly_rows": len(expected),
        "observed_hourly_rows": len(observed),
        "raw_delivery_days": len(raw_days),
        "first_timestamp_utc": str(observed[0]),
        "last_timestamp_utc": str(observed[-1]),
    }


def build_day_forecast_frame(
    day_local_date,
    day_hourly_df: pd.DataFrame,
    full_model_info: dict,
    tier1_model_info: dict,
    full_residual_history: pd.DataFrame,
    tier1_residual_history: pd.DataFrame,
) -> tuple:
    day_df = day_hourly_df.sort_values(
        "timestamp_utc"
    ).reset_index(drop=True)

    full_point = predict_xgboost(
        full_model_info["model"],
        day_df,
        full_model_info["predictor_cols"],
    ).to_numpy()
    tier1_point = predict_xgboost(
        tier1_model_info["model"],
        day_df,
        tier1_model_info["predictor_cols"],
    ).to_numpy()

    full_offsets = residual_quantile_offsets_for_delivery_day(
        full_residual_history,
        day_local_date,
        QUANTILES,
        WINDOW_DAYS,
        MIN_PERIODS_DAYS,
    )
    tier1_offsets = residual_quantile_offsets_for_delivery_day(
        tier1_residual_history,
        day_local_date,
        QUANTILES,
        WINDOW_DAYS,
        MIN_PERIODS_DAYS,
    )
    for label, offsets in (
        ("full", full_offsets),
        ("tier1", tier1_offsets),
    ):
        if any(pd.isna(v) for v in offsets.values()):
            raise ValueError(
                f"NaN uncertainty offset for {label} on {day_local_date}."
            )

    # Actual target is carried into the audit table only after forecasts
    # and the pre-D uncertainty snapshots are fixed.
    actual = day_df[TARGET_COL].to_numpy()

    out = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(
                day_df["timestamp_utc"], utc=True
            ),
            "delivery_date": day_local_date,
            TARGET_COL: actual,
            "lag_24_pred": day_df["lag_24_pred"].to_numpy(),
            "lag_168_pred": day_df["lag_168_pred"].to_numpy(),
            "xgboost_full_pred": full_point,
            "xgboost_tier1_pred": tier1_point,
            "full_q10_offset": full_offsets["q10"],
            "full_q50_offset": full_offsets["q50"],
            "full_q90_offset": full_offsets["q90"],
            "tier1_q10_offset": tier1_offsets["q10"],
            "tier1_q50_offset": tier1_offsets["q50"],
            "tier1_q90_offset": tier1_offsets["q90"],
        }
    )
    out["full_L"] = out["xgboost_full_pred"] + out["full_q10_offset"]
    out["full_M"] = out["xgboost_full_pred"] + out["full_q50_offset"]
    out["full_U"] = out["xgboost_full_pred"] + out["full_q90_offset"]
    out["tier1_L"] = (
        out["xgboost_tier1_pred"] + out["tier1_q10_offset"]
    )
    out["tier1_M"] = (
        out["xgboost_tier1_pred"] + out["tier1_q50_offset"]
    )
    out["tier1_U"] = (
        out["xgboost_tier1_pred"] + out["tier1_q90_offset"]
    )

    full_today = pd.DataFrame(
        {
            "timestamp_utc": out["timestamp_utc"],
            "prediction": out["xgboost_full_pred"],
            "residual": out[TARGET_COL] - out["xgboost_full_pred"],
        }
    )
    tier1_today = pd.DataFrame(
        {
            "timestamp_utc": out["timestamp_utc"],
            "prediction": out["xgboost_tier1_pred"],
            "residual": out[TARGET_COL] - out["xgboost_tier1_pred"],
        }
    )
    return out, full_today, tier1_today


def build_sequential_holdout_forecasts(
    holdout_df: pd.DataFrame,
    full_model_info: dict,
    tier1_model_info: dict,
    full_residual_history: pd.DataFrame,
    tier1_residual_history: pd.DataFrame,
) -> pd.DataFrame:
    working = baseline_predictions(holdout_df.copy())
    working = add_local_time_columns(working)
    working = working.sort_values("timestamp_utc").reset_index(drop=True)

    frames = []
    for day_local_date in sorted(working["delivery_date"].unique()):
        day_df = working[
            working["delivery_date"] == day_local_date
        ].copy()

        day_frame, full_today, tier1_today = build_day_forecast_frame(
            day_local_date,
            day_df,
            full_model_info,
            tier1_model_info,
            full_residual_history,
            tier1_residual_history,
        )
        frames.append(day_frame)

        full_residual_history = pd.concat(
            [full_residual_history, full_today],
            ignore_index=True,
        )
        tier1_residual_history = pd.concat(
            [tier1_residual_history, tier1_today],
            ignore_index=True,
        )

    return pd.concat(frames, ignore_index=True)


def run_primary_economics_on_common_days(
    hourly_inputs: pd.DataFrame,
    common_days: list,
) -> tuple:
    per_day_results = []
    for day_local_date in common_days:
        day_df = hourly_inputs[
            hourly_inputs["delivery_date"] == day_local_date
        ].copy()
        result = run_backtest_for_day(
            day_df, ETA_RT_PRIMARY, C_PRIMARY
        )
        result["delivery_date"] = day_local_date
        result["n_hours"] = len(day_df)
        per_day_results.append(result)

    verify_structural_invariants(per_day_results)

    rows = []
    for result in per_day_results:
        row = {
            "delivery_date": result["delivery_date"],
            "n_hours": result["n_hours"],
            "oracle_pnl": result["oracle"],
        }
        for strat in STRATEGIES:
            r = result[strat]
            row[f"{strat}_traded"] = r["traded"]
            row[f"{strat}_i"] = r["i"]
            row[f"{strat}_j"] = r["j"]
            row[f"{strat}_net_pnl"] = r["net_pnl"]
            row[f"{strat}_gross_pnl"] = r["gross_pnl"]
        rows.append(row)

    return per_day_results, pd.DataFrame(rows)


def _coverage_to_jsonable(coverage: dict) -> dict:
    out = {}
    for key, value in coverage.items():
        if isinstance(value, np.integer):
            out[key] = int(value)
        elif isinstance(value, np.floating):
            out[key] = float(value)
        else:
            out[key] = value
    return out


def main():
    cfg = load_config()
    logger = setup_logging(cfg["logging"]["level"])
    output_run_version = resolve_run_args(sys.argv[1:])
    assert_frozen_economic_contract()

    out_dir = (
        REPO_ROOT
        / "outputs"
        / "holdout"
        / INPUT_STEM
        / output_run_version
    )
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"{out_dir} already contains results. Use a new output run "
            "version; holdout evidence is immutable."
        )

    in_path = REPO_ROOT / cfg["data"]["processed_dir"] / INPUT_FILENAME
    if not in_path.exists():
        raise FileNotFoundError(f"Missing {in_path}.")

    logger.info("Verifying frozen development/uncertainty lineage...")
    lineage = verify_development_lineage(in_path)

    logger.info("Verifying exact frozen final model artifacts...")
    final_fit = load_final_models()
    full_model_info = final_fit["models"]["full"]
    tier1_model_info = final_fit["models"]["tier1"]

    logger.info("Loading corrected development OOS residual histories...")
    full_history = build_initial_residual_history(FULL_PRED_COL)
    tier1_history = build_initial_residual_history(TIER1_PRED_COL)

    logger.info("Loading processed feature parquet...")
    df = pd.read_parquet(in_path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    window = get_holdout_window(
        holdout_start=HOLDOUT_START_LOCAL,
        holdout_end=pd.Timestamp(
            local_delivery_date_to_utc(HOLDOUT_END_LOCAL_EXCLUSIVE)
        ),
    )
    _, holdout_df = slice_window(df, window)

    structural = assert_exact_holdout_coverage(holdout_df)
    logger.info(
        "Frozen holdout coverage verified: %d hourly rows, %d delivery days.",
        structural["observed_hourly_rows"],
        structural["raw_delivery_days"],
    )

    hourly_inputs = build_sequential_holdout_forecasts(
        holdout_df,
        full_model_info,
        tier1_model_info,
        full_history,
        tier1_history,
    )
    if len(hourly_inputs) != len(holdout_df):
        raise AssertionError(
            "Sequential forecast row count differs from raw holdout rows."
        )
    if hourly_inputs["timestamp_utc"].duplicated().any():
        raise AssertionError(
            "Sequential forecast table contains duplicate timestamps."
        )

    common = build_common_evaluation_days(hourly_inputs.copy())
    common_days = list(common["common_days"])
    common_set = set(common_days)
    hourly_inputs["common_evaluation_day"] = hourly_inputs[
        "delivery_date"
    ].isin(common_set)

    _, results_df = run_primary_economics_on_common_days(
        hourly_inputs, common_days
    )
    if len(results_df) != len(common_days):
        raise AssertionError(
            "Per-day result count differs from common-day count."
        )

    # Only after all gates have passed are 2026 artifacts saved.
    out_dir.mkdir(parents=True, exist_ok=True)
    hourly_inputs.to_csv(
        out_dir / "hourly_holdout_inputs.csv", index=False
    )
    results_df.to_csv(
        out_dir / "per_day_results.csv", index=False
    )

    common_coverage = _coverage_to_jsonable(common["coverage"])
    with open(
        out_dir / "common_day_coverage.json", "w", encoding="utf-8"
    ) as f:
        json.dump(common_coverage, f, indent=2, default=str)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_filename": INPUT_FILENAME,
        "input_sha256": lineage["current_input_sha256"],
        "xgboost_run_version": XGBOOST_RUN_VERSION,
        "uncertainty_selected_run_version": UNCERTAINTY_SELECTED_RUN_VERSION,
        "tier1_uncertainty_run_version": TIER1_UNCERTAINTY_RUN_VERSION,
        "final_model_fit_run_version": FINAL_MODEL_FIT_RUN_VERSION,
        "output_run_version": output_run_version,
        "holdout_start_local": HOLDOUT_START_LOCAL,
        "holdout_end_local_exclusive": HOLDOUT_END_LOCAL_EXCLUSIVE,
        "raw_holdout_hourly_rows": structural["observed_hourly_rows"],
        "raw_holdout_delivery_days": structural["raw_delivery_days"],
        "n_common_holdout_days": len(common_days),
        "common_day_coverage": common_coverage,
        "window_days": WINDOW_DAYS,
        "min_periods_days": MIN_PERIODS_DAYS,
        "quantiles": QUANTILES,
        "information_set_method": INFORMATION_SET_METHOD,
        "eta_rt_primary": ETA_RT_PRIMARY,
        "c_primary": C_PRIMARY,
        "strategies": list(STRATEGIES),
        "structural_invariants": "PASSED",
        "exact_raw_holdout_coverage": "PASSED",
        "common_mask_method": (
            "run_strategy_backtest.build_common_evaluation_days"
        ),
        "sequential_residual_update_scope": (
            "all raw holdout delivery days; common mask applied afterward"
        ),
        "protocol": "docs/holdout_protocol_v2.md",
        "protocol_tag": PROTOCOL_TAG,
        "protocol_commit": PROTOCOL_COMMIT,
        "development_evidence_tag": DEVELOPMENT_TAG,
        "development_evidence_commit": DEVELOPMENT_COMMIT,
        "final_model_fit_implementation_tag": FINAL_MODEL_FIT_IMPLEMENTATION_TAG,
        "final_model_fit_implementation_commit": FINAL_MODEL_FIT_IMPLEMENTATION_COMMIT,
        "final_model_fit_tag": FINAL_MODEL_FIT_TAG,
        "final_model_fit_commit": FINAL_MODEL_FIT_COMMIT,
        "final_model_fit_sha256": EXPECTED_FINAL_FIT_SHA256,
        "holdout_used": True,
        "classification_complete": False,
        "classification_note": (
            "The preregistered confirmation/partial/failure classification "
            "requires the frozen 3x3 sensitivity grid from the saved hourly "
            "holdout artifact."
        ),
    }
    with open(
        out_dir / "holdout_manifest.json", "w", encoding="utf-8"
    ) as f:
        json.dump(manifest, f, indent=2, default=str)

    # First user-visible economic output is after every gate and save.
    print("\n" + "=" * 78)
    print(f"2026 HOLDOUT EXPOSURE: {output_run_version}")
    print("=" * 78)
    print(
        f"Holdout window: {HOLDOUT_START_LOCAL} -> "
        f"{HOLDOUT_END_LOCAL_EXCLUSIVE} (exclusive)"
    )
    print(f"Raw delivery days: {structural['raw_delivery_days']}")
    print(f"Common evaluation days: {len(common_days)}")
    print("Structural invariants: PASSED")

    summary_rows = []
    for strat in STRATEGIES:
        summary_rows.append(
            {
                "strategy": strat,
                "total_net_pnl": float(
                    results_df[f"{strat}_net_pnl"].sum()
                ),
                "trading_days": int(
                    results_df[f"{strat}_traded"].sum()
                ),
            }
        )
    print("\nPRIMARY CELL ONLY (eta_rt=0.85, c=10):")
    print(pd.DataFrame(summary_rows).to_string(index=False))

    delta_forecast = float(
        results_df["S2_net_pnl"].sum()
        - results_df["S1_net_pnl"].sum()
    )
    print(
        "\nDelta_forecast (S2-S1, primary cell only): "
        f"EUR {delta_forecast:.2f}"
    )
    print(
        "NO confirmation/partial/failure verdict is assigned here; "
        "the preregistered verdict requires the frozen 9-cell sensitivity."
    )
    print(f"\nSaved frozen holdout artifacts to {out_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
