"""Run locally: python run_final_holdout_sensitivity.py <output_run_version>

Example:
  python run_final_holdout_sensitivity.py holdout_sensitivity_v1

FROZEN companion to run_final_holdout.py.

The source holdout is fixed to holdout_v1. This script never regenerates
2026 forecasts, uncertainty bounds, or residual histories. It reads the
already-saved hourly_holdout_inputs.csv from holdout_v1, filters to the
frozen common_evaluation_day mask saved by run_final_holdout.py, and
re-scores those SAME common days over the pre-registered 3x3 eta_rt/c grid.

Before the other eight cells are trusted, the (0.85, 10) cell recomputed
here must reproduce holdout_v1/per_day_results.csv:
  - identical delivery-day set and n_hours;
  - identical trade/no-trade decisions and (i, j) selections;
  - gross/net P&L and oracle equal within a tiny numeric tolerance needed
    only for CSV serialization round-tripping.

The mechanical Section 8 verdict is then:
  CONFIRMATION         primary delta_forecast > 0 AND all 9/9 cells > 0
  PARTIAL CONFIRMATION primary delta_forecast > 0 but fewer than 9/9 > 0
  FAILURE              primary delta_forecast <= 0

No alternative holdout source, grid, strategy set, or primary economic
parameters can be supplied by CLI.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from src.clean import add_local_time_columns
from src.utils import REPO_ROOT, load_config, setup_logging
from run_strategy_backtest import (
    STRATEGIES,
    run_backtest_for_day,
    verify_structural_invariants,
)
from run_strategy_sensitivity import ETA_RT_GRID, C_GRID

INPUT_STEM = "delu_features"
HOLDOUT_RUN_VERSION = "holdout_v1"

PRIMARY_ETA_RT = 0.85
PRIMARY_C = 10.0

PROTOCOL_TAG = "pre-holdout-protocol-v2"
PROTOCOL_COMMIT = "a2d1f9a79022ca1e2731d52803061825038c69f5"
DEVELOPMENT_TAG = "corrected-development-v2-dayorigin"
DEVELOPMENT_COMMIT = "704cc933b99bb32c21cd358e613d1cd667d71488"
FINAL_MODEL_FIT_TAG = "final-model-fit-v1"
FINAL_MODEL_FIT_COMMIT = "54a7c2e77122134eca31a09798cc1614df1c634c"

EXPECTED_COMMON_MASK_METHOD = "run_strategy_backtest.build_common_evaluation_days"
EXPECTED_STRATEGIES = ["S0", "S1", "S2", "S3", "S4", "S5"]
FLOAT_REPRO_ATOL = 1e-9


def resolve_run_args(args: list) -> str:
    if len(args) != 1:
        raise SystemExit(
            "Usage:\n"
            "  python run_final_holdout_sensitivity.py <output_run_version>\n\n"
            "Example:\n"
            "  python run_final_holdout_sensitivity.py holdout_sensitivity_v1\n\n"
            "The source holdout is frozen to holdout_v1 and is not a CLI argument."
        )
    return args[0]


def assert_frozen_grid_contract() -> None:
    if ETA_RT_GRID != [0.70, 0.85, 0.92]:
        raise AssertionError(f"Frozen ETA_RT_GRID drifted: {ETA_RT_GRID!r}.")
    if C_GRID != [5, 10, 15]:
        raise AssertionError(f"Frozen C_GRID drifted: {C_GRID!r}.")
    if list(STRATEGIES) != EXPECTED_STRATEGIES:
        raise AssertionError(f"Frozen strategy set drifted: {STRATEGIES!r}.")
    if PRIMARY_ETA_RT not in ETA_RT_GRID or PRIMARY_C not in C_GRID:
        raise AssertionError("Primary economic cell is absent from frozen grid.")


def _coerce_common_flag(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    bad = sorted(set(normalized) - set(mapping))
    if bad:
        raise ValueError(
            "common_evaluation_day contains unrecognized values: "
            f"{bad[:10]}"
        )
    return normalized.map(mapping).astype(bool)


def load_holdout_artifacts() -> dict:
    holdout_dir = (
        REPO_ROOT / "outputs" / "holdout" / INPUT_STEM / HOLDOUT_RUN_VERSION
    )
    hourly_path = holdout_dir / "hourly_holdout_inputs.csv"
    per_day_path = holdout_dir / "per_day_results.csv"
    manifest_path = holdout_dir / "holdout_manifest.json"

    for path in (hourly_path, per_day_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. The frozen holdout_v1 exposure must be completed first."
            )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_exact = {
        "output_run_version": HOLDOUT_RUN_VERSION,
        "holdout_used": True,
        "classification_complete": False,
        "eta_rt_primary": PRIMARY_ETA_RT,
        "c_primary": PRIMARY_C,
        "strategies": EXPECTED_STRATEGIES,
        "structural_invariants": "PASSED",
        "exact_raw_holdout_coverage": "PASSED",
        "common_mask_method": EXPECTED_COMMON_MASK_METHOD,
        "protocol_tag": PROTOCOL_TAG,
        "protocol_commit": PROTOCOL_COMMIT,
        "development_evidence_tag": DEVELOPMENT_TAG,
        "development_evidence_commit": DEVELOPMENT_COMMIT,
        "final_model_fit_tag": FINAL_MODEL_FIT_TAG,
        "final_model_fit_commit": FINAL_MODEL_FIT_COMMIT,
    }
    problems = []
    for field, expected in required_exact.items():
        actual = manifest.get(field)
        if actual != expected:
            problems.append(f"{field}={actual!r}, expected {expected!r}")
    if problems:
        raise ValueError(
            "HOLDOUT SOURCE LINEAGE CHECK FAILED -- refusing sensitivity scoring:\n  - "
            + "\n  - ".join(problems)
        )

    hourly = pd.read_csv(hourly_path)
    required_hourly = [
        "timestamp_utc",
        "price_eur_mwh",
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
            f"hourly_holdout_inputs.csv missing required columns: {missing}"
        )

    hourly["timestamp_utc"] = pd.to_datetime(hourly["timestamp_utc"], utc=True)
    if hourly["timestamp_utc"].duplicated().any():
        raise ValueError("hourly_holdout_inputs.csv contains duplicate timestamps.")

    hourly = add_local_time_columns(hourly)
    hourly["common_evaluation_day"] = _coerce_common_flag(
        hourly["common_evaluation_day"]
    )
    common_hourly = hourly[hourly["common_evaluation_day"]].copy()
    if common_hourly.empty:
        raise ValueError("Frozen holdout artifact contains zero common rows.")

    strategy_cols = [
        "price_eur_mwh",
        "lag_24_pred",
        "xgboost_full_pred",
        "xgboost_tier1_pred",
        "full_L",
        "full_U",
        "tier1_L",
        "tier1_U",
    ]
    if common_hourly[strategy_cols].isna().any().any():
        raise ValueError(
            "A common_evaluation_day=True row has missing strategy input(s)."
        )

    per_day = pd.read_csv(per_day_path)
    required_per_day = ["delivery_date", "n_hours", "oracle_pnl"]
    for strategy in STRATEGIES:
        required_per_day.extend(
            [
                f"{strategy}_traded",
                f"{strategy}_i",
                f"{strategy}_j",
                f"{strategy}_net_pnl",
                f"{strategy}_gross_pnl",
            ]
        )
    missing_per_day = [c for c in required_per_day if c not in per_day.columns]
    if missing_per_day:
        raise ValueError(
            f"per_day_results.csv missing required columns: {missing_per_day}"
        )

    per_day["delivery_date"] = per_day["delivery_date"].astype(str)
    common_days = sorted(str(d) for d in common_hourly["delivery_date"].unique())
    saved_days = sorted(per_day["delivery_date"].tolist())
    if common_days != saved_days:
        raise ValueError(
            "Frozen hourly common-day set differs from per_day_results.csv."
        )

    observed_hours = {
        str(day): int(len(group))
        for day, group in common_hourly.groupby("delivery_date")
    }
    for _, row in per_day.iterrows():
        day = str(row["delivery_date"])
        if observed_hours.get(day) != int(row["n_hours"]):
            raise ValueError(
                f"Common-day hour count mismatch for {day}: "
                f"hourly={observed_hours.get(day)}, saved={row['n_hours']}."
            )

    if len(saved_days) != int(manifest.get("n_common_holdout_days", -1)):
        raise ValueError(
            "per_day_results row count differs from manifest n_common_holdout_days."
        )

    return {
        "hourly_all": hourly,
        "common_hourly": common_hourly,
        "per_day_df": per_day,
        "manifest": manifest,
        "holdout_dir": holdout_dir,
    }


def compute_cell(common_hourly: pd.DataFrame, eta_rt: float, c: float) -> pd.DataFrame:
    rows = []
    for delivery_date, group in common_hourly.groupby("delivery_date", sort=True):
        day_result = run_backtest_for_day(group, eta_rt, c)
        row = {
            "delivery_date": str(delivery_date),
            "n_hours": int(len(group)),
            "oracle_pnl": float(day_result["oracle"]),
        }
        for strategy in STRATEGIES:
            result = day_result[strategy]
            row[f"{strategy}_traded"] = bool(result["traded"])
            row[f"{strategy}_i"] = result["i"]
            row[f"{strategy}_j"] = result["j"]
            row[f"{strategy}_net_pnl"] = float(result["net_pnl"])
            row[f"{strategy}_gross_pnl"] = float(result["gross_pnl"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("delivery_date").reset_index(drop=True)


def _same_nullable_exact(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a == b) | (a.isna() & b.isna())


def verify_base_case_reproduction(
    computed_base: pd.DataFrame, saved_per_day: pd.DataFrame
) -> None:
    computed = computed_base.sort_values("delivery_date").reset_index(drop=True)
    saved = saved_per_day.copy()
    saved["delivery_date"] = saved["delivery_date"].astype(str)
    saved = saved.sort_values("delivery_date").reset_index(drop=True)

    problems = []
    if computed["delivery_date"].tolist() != saved["delivery_date"].tolist():
        problems.append("delivery_date sets/order differ")
    elif not np.array_equal(
        computed["n_hours"].to_numpy(dtype=int),
        saved["n_hours"].to_numpy(dtype=int),
    ):
        problems.append("n_hours differs")
    else:
        for strategy in STRATEGIES:
            traded_col = f"{strategy}_traded"
            if not _same_nullable_exact(
                computed[traded_col], saved[traded_col]
            ).all():
                problems.append(f"{traded_col} differs")

            for suffix in ("i", "j"):
                col = f"{strategy}_{suffix}"
                if not _same_nullable_exact(computed[col], saved[col]).all():
                    problems.append(f"{col} differs")

            for suffix in ("net_pnl", "gross_pnl"):
                col = f"{strategy}_{suffix}"
                if not np.allclose(
                    computed[col].to_numpy(dtype=float),
                    saved[col].to_numpy(dtype=float),
                    rtol=0.0,
                    atol=FLOAT_REPRO_ATOL,
                    equal_nan=True,
                ):
                    problems.append(f"{col} differs")

        if not np.allclose(
            computed["oracle_pnl"].to_numpy(dtype=float),
            saved["oracle_pnl"].to_numpy(dtype=float),
            rtol=0.0,
            atol=FLOAT_REPRO_ATOL,
            equal_nan=True,
        ):
            problems.append("oracle_pnl differs")

    if problems:
        raise AssertionError(
            "BASE-CASE REPRODUCTION FAILED -- sensitivity scoring does not "
            "reproduce frozen holdout_v1:\n  - " + "\n  - ".join(problems)
        )


def cell_to_structural_results(cell: pd.DataFrame) -> list:
    results = []
    for row in cell.to_dict(orient="records"):
        day_result = {
            "delivery_date": row["delivery_date"],
            "oracle": row["oracle_pnl"],
        }
        for strategy in STRATEGIES:
            day_result[strategy] = {
                "i": row[f"{strategy}_i"],
                "j": row[f"{strategy}_j"],
                "traded": row[f"{strategy}_traded"],
                "net_pnl": row[f"{strategy}_net_pnl"],
                "gross_pnl": row[f"{strategy}_gross_pnl"],
            }
        results.append(day_result)
    return results


def summarize_cell(cell: pd.DataFrame, eta_rt: float, c: float) -> dict:
    return {
        "eta_rt": eta_rt,
        "c": c,
        "n_days": int(len(cell)),
        "S0_total_net_pnl": float(cell["S0_net_pnl"].sum()),
        "S1_total_net_pnl": float(cell["S1_net_pnl"].sum()),
        "S2_total_net_pnl": float(cell["S2_net_pnl"].sum()),
        "S3_total_net_pnl": float(cell["S3_net_pnl"].sum()),
        "S4_total_net_pnl": float(cell["S4_net_pnl"].sum()),
        "S5_total_net_pnl": float(cell["S5_net_pnl"].sum()),
        "delta_forecast": float(
            cell["S2_net_pnl"].sum() - cell["S1_net_pnl"].sum()
        ),
        "delta_uncertainty": float(
            cell["S3_net_pnl"].sum() - cell["S2_net_pnl"].sum()
        ),
        "delta_tier1": float(
            cell["S4_net_pnl"].sum() - cell["S2_net_pnl"].sum()
        ),
        "delta_tier1_u": float(
            cell["S5_net_pnl"].sum() - cell["S3_net_pnl"].sum()
        ),
    }


def classify_section_8(grid_df: pd.DataFrame) -> tuple:
    if len(grid_df) != 9:
        raise AssertionError(
            f"Frozen 3x3 grid must have exactly 9 cells, got {len(grid_df)}."
        )
    if grid_df[["eta_rt", "c"]].duplicated().any():
        raise AssertionError("Sensitivity grid contains duplicate cells.")

    primary = grid_df[
        (grid_df["eta_rt"] == PRIMARY_ETA_RT)
        & (grid_df["c"] == PRIMARY_C)
    ]
    if len(primary) != 1:
        raise AssertionError(
            "Sensitivity grid does not contain exactly one primary cell."
        )

    primary_delta = float(primary.iloc[0]["delta_forecast"])
    n_positive = int((grid_df["delta_forecast"] > 0).sum())

    if primary_delta <= 0:
        verdict = "FAILURE"
    elif n_positive == 9:
        verdict = "CONFIRMATION"
    else:
        verdict = "PARTIAL CONFIRMATION"

    return verdict, n_positive, primary_delta


def main():
    cfg = load_config()
    logger = setup_logging(cfg["logging"]["level"])

    output_run_version = resolve_run_args(sys.argv[1:])
    assert_frozen_grid_contract()

    out_dir = (
        REPO_ROOT / "outputs" / "holdout" / INPUT_STEM / output_run_version
    )
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"{out_dir} already contains results. Sensitivity evidence is immutable."
        )

    logger.info("Loading frozen holdout_v1 artifacts...")
    artifacts = load_holdout_artifacts()
    common_hourly = artifacts["common_hourly"]

    logger.info("Verifying base-cell reproduction...")
    base_cell = compute_cell(common_hourly, PRIMARY_ETA_RT, PRIMARY_C)
    verify_structural_invariants(cell_to_structural_results(base_cell))
    verify_base_case_reproduction(base_cell, artifacts["per_day_df"])

    grid_rows = []
    detailed_frames = []
    for eta_rt in ETA_RT_GRID:
        for c in C_GRID:
            if eta_rt == PRIMARY_ETA_RT and c == PRIMARY_C:
                cell = base_cell.copy()
            else:
                cell = compute_cell(common_hourly, eta_rt, c)

            verify_structural_invariants(cell_to_structural_results(cell))
            tagged = cell.copy()
            tagged.insert(0, "c", c)
            tagged.insert(0, "eta_rt", eta_rt)
            detailed_frames.append(tagged)
            grid_rows.append(summarize_cell(cell, eta_rt, c))

    grid_df = pd.DataFrame(grid_rows).sort_values(
        ["eta_rt", "c"]
    ).reset_index(drop=True)
    verdict, n_positive, primary_delta = classify_section_8(grid_df)
    detailed_df = pd.concat(detailed_frames, ignore_index=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    grid_df.to_csv(out_dir / "sensitivity_grid.csv", index=False)
    detailed_df.to_csv(
        out_dir / "per_day_sensitivity_results.csv", index=False
    )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_holdout_run_version": HOLDOUT_RUN_VERSION,
        "output_run_version": output_run_version,
        "source_holdout_used": True,
        "source_holdout_protocol_commit": PROTOCOL_COMMIT,
        "source_final_model_fit_commit": FINAL_MODEL_FIT_COMMIT,
        "eta_rt_grid": list(ETA_RT_GRID),
        "c_grid": list(C_GRID),
        "primary_eta_rt": PRIMARY_ETA_RT,
        "primary_c": PRIMARY_C,
        "base_case_reproduction": "PASSED",
        "float_reproduction_atol": FLOAT_REPRO_ATOL,
        "n_common_holdout_days": int(len(base_cell)),
        "n_grid_cells": int(len(grid_df)),
        "n_positive_delta_forecast": n_positive,
        "primary_delta_forecast": primary_delta,
        "section_8_verdict": verdict,
        "holdout_used": True,
        "classification_complete": True,
    }
    with open(out_dir / "sensitivity_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    print("\n" + "=" * 78)
    print(f"2026 HOLDOUT 3x3 SENSITIVITY: {output_run_version}")
    print(f"Source: {HOLDOUT_RUN_VERSION}")
    print("BASE-CASE REPRODUCTION: PASSED")
    print("=" * 78)
    print(
        grid_df[
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
    print(f"\ndelta_forecast positive in {n_positive}/9 cells")
    print(f"Primary delta_forecast: EUR {primary_delta:.2f}")
    print(f"SECTION 8 MECHANICAL VERDICT: {verdict}")
    print(f"\nSaved sensitivity evidence to {out_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
