"""Run locally: python run_strategy_backtest.py <xgboost_run_version> <uncertainty_run_version> <tier1_uncertainty_run_version> <output_run_version>

Example:
  python run_strategy_backtest.py xgboost_v1_a03fix uncertainty_selected_v1 uncertainty_tier1_robustness_v2 strategy_backtest_v1

Orchestrates the frozen S0-S5 strategy set (docs/economic_contract_v1.md)
at the PRIMARY parameters (eta_rt=0.85, c=EUR10) across the 2023-2025
development period: loads the required frozen artifacts (never
recomputing what's already saved, per the contract's provenance rule),
builds common_strategy_evaluation_days with a full exclusion report,
runs every strategy plus the ex-post oracle for every common day,
verifies structural invariants against the ACTUAL results (not just
the synthetic unit tests) before any interpretation, and saves both
per-day raw results and the aggregate summary report.

Does NOT run the 3x3 sensitivity grid (Step G in the contract) --
that's a deliberately separate, later run once this primary result is
structurally validated.

STANDING CAVEAT: never touches 2026 -- assert_no_holdout_access() is
called before any per-day processing.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from src.clean import add_local_time_columns, local_delivery_date_to_utc
from src.models import TARGET_COL
from src.oracle import oracle_pnl
from src.strategy import assert_no_holdout_access, degradation_cost, run_day
from src.utils import load_config, setup_logging, REPO_ROOT

ETA_RT_PRIMARY = 0.85
C_PRIMARY = 10.0

STRATEGIES = ["S0", "S1", "S2", "S3", "S4", "S5"]


def resolve_run_args(args: list) -> tuple:
    """5th arg (legacy_run_version) is OPTIONAL: when given, this run
    is treated as a schema-extension of an existing legacy run --
    every field the legacy run saved must be reproduced EXACTLY before
    this run's results are trusted or saved (see
    verify_legacy_schema_extension_reproduction). Omit it for a
    genuinely first-ever run, or a deliberately new experiment that
    isn't extending a previous one.
    """
    if len(args) not in (4, 5):
        raise SystemExit(
            "Usage:\n"
            "  python run_strategy_backtest.py <xgboost_run_version> <uncertainty_run_version> "
            "<tier1_uncertainty_run_version> <output_run_version> [legacy_run_version]\n\n"
            "Example (first-ever run):\n"
            "  python run_strategy_backtest.py xgboost_v1_a03fix uncertainty_selected_v1 "
            "uncertainty_tier1_robustness_v2 strategy_backtest_v1\n\n"
            "Example (schema-extension of an existing run -- verified reproduction required):\n"
            "  python run_strategy_backtest.py xgboost_v1_a03fix uncertainty_selected_v1 "
            "uncertainty_tier1_robustness_v2 strategy_backtest_v2 strategy_backtest_v1"
        )
    legacy_run_version = args[4] if len(args) == 5 else None
    return args[0], args[1], args[2], args[3], legacy_run_version


def load_combined_data(input_stem: str, xgboost_run_version: str, uncertainty_run_version: str, tier1_run_version: str) -> pd.DataFrame:
    """Loads and merges every frozen artifact needed for all six
    strategies. Reads ONLY already-saved files -- no recomputation of
    predictions or uncertainty bounds, per the contract's provenance
    rule.
    """
    xgboost_dir = REPO_ROOT / "outputs" / "models" / input_stem / xgboost_run_version
    fold_names = ["fold_1", "fold_2", "fold_3", "regime_stress_test"]
    frames = []
    for fold_name in fold_names:
        path = xgboost_dir / f"{fold_name}_predictions.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run run_xgboost.py for '{xgboost_run_version}' first.")
        frames.append(pd.read_csv(path))
    predictions = pd.concat(frames, ignore_index=True)
    predictions["timestamp_utc"] = pd.to_datetime(predictions["timestamp_utc"], utc=True)
    required_pred_cols = ["timestamp_utc", TARGET_COL, "lag_24_pred", "xgboost_full_pred", "xgboost_tier1_pred"]
    missing = [c for c in required_pred_cols if c not in predictions.columns]
    if missing:
        raise ValueError(f"Fold predictions missing required column(s): {missing}")
    predictions = predictions[required_pred_cols].drop_duplicates(subset=["timestamp_utc"])

    full_bounds_path = REPO_ROOT / "outputs" / "uncertainty" / input_stem / uncertainty_run_version / "quantile_forecasts.csv"
    if not full_bounds_path.exists():
        raise FileNotFoundError(f"Missing {full_bounds_path}. Run run_uncertainty.py for '{uncertainty_run_version}' first.")
    full_bounds = pd.read_csv(full_bounds_path)
    full_bounds["timestamp_utc"] = pd.to_datetime(full_bounds["timestamp_utc"], utc=True)
    full_bounds = full_bounds[["timestamp_utc", "forecast_q10", "forecast_q90"]].rename(
        columns={"forecast_q10": "full_L", "forecast_q90": "full_U"}
    )

    tier1_bounds_path = REPO_ROOT / "outputs" / "uncertainty" / input_stem / tier1_run_version / "quantile_forecasts_tier1.csv"
    if not tier1_bounds_path.exists():
        raise FileNotFoundError(
            f"Missing {tier1_bounds_path}. Run run_uncertainty_tier1_robustness.py for "
            f"'{tier1_run_version}' first (must be a version built after the per-row-bounds fix)."
        )
    tier1_bounds = pd.read_csv(tier1_bounds_path)
    tier1_bounds["timestamp_utc"] = pd.to_datetime(tier1_bounds["timestamp_utc"], utc=True)
    tier1_bounds = tier1_bounds[["timestamp_utc", "forecast_q10", "forecast_q90"]].rename(
        columns={"forecast_q10": "tier1_L", "forecast_q90": "tier1_U"}
    )

    combined = predictions.merge(full_bounds, on="timestamp_utc", how="left").merge(
        tier1_bounds, on="timestamp_utc", how="left"
    )
    return combined


def build_common_evaluation_days(df: pd.DataFrame) -> dict:
    """Returns a dict with the exclusion/coverage report AND the final
    common_strategy_evaluation_days set. A day is included only if
    EVERY hour of that day has every column required by EVERY strategy
    -- a day-level decision needs a complete intraday vector; a day
    with even one missing hour has an undefined candidate-pair search.
    """
    df = add_local_time_columns(df)
    raw_days = sorted(df["delivery_date"].unique())

    required_cols = {
        "lag24": ["lag_24_pred"],
        "full": ["xgboost_full_pred"],
        "tier1": ["xgboost_tier1_pred"],
        "uncertainty_full": ["full_L", "full_U"],
        "uncertainty_tier1": ["tier1_L", "tier1_U"],
    }

    per_day_complete = {}
    for day, group in df.groupby("delivery_date"):
        per_day_complete[day] = {
            label: bool(group[cols].notna().all().all())
            for label, cols in required_cols.items()
        }

    coverage = {
        "raw_delivery_days": len(raw_days),
        "days_lag24_available": sum(v["lag24"] for v in per_day_complete.values()),
        "days_full_available": sum(v["full"] for v in per_day_complete.values()),
        "days_tier1_available": sum(v["tier1"] for v in per_day_complete.values()),
        "days_uncertainty_full_available": sum(v["uncertainty_full"] for v in per_day_complete.values()),
        "days_uncertainty_tier1_available": sum(v["uncertainty_tier1"] for v in per_day_complete.values()),
    }
    common_days = sorted(
        day for day, flags in per_day_complete.items() if all(flags.values())
    )
    coverage["days_excluded_for_incomplete_intraday_vector"] = len(raw_days) - len(common_days)
    coverage["common_evaluation_days"] = len(common_days)

    return {"coverage": coverage, "common_days": common_days, "df_with_local_time": df}


def run_backtest_for_day(day_df: pd.DataFrame, eta_rt: float, c: float) -> dict:
    """Runs all six strategies plus the oracle for one delivery day's
    already-complete data (sorted chronologically). Returns a dict of
    per-strategy results plus the oracle.
    """
    day_df = day_df.sort_values("timestamp_utc").reset_index(drop=True)
    actual = day_df[TARGET_COL].to_numpy()
    C = degradation_cost(eta_rt, c)

    results = {}
    results["S0"] = {"i": None, "j": None, "traded": False, "gross_pnl": 0.0, "net_pnl": 0.0}
    results["S1"] = run_day(day_df["lag_24_pred"].to_numpy(), actual, eta_rt, c)
    results["S2"] = run_day(day_df["xgboost_full_pred"].to_numpy(), actual, eta_rt, c)
    results["S3"] = run_day(
        day_df["xgboost_full_pred"].to_numpy(), actual, eta_rt, c,
        L=day_df["full_L"].to_numpy(), U=day_df["full_U"].to_numpy(),
    )
    results["S4"] = run_day(day_df["xgboost_tier1_pred"].to_numpy(), actual, eta_rt, c)
    results["S5"] = run_day(
        day_df["xgboost_tier1_pred"].to_numpy(), actual, eta_rt, c,
        L=day_df["tier1_L"].to_numpy(), U=day_df["tier1_U"].to_numpy(),
    )
    results["oracle"] = oracle_pnl(actual, eta_rt, c)
    return results


def verify_structural_invariants(per_day_results: list) -> None:
    """Checks the required invariants against the ACTUAL backtest
    results, not just the synthetic unit tests -- if any of these
    fail, STOP before any economic interpretation, per the contract's
    Step F.
    """
    violations = []
    for day_result in per_day_results:
        oracle_val = day_result["oracle"]
        for strat in STRATEGIES:
            r = day_result[strat]
            if r["traded"]:
                if not (r["i"] < r["j"]):
                    violations.append(f"{day_result['delivery_date']} {strat}: i={r['i']} >= j={r['j']}")
            if r["net_pnl"] > oracle_val + 1e-6:
                violations.append(
                    f"{day_result['delivery_date']} {strat}: net_pnl={r['net_pnl']:.4f} "
                    f"exceeds oracle={oracle_val:.4f}"
                )
    if violations:
        raise AssertionError(
            f"{len(violations)} structural invariant violation(s) found -- STOPPING before "
            f"any economic interpretation, per docs/economic_contract_v1.md. First few:\n" +
            "\n".join(violations[:10])
        )


def verify_legacy_schema_extension_reproduction(results_df: pd.DataFrame, legacy_run_version: str, input_stem: str) -> dict:
    """When this run is a schema-extension of an existing legacy run
    (e.g. adding persisted (i, j) columns without touching any
    decision/P&L logic), every field the legacy run saved must be
    reproduced EXACTLY -- not assumed unchanged just because the code
    diff looks purely additive. This project has repeatedly found
    "obviously safe" changes that weren't (the A03 parser, the
    ingestion log append bug); a cheap, mechanical check here costs
    little and catches the same class of mistake before it can
    silently become the new reproduction anchor for the sensitivity
    grid.

    Compares ONLY the fields that existed in the legacy run (i/j are
    new-only columns and are correctly never compared, since the
    legacy run never had them) -- n_hours, oracle_pnl, and every
    strategy's traded/net_pnl/gross_pnl, plus the four aggregate
    deltas recomputed from each. Returns a dict summarizing the
    comparison on success; raises AssertionError on any mismatch.
    """
    legacy_path = REPO_ROOT / "outputs" / "strategy" / input_stem / legacy_run_version / "per_day_results.csv"
    if not legacy_path.exists():
        raise FileNotFoundError(f"No legacy results at {legacy_path}.")
    legacy = pd.read_csv(legacy_path)
    legacy["delivery_date"] = legacy["delivery_date"].astype(str)
    new = results_df.copy()
    new["delivery_date"] = new["delivery_date"].astype(str)

    merged = legacy.merge(new, on="delivery_date", how="outer", suffixes=("_legacy", "_new"), indicator=True)
    only_one_side = merged[merged["_merge"] != "both"]
    if len(only_one_side) > 0:
        raise AssertionError(
            f"LEGACY SCHEMA-EXTENSION REPRODUCTION FAILED: {len(only_one_side)} day(s) present "
            f"in only one of the legacy vs. new results -- the common-day sets don't even match."
        )

    mismatches = []
    n_hours_diff = (merged["n_hours_legacy"] - merged["n_hours_new"]).abs()
    if (n_hours_diff > 0).any():
        mismatches.append(f"n_hours: {(n_hours_diff > 0).sum()} day(s) differ")
    oracle_diff = (merged["oracle_pnl_legacy"] - merged["oracle_pnl_new"]).abs()
    if (oracle_diff > 1e-9).any():
        mismatches.append(f"oracle_pnl: {(oracle_diff > 1e-9).sum()} day(s) differ")

    for strat in STRATEGIES:
        traded_col_l, traded_col_n = f"{strat}_traded_legacy", f"{strat}_traded_new"
        if (merged[traded_col_l] != merged[traded_col_n]).any():
            mismatches.append(f"{strat}_traded: {(merged[traded_col_l] != merged[traded_col_n]).sum()} day(s) differ")
        for col_suffix in ("net_pnl", "gross_pnl"):
            l_col, n_col = f"{strat}_{col_suffix}_legacy", f"{strat}_{col_suffix}_new"
            diff = (merged[l_col] - merged[n_col]).abs()
            if (diff > 1e-9).any():
                mismatches.append(f"{strat}_{col_suffix}: {(diff > 1e-9).sum()} day(s) differ")

    def _deltas(df, suffix):
        return {
            "delta_forecast": float(df[f"S2_net_pnl{suffix}"].sum() - df[f"S1_net_pnl{suffix}"].sum()),
            "delta_uncertainty": float(df[f"S3_net_pnl{suffix}"].sum() - df[f"S2_net_pnl{suffix}"].sum()),
            "delta_tier1": float(df[f"S4_net_pnl{suffix}"].sum() - df[f"S2_net_pnl{suffix}"].sum()),
            "delta_tier1_u": float(df[f"S5_net_pnl{suffix}"].sum() - df[f"S3_net_pnl{suffix}"].sum()),
        }
    legacy_deltas = _deltas(merged, "_legacy")
    new_deltas = _deltas(merged, "_new")
    for key in legacy_deltas:
        if abs(legacy_deltas[key] - new_deltas[key]) > 1e-6:
            mismatches.append(f"{key}: legacy={legacy_deltas[key]:.6f}, new={new_deltas[key]:.6f}")

    if mismatches:
        raise AssertionError(
            f"LEGACY SCHEMA-EXTENSION REPRODUCTION FAILED against '{legacy_run_version}' -- "
            f"this run cannot be treated as a schema-extension of the legacy result. "
            f"STOPPING before saving anything.\n" + "\n".join(mismatches)
        )

    return {"parent_run": legacy_run_version, "legacy_deltas": legacy_deltas, "new_deltas": new_deltas}


def main():
    cfg = load_config()
    logger = setup_logging(cfg["logging"]["level"])

    xgboost_run_version, uncertainty_run_version, tier1_run_version, output_run_version, legacy_run_version = resolve_run_args(sys.argv[1:])
    input_stem = "delu_features"

    out_dir = REPO_ROOT / "outputs" / "strategy" / input_stem / output_run_version
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} already contains results. Pass a new output_run_version.")

    logger.info("Loading frozen artifacts (no recomputation)...")
    combined = load_combined_data(input_stem, xgboost_run_version, uncertainty_run_version, tier1_run_version)

    assert_no_holdout_access(combined["timestamp_utc"])
    logger.info("Holdout check passed -- no timestamps at or after 2026-01-01.")

    logger.info("Building common_strategy_evaluation_days...")
    day_info = build_common_evaluation_days(combined)
    coverage = day_info["coverage"]
    common_days = day_info["common_days"]
    df = day_info["df_with_local_time"]

    print("\n" + "=" * 78)
    print(f"COVERAGE / EXCLUSION REPORT: {output_run_version}")
    print("=" * 78)
    for k, v in coverage.items():
        print(f"  {k}: {v}")
    if coverage["common_evaluation_days"] == 0:
        raise RuntimeError("Zero common evaluation days -- stopping, nothing to backtest.")

    logger.info("Running S0-S5 + oracle for %d common days at PRIMARY parameters "
                "(eta_rt=%.2f, c=EUR%.0f)...", len(common_days), ETA_RT_PRIMARY, C_PRIMARY)
    per_day_results = []
    for day in common_days:
        day_df = df[df["delivery_date"] == day]
        result = run_backtest_for_day(day_df, ETA_RT_PRIMARY, C_PRIMARY)
        result["delivery_date"] = day
        result["n_hours"] = len(day_df)
        per_day_results.append(result)

    print("\nVerifying structural invariants against actual results (i<j, oracle dominance)...")
    verify_structural_invariants(per_day_results)
    print("All structural invariants PASSED.")

    rows = []
    for r in per_day_results:
        row = {"delivery_date": r["delivery_date"], "n_hours": r["n_hours"], "oracle_pnl": r["oracle"]}
        for strat in STRATEGIES:
            row[f"{strat}_traded"] = r[strat]["traded"]
            row[f"{strat}_i"] = r[strat]["i"]
            row[f"{strat}_j"] = r[strat]["j"]
            row[f"{strat}_net_pnl"] = r[strat]["net_pnl"]
            row[f"{strat}_gross_pnl"] = r[strat]["gross_pnl"]
        rows.append(row)
    results_df = pd.DataFrame(rows)

    # SCHEMA-EXTENSION GATE, run BEFORE anything about this run's own
    # economics is computed or printed -- not just before saving. If a
    # supposedly schema-only change actually altered a P&L value, that
    # altered figure must never be visible even transiently, since
    # seeing it first (even followed immediately by a FAILED message)
    # already means an invalid result was observed. Order matters here,
    # not just the final outcome.
    schema_extension_info = None
    if legacy_run_version is not None:
        logger.info("Verifying this run reproduces legacy run '%s' on every previously-saved "
                    "field BEFORE computing or displaying this run's own economic summary...",
                    legacy_run_version)
        verify_legacy_schema_extension_reproduction(results_df, legacy_run_version, input_stem)
        print(f"\nLEGACY SCHEMA-EXTENSION REPRODUCTION CHECK PASSED against '{legacy_run_version}' -- "
              f"every field that run saved (n_hours, oracle_pnl, S0-S5 traded/net/gross P&L, all "
              f"four aggregate deltas) is reproduced exactly. This run is a schema-extension, not "
              f"a new economic experiment. Proceeding to display this run's results.")
        new_fields = [f"{s}_{suffix}" for s in STRATEGIES for suffix in ("i", "j")]
        schema_extension_info = {
            "parent_run": legacy_run_version,
            "change_type": "schema_extension_only",
            "legacy_result_reproduction": "PASSED",
            "new_fields": new_fields,
        }

    print("\n" + "=" * 78)
    print("SUMMARY (primary parameters only -- structural validation, not full reporting)")
    print("=" * 78)
    summary_rows = []
    for strat in STRATEGIES:
        net = results_df[f"{strat}_net_pnl"]
        traded = results_df[f"{strat}_traded"]
        summary_rows.append({
            "strategy": strat,
            "total_net_pnl": net.sum(),
            "trading_days": int(traded.sum()),
            "profitable_days": int((net > 0).sum()),
        })
    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))

    delta_forecast = results_df["S2_net_pnl"].sum() - results_df["S1_net_pnl"].sum()
    delta_uncertainty = results_df["S3_net_pnl"].sum() - results_df["S2_net_pnl"].sum()
    delta_tier1 = results_df["S4_net_pnl"].sum() - results_df["S2_net_pnl"].sum()
    delta_tier1_u = results_df["S5_net_pnl"].sum() - results_df["S3_net_pnl"].sum()
    print(f"\nDelta_forecast (S2-S1, PRIMARY result): EUR {delta_forecast:.2f}")
    print(f"Delta_uncertainty (S3-S2): EUR {delta_uncertainty:.2f}")
    print(f"Delta_Tier1 (S4-S2): EUR {delta_tier1:.2f}")
    print(f"Delta_Tier1,U (S5-S3): EUR {delta_tier1_u:.2f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_dir / "per_day_results.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "xgboost_run_version": xgboost_run_version,
        "uncertainty_run_version": uncertainty_run_version,
        "tier1_uncertainty_run_version": tier1_run_version,
        "output_run_version": output_run_version,
        "eta_rt": ETA_RT_PRIMARY, "c": C_PRIMARY,
        "coverage": coverage,
        "structural_invariants": "PASSED",
        "delta_forecast": float(delta_forecast),
        "delta_uncertainty": float(delta_uncertainty),
        "delta_tier1": float(delta_tier1),
        "delta_tier1_u": float(delta_tier1_u),
        "holdout_used": False,
    }
    if schema_extension_info is not None:
        manifest.update(schema_extension_info)
    with open(out_dir / "strategy_backtest_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\nSaved per-day results + summary + manifest to {out_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
