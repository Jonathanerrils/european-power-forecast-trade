"""Run locally: python run_xgboost.py <input_filename> <run_version> [elasticnet_search_profile]

Evaluates all SIX models -- lag-24, lag-168, ElasticNet-full,
ElasticNet-Tier1, XGBoost-full, XGBoost-Tier1 -- on the identical
common-row set, across every chronological fold. elasticnet_search_profile
defaults to "v1", the ACCEPTED and FROZEN ElasticNet profile (see
README's "Baseline & ElasticNet Results" -- baseline_v2 was tested and
rejected by the pre-registered acceptance rule).

The canonical experiment name "xgboost_v1" is structurally locked to
elasticnet_search_profile="v1" -- same class of guard as run_models.py's
CANONICAL_RUN_PROFILES, for the same reason: a directory literally named
"xgboost_v1" must not be able to silently contain a comparison against
the REJECTED ElasticNet-v2 profile.

After evaluation, this script verifies that the re-fitted ElasticNet-full
metrics reproduce the frozen baseline_v1 numbers (documented in the
README) to a tight tolerance. baseline_v1 is supposed to be immutable
evidence; XGBoost's evaluation refits ElasticNet rather than reading
old prediction files (methodologically fine today since v1 is frozen,
fitting is deterministic, and predictor sets are unchanged), but this
check protects against future code drift silently invalidating the
whole comparison. If it fails, the run stops before printing/saving
results that would be uninterpretable.

Promotion rule (pre-registered before the first real-data run -- see
PROMOTION_RULE below and the README): XGBoost-full is promoted to the
primary point-forecast model only if its row-weighted development MAE
improves by >= 1.0% versus frozen ElasticNet-full baseline_v1.
Otherwise ElasticNet-full is retained. No XGBoost retuning after seeing
results.

Outputs are scoped by input file and run version
(outputs/models/<input_stem>/<run_version>/) and immutable once
written -- an existing populated directory raises FileExistsError
rather than being silently overwritten, same discipline as
run_models.py.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import sklearn
import xgboost

from src.xgboost_model import evaluate_all_folds_with_xgboost, XGBOOST_PARAM_GRID
from src.models import TARGET_COL, ELASTICNET_SEARCH_PROFILES
from src.utils import load_config, setup_logging, REPO_ROOT


# --- baseline_v1's frozen, documented ElasticNet-full MAE per fold ---
# (see README "Baseline & ElasticNet Results" -- baseline_v1 table).
# Since v1 is supposed to be immutable evidence and this script refits
# ElasticNet rather than reading old prediction files, this is a
# runtime guard against future code drift silently invalidating the
# comparison, not a substitute for actually preserving baseline_v1/.
FROZEN_BASELINE_V1_ELASTICNET_FULL_MAE = {
    "fold_1": 19.510690,
    "fold_2": 20.895399,
    "fold_3": 18.813946,
    "regime_stress_test": 20.028231,
}
REPRODUCTION_TOLERANCE_RELATIVE = 0.001  # 0.1% -- tightened from an initial 0.3%, since the promotion threshold
                                          # itself is only 1.0%; a loose comparator tolerance would eat into that
                                          # decision margin. Input SHA-256 and dependency versions are already
                                          # recorded in the manifest, so float noise beyond this is worth investigating.

# Promotion rule, fixed before seeing any xgboost_v1 results.
PROMOTION_THRESHOLD_RELATIVE = 0.01  # XGBoost-full must improve row-weighted MAE by >= 1.0% vs ElasticNet-full v1


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# Canonical experiment names -> the ONLY elasticnet_search_profile
# allowed for them -- same class of guard as run_models.py's
# CANONICAL_RUN_PROFILES, structural not just conventional.
CANONICAL_XGBOOST_RUN_PROFILES = {"xgboost_v1": "v1"}

# For NON-canonical run_versions, which baseline run_version's ACTUAL
# saved results (read from disk, never hardcoded a second time) should
# be reproduced instead of the original frozen baseline_v1 numbers.
# Exists specifically for corrected-data re-verification runs: e.g.
# xgboost_v1_a03fix must reproduce baseline_v1_a03fix's own numbers
# (computed from the SAME corrected data, in the same session), not the
# ORIGINAL baseline_v1 numbers, which were deliberately computed from
# different (pre-A03-fix) data and are expected to differ now. Add an
# entry here whenever a new corrected-data run pair is created -- do
# NOT infer this from naming convention (e.g. stripping "_a03fix"),
# since an implicit convention is exactly the kind of thing that drifts
# silently; an explicit registry is checkable and self-documenting.
REPRODUCTION_TARGET_OVERRIDES = {
    "xgboost_v1_a03fix": "baseline_v1_a03fix",
}


def resolve_run_args(args: list) -> tuple:
    """input_filename and run_version are mandatory; elasticnet_search_profile
    is optional and defaults to 'v1'. The canonical name 'xgboost_v1' is
    structurally locked to 'v1' -- passing anything else for that run_version
    raises, rather than merely warning.
    """
    if len(args) not in (2, 3):
        raise SystemExit(
            "Usage:\n"
            "  python run_xgboost.py <input_filename> <run_version> [elasticnet_search_profile]\n\n"
            "Example:\n"
            "  python run_xgboost.py delu_features.parquet xgboost_v1"
        )
    input_filename = args[0]
    run_version = args[1]
    elasticnet_profile = args[2] if len(args) == 3 else "v1"

    if elasticnet_profile not in ELASTICNET_SEARCH_PROFILES:
        raise ValueError(
            f"Unknown ElasticNet search profile '{elasticnet_profile}'. Available: "
            f"{list(ELASTICNET_SEARCH_PROFILES.keys())}"
        )

    expected = CANONICAL_XGBOOST_RUN_PROFILES.get(run_version)
    if expected is not None and elasticnet_profile != expected:
        raise ValueError(
            f"run_version='{run_version}' must use elasticnet_search_profile='{expected}', "
            f"not '{elasticnet_profile}'. This guard exists specifically to prevent the canonical "
            f"xgboost_v1 experiment from silently comparing against the rejected ElasticNet-v2 profile."
        )

    return input_filename, run_version, elasticnet_profile


def load_baseline_reference_from_disk(input_stem: str, baseline_run_version: str) -> dict:
    """Reads a baseline run's ACTUAL saved elasticnet_full MAE per fold
    from outputs/models/<input_stem>/<baseline_run_version>/, rather
    than a second hardcoded dict -- a hand-maintained sentinel here
    would be exactly the kind of thing that silently drifts out of
    sync with what a corrected-data baseline run actually produced.
    Raises FileNotFoundError with a clear message (naming the missing
    run) if that baseline hasn't been run yet -- refuses to guess or
    fall back to the original frozen numbers, since that would produce
    the exact false-positive/false-negative mismatch this override
    exists to prevent.
    """
    baseline_dir = REPO_ROOT / "outputs" / "models" / input_stem / baseline_run_version
    if not baseline_dir.exists():
        raise FileNotFoundError(
            f"Reproduction target '{baseline_run_version}' has no saved results at {baseline_dir} -- "
            f"run it first (e.g. python run_models.py {input_stem}.parquet {baseline_run_version} v1) "
            f"before running the corresponding XGBoost re-verification."
        )
    reference = {}
    for csv_path in sorted(baseline_dir.glob("*_overall_metrics.csv")):
        fold_name = csv_path.name.removesuffix("_overall_metrics.csv")
        df = pd.read_csv(csv_path, index_col=0)
        if "elasticnet_full" not in df.index:
            raise ValueError(f"{csv_path} has no 'elasticnet_full' row -- unexpected format, refusing to guess.")
        reference[fold_name] = float(df.loc["elasticnet_full", "mae"])
    if not reference:
        raise FileNotFoundError(f"No *_overall_metrics.csv files found in {baseline_dir}.")
    return reference


def verify_baseline_v1_reproduction(results: dict, reference: dict = None) -> None:
    """STOP the run if the re-fitted ElasticNet-full metrics don't
    reproduce the reference numbers within a tight tolerance. Guards
    against silent code/data drift making the XGBoost comparison
    meaningless.

    reference defaults to FROZEN_BASELINE_V1_ELASTICNET_FULL_MAE (the
    ORIGINAL, immutable baseline_v1 evidence) -- pass a different
    reference (e.g. from load_baseline_reference_from_disk) when
    re-verifying against deliberately different, corrected data; see
    REPRODUCTION_TARGET_OVERRIDES.
    """
    if reference is None:
        reference = FROZEN_BASELINE_V1_ELASTICNET_FULL_MAE
    mismatches = []
    for fold_name, expected_mae in reference.items():
        if fold_name not in results:
            continue
        actual_mae = results[fold_name].overall_metrics.loc["elasticnet_full", "mae"]
        rel_diff = abs(actual_mae - expected_mae) / expected_mae
        if rel_diff > REPRODUCTION_TOLERANCE_RELATIVE:
            mismatches.append(
                f"  {fold_name}: expected {expected_mae:.6f}, got {actual_mae:.6f} "
                f"({rel_diff*100:.3f}% relative difference, tolerance {REPRODUCTION_TOLERANCE_RELATIVE*100}%)"
            )
    if mismatches:
        raise RuntimeError(
            "REPRODUCTION CHECK FAILED -- the re-fitted ElasticNet-full metrics do not "
            "match the reference numbers within tolerance:\n" + "\n".join(mismatches) +
            "\n\nSTOPPING before printing/saving results. This means something has changed since "
            "the reference was produced (code, data, or dependency versions) -- the XGBoost "
            "comparison below would not be interpretable until this is investigated and resolved."
        )


def apply_promotion_rule(results: dict) -> dict:
    """Applies the pre-registered rule: promote XGBoost-full only if
    row-weighted development MAE improves by >= 1.0% vs ElasticNet-full
    v1. Returns a dict with the decision and supporting numbers so it
    can be printed and saved to the manifest -- not just eyeballed.
    """
    total_n = sum(len(r.predictions) for r in results.values())
    en_weighted = sum(
        r.overall_metrics.loc["elasticnet_full", "mae"] * len(r.predictions) for r in results.values()
    ) / total_n
    xgb_weighted = sum(
        r.overall_metrics.loc["xgboost_full", "mae"] * len(r.predictions) for r in results.values()
    ) / total_n

    improvement = (en_weighted - xgb_weighted) / en_weighted
    promote = improvement >= PROMOTION_THRESHOLD_RELATIVE

    return {
        "elasticnet_full_weighted_mae": en_weighted,
        "xgboost_full_weighted_mae": xgb_weighted,
        "relative_improvement": improvement,
        "promotion_threshold": PROMOTION_THRESHOLD_RELATIVE,
        "decision": "PROMOTE_XGBOOST_FULL" if promote else "RETAIN_ELASTICNET_FULL",
    }


def main():
    cfg = load_config()
    logger = setup_logging(cfg["logging"]["level"])

    input_filename, run_version, elasticnet_search_profile = resolve_run_args(sys.argv[1:])

    in_path = REPO_ROOT / cfg["data"]["processed_dir"] / input_filename
    if not in_path.exists():
        available = list((REPO_ROOT / cfg["data"]["processed_dir"]).glob("*.parquet"))
        raise FileNotFoundError(
            f"{in_path} not found. Run run_features.py first, or pass the "
            f"correct filename: python run_xgboost.py <filename> <run_version>.\n"
            f"Files found: {[p.name for p in available] or 'none'}"
        )

    df = pd.read_parquet(in_path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    table_dir = REPO_ROOT / "outputs" / "models" / in_path.stem / run_version
    if table_dir.exists() and any(table_dir.iterdir()):
        raise FileExistsError(
            f"{table_dir} already contains results. Runs are meant to be immutable "
            f"evidence -- pass a new run_version instead of overwriting it, e.g.\n"
            f"  python run_xgboost.py {input_filename} xgboost_v2"
        )

    results = evaluate_all_folds_with_xgboost(
        df, elasticnet_search_profile=elasticnet_search_profile,
        xgboost_day_aligned_cv=True,  # explicit, not relying on the function default -- auditability
    )

    # STOP here, before writing anything, if the reproduction target isn't matched.
    if elasticnet_search_profile == "v1":
        reproduction_target_run_version = REPRODUCTION_TARGET_OVERRIDES.get(run_version)
        if reproduction_target_run_version is not None:
            reference = load_baseline_reference_from_disk(in_path.stem, reproduction_target_run_version)
            verify_baseline_v1_reproduction(results, reference=reference)
            logger.info(
                "Reproduction check PASSED against '%s' (corrected-data reference, read from disk, "
                "not the original frozen baseline_v1 numbers).", reproduction_target_run_version
            )
        else:
            verify_baseline_v1_reproduction(results)
            logger.info("baseline_v1 reproduction check PASSED -- ElasticNet-full metrics match the "
                        "frozen, documented numbers within tolerance.")

    table_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 78)
    print("ALL SIX MODELS: LAG-24, LAG-168, ELASTICNET (FULL/TIER1), XGBOOST (FULL/TIER1)")
    print(f"elasticnet_search_profile = {elasticnet_search_profile}")
    print("=" * 78)

    all_overall = []
    for fold_name, result in results.items():
        print(f"\n--- Fold: {fold_name} ---")
        print("\nCoverage (raw vs usable-per-model vs common comparison set):")
        print(result.coverage.to_string(index=False))
        result.coverage.to_csv(table_dir / f"{fold_name}_coverage.csv", index=False)

        print("\nOverall metrics (ALL SIX models scored on the identical common row set):")
        print(result.overall_metrics.to_string())
        result.overall_metrics.to_csv(table_dir / f"{fold_name}_overall_metrics.csv")

        for model_name, df_hour in result.metrics_by_hour.items():
            df_hour.to_csv(table_dir / f"{fold_name}_{model_name}_by_hour.csv")

        for model_name, df_regime in result.metrics_by_train_quantile_regime.items():
            df_regime.to_csv(table_dir / f"{fold_name}_{model_name}_by_train_quantile_regime.csv")

        print("\nFixed stress buckets (<0, >200, >500 EUR/MWh -- fold-independent thresholds):")
        for model_name, df_fixed in result.metrics_by_fixed_regime.items():
            df_fixed.to_csv(table_dir / f"{fold_name}_{model_name}_by_fixed_regime.csv")
            if not df_fixed.empty:
                print(f"\n  {model_name}:")
                print(df_fixed.to_string())

        print(f"\nElasticNet hyperparameters: {result.elasticnet_hyperparams}")
        with open(table_dir / f"{fold_name}_elasticnet_hyperparams.json", "w") as f:
            json.dump(result.elasticnet_hyperparams, f, indent=2)
        for variant_name, coef_df in result.elasticnet_coefficients.items():
            coef_df.to_csv(table_dir / f"{fold_name}_{variant_name}_coefficients.csv", index=False)

        print(f"\nXGBoost hyperparameters: {result.xgboost_hyperparams}")
        with open(table_dir / f"{fold_name}_xgboost_hyperparams.json", "w") as f:
            json.dump(result.xgboost_hyperparams, f, indent=2)
        # NOTE: XGBoost gain-based feature importance is descriptive model
        # attribution, not a causal or economically independent contribution
        # claim -- the Full predictor set contains algebraically related
        # variables (load/renewables/residual_load/renewable_share), and
        # tree importance can be redistributed among correlated predictors.
        for variant_name, imp_df in result.xgboost_feature_importances.items():
            imp_df.to_csv(table_dir / f"{fold_name}_{variant_name}_feature_importances.csv", index=False)
            print(f"\nTop 5 XGBoost gain-based feature importances ({variant_name}), descriptive not causal:")
            print(imp_df.head(5).to_string(index=False))

        if result.predictions is not None:
            result.predictions.to_csv(table_dir / f"{fold_name}_predictions.csv", index=False)

        tagged = result.overall_metrics.copy()
        tagged["fold"] = fold_name
        all_overall.append(tagged)

    summary = pd.concat(all_overall)
    summary.to_csv(table_dir / "all_folds_overall_summary.csv")

    promotion_result = apply_promotion_rule(results)
    with open(table_dir / "promotion_rule_result.json", "w") as f:
        json.dump(promotion_result, f, indent=2)

    manifest = {
        "input_file": str(in_path),
        "input_sha256": sha256_file(in_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_version": run_version,
        "elasticnet_search_profile": elasticnet_search_profile,
        "xgboost_param_grid": XGBOOST_PARAM_GRID,
        "xgboost_tree_method": "hist",
        "target": TARGET_COL,
        "models": ["lag_24", "lag_168", "elasticnet_full", "elasticnet_tier1", "xgboost_full", "xgboost_tier1"],
        "primary_metric": "MAE",
        "elasticnet_hyperparameter_selection": "GridSearchCV, scoring=neg_mean_absolute_error, "
                                                "inner cv=TimeSeriesSplit(hourly rows, train-only, never touches "
                                                "outer validation) -- UNCHANGED from frozen baseline_v1/v2, "
                                                "deliberately not delivery-day-aligned (see xgboost_model.py "
                                                "module docstring for why this asymmetry is intentional)",
        "xgboost_hyperparameter_selection": "GridSearchCV, scoring=neg_mean_absolute_error, "
                                             "3 inner splits over unique Europe/Berlin delivery dates; "
                                             "delivery days kept intact (never split across train/val); train-only",
        "xgboost_day_aligned_cv": True,
        "xgboost_cv_timezone": "Europe/Berlin",
        "xgboost_cv_unit": "local_delivery_date",
        "promotion_rule": promotion_result,
        "holdout_used": False,
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "pandas_version": pd.__version__,
        "folds": list(results.keys()),
    }
    with open(table_dir / "model_run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 78)
    print("SUMMARY ACROSS ALL FOLDS")
    print("=" * 78)
    print(summary.to_string())

    print("\n" + "=" * 78)
    print("PRE-REGISTERED PROMOTION RULE APPLIED")
    print("=" * 78)
    print(f"ElasticNet-full (v1, frozen) row-weighted MAE:  {promotion_result['elasticnet_full_weighted_mae']:.4f}")
    print(f"XGBoost-full row-weighted MAE:                  {promotion_result['xgboost_full_weighted_mae']:.4f}")
    print(f"Relative improvement:                           {promotion_result['relative_improvement']*100:.3f}%")
    print(f"Threshold required:                             {promotion_result['promotion_threshold']*100:.1f}%")
    print(f"DECISION: {promotion_result['decision']}")

    print(f"\nSaved all results + manifest to {table_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
