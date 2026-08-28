"""Run locally: python run_models.py <input_filename> <run_version> <search_profile>

All three arguments are mandatory -- there is no default for
search_profile. Evaluates lag-24, lag-168, ElasticNet-full, and
ElasticNet-Tier1 across every chronological fold from src/splits.py
(including the regime-stress-test fold), and saves per-fold metrics
tables. XGBoost deliberately not included here -- per spec section 10,
naive baselines and the interpretable statistical model must be
established and saved BEFORE any nonlinear model is tried.

search_profile ("v1" or "v2", from src.models.ELASTICNET_SEARCH_PROFILES)
is REQUIRED on the command line, not defaulted, and is additionally
guarded for the two canonical experiment names: run_version="baseline_v1"
must use search_profile="v1", and "baseline_v2" must use "v2" -- enforced
by resolve_run_args(), not just convention. An earlier version of this
script defaulted search_profile when omitted, so a bare
`python run_models.py` could silently create a directory named
'baseline_v1' containing a 'v2' experiment. Naming the profile
explicitly on the command line, and recording it in the manifest,
makes "which experiment produced this" a fact you can check.

Outputs are scoped by input file and run version
(outputs/models/<input_stem>/<run_version>/) so a later rerun (e.g.
after adding XGBoost) can never silently overwrite the frozen baseline
evidence -- pass a new run_version explicitly if you want a fresh,
non-overwriting run.
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

from src.models import evaluate_all_folds, TARGET_COL, ELASTICNET_SEARCH_PROFILES
from src.utils import load_config, setup_logging, REPO_ROOT


def sha256_file(path: Path) -> str:
    """Content hash of the input file, not just its name/path. A
    filename like 'delu_features.parquet' can refer to different bytes
    across reruns of the upstream pipeline -- the manifest should be
    able to prove which exact feature matrix produced a given result,
    not just which filename was passed on the command line.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# Canonical experiment names -> the ONLY search_profile allowed for them.
# Prevents e.g. "baseline_v1" accidentally being created with the v2 grid --
# structurally, not just by user discipline. Other run_version names (for
# ad-hoc experiments) aren't constrained by this map.
CANONICAL_RUN_PROFILES = {"baseline_v1": "v1", "baseline_v2": "v2"}


def resolve_run_args(args: list) -> tuple:
    """Parses and validates CLI args. ALL THREE are mandatory -- no
    defaulting search_profile to DEFAULT_SEARCH_PROFILE, because that
    was exactly the failure mode this function exists to close: a bare
    `python run_models.py` would silently create a directory named
    'baseline_v1' containing a 'v2' experiment. Also enforces the
    canonical-name guard: 'baseline_v1' must use search_profile='v1'
    and 'baseline_v2' must use 'v2', structurally, not by convention.
    """
    if len(args) != 3:
        raise SystemExit(
            "Usage:\n"
            "  python run_models.py <input_filename> <run_version> <search_profile>\n\n"
            "Examples:\n"
            "  python run_models.py delu_features.parquet baseline_v1 v1\n"
            "  python run_models.py delu_features.parquet baseline_v2 v2"
        )
    input_filename, run_version, search_profile = args

    if search_profile not in ELASTICNET_SEARCH_PROFILES:
        raise ValueError(
            f"Unknown search_profile '{search_profile}'. Available profiles: "
            f"{list(ELASTICNET_SEARCH_PROFILES.keys())}"
        )

    expected_profile = CANONICAL_RUN_PROFILES.get(run_version)
    if expected_profile is not None and search_profile != expected_profile:
        raise ValueError(
            f"run_version='{run_version}' must use search_profile='{expected_profile}', "
            f"not '{search_profile}'. This guard exists specifically to prevent a canonical "
            f"baseline experiment being silently created with the wrong search grid."
        )

    return input_filename, run_version, search_profile


def main():
    cfg = load_config()
    logger = setup_logging(cfg["logging"]["level"])

    input_filename, run_version, search_profile = resolve_run_args(sys.argv[1:])

    in_path = REPO_ROOT / cfg["data"]["processed_dir"] / input_filename
    if not in_path.exists():
        available = list((REPO_ROOT / cfg["data"]["processed_dir"]).glob("*.parquet"))
        raise FileNotFoundError(
            f"{in_path} not found. Run run_features.py first, or pass the "
            f"correct filename: python run_models.py <filename>.\n"
            f"Files found: {[p.name for p in available] or 'none'}"
        )

    df = pd.read_parquet(in_path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    table_dir = REPO_ROOT / "outputs" / "models" / in_path.stem / run_version
    if table_dir.exists() and any(table_dir.iterdir()):
        raise FileExistsError(
            f"{table_dir} already contains results. Frozen baseline runs are meant to be "
            f"immutable evidence -- pass a new run_version instead of overwriting it, e.g.\n"
            f"  python run_models.py {input_filename} baseline_v2 v2"
        )
    table_dir.mkdir(parents=True, exist_ok=True)

    results = evaluate_all_folds(df, search_profile=search_profile)

    print("\n" + "=" * 78)
    print("BASELINE + ELASTICNET (FULL & TIER-1) RESULTS BY FOLD")
    print(f"search_profile = {search_profile}")
    print("=" * 78)

    all_overall = []
    for fold_name, result in results.items():
        print(f"\n--- Fold: {fold_name} ---")
        print("\nCoverage (raw vs usable-per-model vs common comparison set):")
        print(result.coverage.to_string(index=False))
        result.coverage.to_csv(table_dir / f"{fold_name}_coverage.csv", index=False)

        print("\nOverall metrics (ALL models scored on the identical common row set):")
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

        print(f"\nElasticNet hyperparameters (train-only inner CV, MAE-scored): {result.elasticnet_hyperparams}")
        with open(table_dir / f"{fold_name}_elasticnet_hyperparams.json", "w") as f:
            json.dump(result.elasticnet_hyperparams, f, indent=2)
        for variant_name, coef_df in result.elasticnet_coefficients.items():
            coef_df.to_csv(table_dir / f"{fold_name}_{variant_name}_coefficients.csv", index=False)
            print(f"\nTop 5 standardized coefficients by magnitude ({variant_name}):")
            print(coef_df.head(5).to_string(index=False))

        if result.predictions is not None:
            result.predictions.to_csv(table_dir / f"{fold_name}_predictions.csv", index=False)

        tagged = result.overall_metrics.copy()
        tagged["fold"] = fold_name
        all_overall.append(tagged)

    summary = pd.concat(all_overall)
    summary.to_csv(table_dir / "all_folds_overall_summary.csv")

    manifest = {
        "input_file": str(in_path),
        "input_sha256": sha256_file(in_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_version": run_version,
        "search_profile": search_profile,
        "alpha_grid": [float(a) for a in ELASTICNET_SEARCH_PROFILES[search_profile]["alpha_grid"]],
        "l1_ratio_grid": [float(l) for l in ELASTICNET_SEARCH_PROFILES[search_profile]["l1_ratio_grid"]],
        "target": TARGET_COL,
        "models": ["lag_24", "lag_168", "elasticnet_full", "elasticnet_tier1"],
        "primary_metric": "MAE",
        "hyperparameter_selection": "GridSearchCV, scoring=neg_mean_absolute_error, "
                                     "inner cv=TimeSeriesSplit(train-only, never touches outer validation)",
        "inner_cv_splits": 4,
        "holdout_used": False,
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "folds": list(results.keys()),
    }
    with open(table_dir / "model_run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 78)
    print("SUMMARY ACROSS ALL FOLDS")
    print("=" * 78)
    print(summary.to_string())
    print(f"\nSaved all results + manifest to {table_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
