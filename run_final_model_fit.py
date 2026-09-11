"""Run locally: python run_final_model_fit.py <input_filename> <output_run_version>

Example:
  python run_final_model_fit.py delu_features.parquet final_model_fit_v1

FROZEN, IMPLEMENTATION-ONLY step of docs/holdout_protocol_v2.md Section 3.
Fits the final Full and Tier-1 XGBoost models used by the 2026 holdout, on
ALL eligible development data (2019-01-01 Europe/Berlin inclusive to
2026-01-01 Europe/Berlin exclusive), using the already-frozen
model-selection procedure: the same 36-combination XGBoost grid, MAE
scoring, delivery-day-aligned 3-fold inner CV, and fixed random state
used by the accepted development XGBoost pipeline.

Full and Tier-1 are searched INDEPENDENTLY. There is deliberately no CLI
or helper-function surface for supplying an alternative predictor set,
grid, scoring rule, CV splitter, or seed.

The source parquet may contain structurally validated 2026 rows. This
runner may load that file, but it slices immediately to the frozen final
training window and structurally verifies that the dataframe passed into
model fitting contains no timestamp at or after the Europe/Berlin
2026-01-01 holdout boundary. No 2026 row is used for model selection,
training, forecast evaluation, strategy evaluation, or interpretation.

Saves, per information set: the fitted model in XGBoost native JSON,
selected hyperparameters, and predictor list. One shared manifest records
training boundaries/counts, frozen search metadata, random seed, and exact
development/protocol freeze lineage.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from sklearn.model_selection import ParameterGrid

from src.clean import local_delivery_date_to_utc
from src.models import TARGET_COL
from src.splits import get_final_train_window, slice_window
from src.strategy import assert_no_holdout_access
from src.utils import REPO_ROOT, load_config, setup_logging
from src.xgboost_model import (
    XGBOOST_PARAM_GRID,
    XGBOOST_PREDICTOR_COLS,
    XGBOOST_TIER1_PREDICTOR_COLS,
    build_xgboost_search,
    fit_xgboost,
)

# ---------------------------------------------------------------------
# Frozen implementation contract.
# ---------------------------------------------------------------------
RANDOM_SEED = 42
INNER_CV_SPLITS = 3
SCORING = "neg_mean_absolute_error"

# Literal copy of the already-frozen development grid. Keeping a local
# expected value lets this runner fail closed if src.xgboost_model is ever
# edited later while preserving the same number of combinations.
FROZEN_XGBOOST_PARAM_GRID = {
    "max_depth": [3, 4, 5],
    "n_estimators": [200, 400],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.8, 1.0],
}

# Exact predictor sets frozen in the accepted development pipeline and present
# at protocol commit a2d1f9a79022ca1e2731d52803061825038c69f5.
# These are literal on purpose: if src.models/src.xgboost_model drifts later,
# assert_frozen_predictor_contract() will fail before any final model is fit.
FROZEN_FULL_PREDICTOR_COLS = (
    "load_forecast_mw",
    "renewables_forecast_mw",
    "residual_load_forecast_mw",
    "renewable_share_forecast",
    "price_lag_24h",
    "price_lag_48h",
    "price_lag_168h",
    "price_rolling_mean_24h",
    "price_rolling_vol_24h",
    "price_rolling_mean_168h",
    "price_rolling_vol_168h",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "weekend",
    "weekend_hour_sin",
    "weekend_hour_cos",
    "post_15min_mtu",
)

FROZEN_TIER1_PREDICTOR_COLS = (
    "load_forecast_mw",
    "price_lag_24h",
    "price_lag_48h",
    "price_lag_168h",
    "price_rolling_mean_24h",
    "price_rolling_vol_24h",
    "price_rolling_mean_168h",
    "price_rolling_vol_168h",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "weekend",
    "weekend_hour_sin",
    "weekend_hour_cos",
    "post_15min_mtu",
)

FROZEN_PREDICTOR_SETS = {
    "full": FROZEN_FULL_PREDICTOR_COLS,
    "tier1": FROZEN_TIER1_PREDICTOR_COLS,
}

FROZEN_TARGET_COL = "price_eur_mwh"

# Exact provenance frozen before this implementation was built.
PROTOCOL_TAG = "pre-holdout-protocol-v2"
PROTOCOL_COMMIT = "a2d1f9a79022ca1e2731d52803061825038c69f5"
DEVELOPMENT_TAG = "corrected-development-v2-dayorigin"
DEVELOPMENT_COMMIT = "704cc933b99bb32c21cd358e613d1cd667d71488"
XGBOOST_DEVELOPMENT_RUN = "xgboost_v1_a03fix"

TRAIN_START_LOCAL = "2019-01-01"
TRAIN_END_LOCAL_EXCLUSIVE = "2026-01-01"


def resolve_run_args(args: list) -> tuple:
    if len(args) != 2:
        raise SystemExit(
            "Usage:\n"
            "  python run_final_model_fit.py <input_filename> <output_run_version>\n\n"
            "Example:\n"
            "  python run_final_model_fit.py delu_features.parquet final_model_fit_v1\n\n"
            "No overrides accepted for the grid, predictor sets, scoring metric, CV "
            "splitter, or random seed -- this procedure is frozen by "
            "docs/holdout_protocol_v2.md Section 3."
        )
    return args[0], args[1]


def assert_frozen_search_contract() -> None:
    """Fail closed if the imported XGBoost search implementation has drifted.

    This checks values, not merely grid size: a different 36-combination grid
    would still be a protocol violation. It also verifies the GridSearchCV
    scoring rule and the estimator settings that are scientifically material
    to the frozen development procedure.
    """
    if XGBOOST_PARAM_GRID != FROZEN_XGBOOST_PARAM_GRID:
        raise AssertionError(
            "XGBOOST_PARAM_GRID has drifted from the frozen 36-combination "
            "development grid. Do not run the final fit."
        )

    n_combinations = len(list(ParameterGrid(FROZEN_XGBOOST_PARAM_GRID)))
    if n_combinations != 36:
        raise AssertionError(
            f"Frozen XGBoost grid must contain exactly 36 combinations, got {n_combinations}."
        )

    search = build_xgboost_search(
        inner_cv_splits=INNER_CV_SPLITS,
        param_grid=FROZEN_XGBOOST_PARAM_GRID,
    )
    if search.scoring != SCORING:
        raise AssertionError(
            f"Frozen scoring drifted: {search.scoring!r} != {SCORING!r}."
        )

    estimator_params = search.estimator.get_params()
    if estimator_params.get("random_state") != RANDOM_SEED:
        raise AssertionError(
            "Frozen random seed drifted: "
            f"{estimator_params.get('random_state')!r} != {RANDOM_SEED!r}."
        )
    if estimator_params.get("objective") != "reg:absoluteerror":
        raise AssertionError(
            "Frozen XGBoost objective drifted: "
            f"{estimator_params.get('objective')!r} != 'reg:absoluteerror'."
        )
    if estimator_params.get("tree_method") != "hist":
        raise AssertionError(
            "Frozen XGBoost tree_method drifted: "
            f"{estimator_params.get('tree_method')!r} != 'hist'."
        )


def assert_frozen_predictor_contract() -> None:
    """Fail closed if the imported target or predictor lists have drifted."""
    if TARGET_COL != FROZEN_TARGET_COL:
        raise AssertionError(
            f"Frozen target column drifted: {TARGET_COL!r} != {FROZEN_TARGET_COL!r}."
        )

    if tuple(XGBOOST_PREDICTOR_COLS) != FROZEN_FULL_PREDICTOR_COLS:
        raise AssertionError(
            "XGBOOST_PREDICTOR_COLS has drifted from the exact frozen Full "
            "information set. Do not run the final fit."
        )

    if tuple(XGBOOST_TIER1_PREDICTOR_COLS) != FROZEN_TIER1_PREDICTOR_COLS:
        raise AssertionError(
            "XGBOOST_TIER1_PREDICTOR_COLS has drifted from the exact frozen "
            "Tier-1 information set. Do not run the final fit."
        )


def assert_frozen_train_window(window) -> None:
    """Verify src.splits still resolves the exact preregistered Berlin window."""
    expected_start = pd.Timestamp(local_delivery_date_to_utc(TRAIN_START_LOCAL))
    expected_end = pd.Timestamp(
        local_delivery_date_to_utc(TRAIN_END_LOCAL_EXCLUSIVE)
    )

    if pd.Timestamp(window.train_start) != expected_start:
        raise AssertionError(
            "Final-train start drifted from the frozen boundary: "
            f"{window.train_start} != {expected_start}."
        )
    if pd.Timestamp(window.train_end) != expected_end:
        raise AssertionError(
            "Final-train end drifted from the frozen boundary: "
            f"{window.train_end} != {expected_end}."
        )


def fit_final_model(train_df: pd.DataFrame, label: str) -> dict:
    """Fit one frozen information set, independently.

    label is the only selector accepted. The predictor list is resolved from
    FROZEN_PREDICTOR_SETS inside this function, so callers cannot supply an
    arbitrary alternative feature set.
    """
    if label not in FROZEN_PREDICTOR_SETS:
        raise ValueError(
            f"Unknown frozen information set {label!r}; expected one of "
            f"{sorted(FROZEN_PREDICTOR_SETS)}."
        )

    predictor_cols = list(FROZEN_PREDICTOR_SETS[label])

    model, used_predictor_cols, hyperparams = fit_xgboost(
        train_df,
        predictor_cols=predictor_cols,
        target_col=FROZEN_TARGET_COL,
        inner_cv_splits=INNER_CV_SPLITS,
        param_grid=FROZEN_XGBOOST_PARAM_GRID,
        day_aligned_cv=True,  # explicit; never rely on fit_xgboost()'s default
    )

    if list(used_predictor_cols) != predictor_cols:
        raise AssertionError(
            f"{label} fit returned a predictor list different from the frozen input set."
        )

    n_rows_used = int(
        train_df.dropna(subset=predictor_cols + [FROZEN_TARGET_COL]).shape[0]
    )
    return {
        "label": label,
        "model": model,
        "predictor_cols": used_predictor_cols,
        "hyperparameters": hyperparams,
        "n_training_rows": n_rows_used,
    }


def main():
    cfg = load_config()
    logger = setup_logging(cfg["logging"]["level"])

    input_filename, output_run_version = resolve_run_args(sys.argv[1:])
    input_stem = Path(input_filename).stem

    out_dir = (
        REPO_ROOT
        / "outputs"
        / "final_model_fit"
        / input_stem
        / output_run_version
    )
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"{out_dir} already contains results. Pass a new output_run_version "
            "instead of overwriting it."
        )

    # Verify every frozen scientific contract before any model fit.
    assert_frozen_search_contract()
    assert_frozen_predictor_contract()

    in_path = REPO_ROOT / cfg["data"]["processed_dir"] / input_filename
    if not in_path.exists():
        raise FileNotFoundError(f"Missing {in_path}.")

    logger.info("Loading features from '%s'...", in_path)
    df = pd.read_parquet(in_path)

    window = get_final_train_window()
    assert_frozen_train_window(window)
    logger.info(
        "Final train window: %s (inclusive) -> %s (exclusive)",
        window.train_start,
        window.train_end,
    )
    train_df, _ = slice_window(df, window)

    # Belt-and-suspenders holdout rejection. The final-train boundary is
    # 2026-01-01 00:00 Europe/Berlin, i.e. 2025-12-31 23:00 UTC.
    assert_no_holdout_access(
        train_df["timestamp_utc"],
        holdout_start_utc=pd.Timestamp(
            local_delivery_date_to_utc(TRAIN_END_LOCAL_EXCLUSIVE)
        ),
    )
    logger.info(
        "Holdout check passed -- no timestamps at or after final train boundary %s.",
        window.train_end,
    )
    logger.info(
        "%d raw rows in final training window (before per-model NaN dropping).",
        len(train_df),
    )

    print("\n" + "=" * 78)
    print(f"FINAL MODEL FIT: {output_run_version}")
    print(f"Train window: {window.train_start} -> {window.train_end} (exclusive)")
    print("=" * 78)

    results = {}
    for label in ("full", "tier1"):
        logger.info(
            "Fitting final %s model -- independent selection, frozen grid...",
            label,
        )
        results[label] = fit_final_model(train_df, label)

        print(f"\n--- {label} ---")
        print(
            f"  Predictors: {len(results[label]['predictor_cols'])} columns"
        )
        print(
            "  Selected hyperparameters: "
            f"{results[label]['hyperparameters']}"
        )
        print(
            f"  Training rows used: {results[label]['n_training_rows']}"
        )

    # Coincidentally equal parameter VALUES are allowed. Object identity is not:
    # sharing one dict object would indicate reuse of one selection result.
    assert (
        results["full"]["hyperparameters"]
        is not results["tier1"]["hyperparameters"]
    ), (
        "Full and Tier-1 hyperparameters are the SAME object -- one "
        "information set's selection appears to have been reused."
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    for label in ("full", "tier1"):
        result = results[label]
        result["model"].save_model(str(out_dir / f"{label}_model.json"))

        with open(
            out_dir / f"{label}_hyperparameters.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(result["hyperparameters"], f, indent=2)

        with open(
            out_dir / f"{label}_predictor_cols.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(result["predictor_cols"], f, indent=2)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_filename": input_filename,
        "output_run_version": output_run_version,
        "train_start_utc": str(window.train_start),
        "train_end_utc_exclusive": str(window.train_end),
        "train_start_local_date": TRAIN_START_LOCAL,
        "train_end_local_date_exclusive": TRAIN_END_LOCAL_EXCLUSIVE,
        "raw_rows_in_final_train_window": int(len(train_df)),
        "target_col": FROZEN_TARGET_COL,
        "full_n_training_rows": results["full"]["n_training_rows"],
        "tier1_n_training_rows": results["tier1"]["n_training_rows"],
        "full_hyperparameters": results["full"]["hyperparameters"],
        "tier1_hyperparameters": results["tier1"]["hyperparameters"],
        "full_predictor_cols": results["full"]["predictor_cols"],
        "tier1_predictor_cols": results["tier1"]["predictor_cols"],
        "frozen_full_predictor_cols": list(FROZEN_FULL_PREDICTOR_COLS),
        "frozen_tier1_predictor_cols": list(FROZEN_TIER1_PREDICTOR_COLS),
        "param_grid": FROZEN_XGBOOST_PARAM_GRID,
        "param_grid_n_combinations": 36,
        "inner_cv_splits": INNER_CV_SPLITS,
        "day_aligned_cv": True,
        "scoring": SCORING,
        "random_seed": RANDOM_SEED,
        "xgboost_objective": "reg:absoluteerror",
        "xgboost_tree_method": "hist",
        "selection_independent_per_information_set": True,
        "holdout_used": False,
        "protocol": "docs/holdout_protocol_v2.md",
        "protocol_section": 3,
        "protocol_tag": PROTOCOL_TAG,
        "protocol_commit": PROTOCOL_COMMIT,
        "development_evidence_tag": DEVELOPMENT_TAG,
        "development_evidence_commit": DEVELOPMENT_COMMIT,
        "xgboost_development_run": XGBOOST_DEVELOPMENT_RUN,
    }

    with open(
        out_dir / "final_model_fit_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(manifest, f, indent=2, default=str)

    print(
        "\nSaved final Full + Tier-1 models + manifest to "
        f"{out_dir}"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()