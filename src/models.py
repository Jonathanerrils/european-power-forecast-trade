"""Naive baselines and ElasticNet, evaluated on the chronological splits
locked in by src/splits.py BEFORE this module was written.

Per spec section 10-11: establish naive baselines first, then a simple
interpretable statistical model, before touching anything nonlinear
(XGBoost is deliberately not in this file). If XGBoost can't beat these,
that's a result to report honestly, not a reason to hide the comparison.

Baseline 1 (lag-24) and Baseline 2 (lag-168) require NO fitting: they
are literally the price_lag_24h / price_lag_168h columns already built
by features.py, evaluated against the actual price. Using the
already-lagged columns (rather than re-deriving them here) means the
same leakage guard that validated those columns in features.py already
covers these baselines -- there's no second, unverified lag
implementation to trust.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .splits import SplitWindow, get_split_windows, slice_window, assert_split_is_chronological

logger = logging.getLogger("power_forecast.models")

# Frozen predictor set for ElasticNet (spec section 7 + this project's
# point-in-time tiering in features.py). Deliberately excludes raw
# wind_onshore/offshore/solar component columns (40% missing for solar)
# in favour of the already-aggregated renewables_forecast_mw /
# residual_load_forecast_mw, which handle the solar nighttime-fill
# assumption once at construction time rather than re-litigating it here.
# DO NOT add features after seeing model results -- see README's
# pre-registered hypotheses.
ELASTICNET_PREDICTOR_COLS: List[str] = [
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
]

# Tier-1-only predictor set: everything in ELASTICNET_PREDICTOR_COLS
# EXCEPT renewables_forecast_mw / residual_load_forecast_mw /
# renewable_share_forecast, which features.py's FEATURE_AVAILABILITY_TIER
# tags as Tier 2 (not compatible with the 11:45 D-1 decision cutoff --
# EU Reg. 543/2013 Art. 14(2)(d) only guarantees wind/solar forecasts by
# 17:00 D-1). Running this alongside the full model turns a documented
# limitation into an actual robustness experiment: how much of the full
# model's improvement survives using only the load forecast, whose
# regulatory publication deadline is compatible with the 11:45 D-1
# cutoff (Art. 6(2)(b)) -- although, as with everything in this
# dataset, exact historical revision vintage remains unreconstructed?
ELASTICNET_TIER1_PREDICTOR_COLS: List[str] = [
    c for c in ELASTICNET_PREDICTOR_COLS
    if c not in {"renewables_forecast_mw", "residual_load_forecast_mw", "renewable_share_forecast"}
]

TARGET_COL = "price_eur_mwh"


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """MAE (primary, per spec section 11), RMSE (secondary, spikes
    matter), Median Absolute Error. NOT MAPE -- prices can be zero or
    negative, making percentage error meaningless/undefined.
    """
    err = y_true - y_pred
    abs_err = err.abs()
    return {
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "median_ae": float(abs_err.median()),
        "n": int(len(y_true)),
    }


def compute_metrics_by_group(
    df: pd.DataFrame, y_true_col: str, y_pred_col: str, group_col: str
) -> pd.DataFrame:
    rows = []
    for group_val, sub in df.groupby(group_col):
        m = compute_metrics(sub[y_true_col], sub[y_pred_col])
        m[group_col] = group_val
        rows.append(m)
    return pd.DataFrame(rows).set_index(group_col)


# ---------------------------------------------------------------------
# Extreme-price regime (spec section 19) -- threshold from TRAINING
# data only, never touching validation/holdout, so the definition of
# "extreme" can't leak information about what the model will be scored on.
# ---------------------------------------------------------------------
@dataclass
class ExtremeRegimeThreshold:
    train_median: float
    abs_deviation_q95: float

    def is_extreme(self, price: pd.Series) -> pd.Series:
        return (price - self.train_median).abs() > self.abs_deviation_q95


def fit_extreme_regime_threshold(train_prices: pd.Series, quantile: float = 0.95) -> ExtremeRegimeThreshold:
    train_prices = train_prices.dropna()
    median = float(train_prices.median())
    abs_dev = (train_prices - median).abs()
    q = float(abs_dev.quantile(quantile))
    return ExtremeRegimeThreshold(train_median=median, abs_deviation_q95=q)


# ---------------------------------------------------------------------
# Baselines (no fitting -- literally the precomputed lag columns)
# ---------------------------------------------------------------------
def baseline_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Returns df with lag_24_pred / lag_168_pred columns, which are
    exactly price_lag_24h / price_lag_168h -- already validated safe
    by features.py's leakage guard. No re-derivation here on purpose.
    """
    out = df.copy()
    out["lag_24_pred"] = out["price_lag_24h"]
    out["lag_168_pred"] = out["price_lag_168h"]
    return out


# ---------------------------------------------------------------------
# ElasticNet
# ---------------------------------------------------------------------
# Explicit, named search profiles -- NOT a single mutable "current
# default" grid. A prior version of this code had one DEFAULT_ALPHA_GRID
# that silently changed across rounds of fixes; anyone re-running
# `python run_models.py delu_features.parquet baseline_v1` on a fresh
# clone would NOT have reproduced the actual historical baseline_v1
# experiment -- they'd have run whatever grid happened to be the current
# default and saved it under a v1 name. Naming the profiles explicitly
# and requiring the caller to choose one (see ELASTICNET_SEARCH_PROFILES)
# makes "which experiment produced this" a fact recorded in the run
# manifest, not something inferred from an output folder name.
BASELINE_V1_ALPHA_GRID: np.ndarray = np.logspace(-3, 2, 15)
BASELINE_V1_L1_RATIO_GRID: List[float] = [0.1, 0.5, 0.9, 1.0]

# v2 extends v1 as a genuine UNION, not a different discretization --
# baseline_v1 selected alpha=0.001 (v1's exact minimum) in 6/8 fits and
# l1_ratio=0.1 (v1's exact minimum) in 3/8 fits, so v2 adds smaller
# candidates on both dimensions while preserving every v1 value exactly.
_ALPHA_LOWER_EXTENSION: np.ndarray = np.array([1e-5, 3e-5, 1e-4, 3e-4])
BASELINE_V2_ALPHA_GRID: np.ndarray = np.unique(np.concatenate([_ALPHA_LOWER_EXTENSION, BASELINE_V1_ALPHA_GRID]))
BASELINE_V2_L1_RATIO_GRID: List[float] = [0.01] + BASELINE_V1_L1_RATIO_GRID

ELASTICNET_SEARCH_PROFILES: Dict[str, Dict[str, object]] = {
    "v1": {"alpha_grid": BASELINE_V1_ALPHA_GRID, "l1_ratio_grid": BASELINE_V1_L1_RATIO_GRID},
    "v2": {"alpha_grid": BASELINE_V2_ALPHA_GRID, "l1_ratio_grid": BASELINE_V2_L1_RATIO_GRID},
}
DEFAULT_SEARCH_PROFILE = "v2"  # current best-known profile; MUST be passed explicitly by callers that care


def build_elasticnet_search(
    inner_cv_splits: int = 4,
    alpha_grid: np.ndarray = None,
    l1_ratio_grid: List[float] = None,
    search_profile: str = None,
) -> GridSearchCV:
    """Constructs the (unfit) GridSearchCV object: Pipeline(scaler,
    elasticnet) + TimeSeriesSplit inner CV + MAE scoring. Factored out
    from fit_elasticnet() so the architecture itself (not just its
    final output) can be asserted on directly in tests.

    Resolution order: explicit alpha_grid/l1_ratio_grid override
    everything (for tests/one-off experiments); otherwise search_profile
    selects a named profile from ELASTICNET_SEARCH_PROFILES; otherwise
    falls back to DEFAULT_SEARCH_PROFILE. Passing search_profile
    explicitly (as run_models.py does) is how a rerun can prove which
    named experiment it's reproducing.
    """
    if alpha_grid is None or l1_ratio_grid is None:
        profile = ELASTICNET_SEARCH_PROFILES[search_profile or DEFAULT_SEARCH_PROFILE]
        alpha_grid = alpha_grid if alpha_grid is not None else profile["alpha_grid"]
        l1_ratio_grid = l1_ratio_grid if l1_ratio_grid is not None else profile["l1_ratio_grid"]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("elasticnet", ElasticNet(max_iter=5000, random_state=42)),
    ])
    param_grid = {"elasticnet__alpha": alpha_grid, "elasticnet__l1_ratio": l1_ratio_grid}
    inner_cv = TimeSeriesSplit(n_splits=inner_cv_splits)

    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
    )


def fit_elasticnet(
    train_df: pd.DataFrame,
    predictor_cols: List[str] = None,
    target_col: str = TARGET_COL,
    ts_col: str = "timestamp_utc",
    inner_cv_splits: int = 4,
    alpha_grid: np.ndarray = None,
    l1_ratio_grid: List[float] = None,
    search_profile: str = None,
) -> Tuple[ElasticNet, StandardScaler, List[str], Dict[str, float]]:
    """Fit ElasticNet on TRAIN data only, with alpha/l1_ratio selected
    by an inner, train-only, chronological CV.

    Uses a Pipeline(scaler, elasticnet) inside GridSearchCV rather than
    scaling once up front: fitting StandardScaler on the full outer
    training set BEFORE TimeSeriesSplit divides it means each inner
    fold's "training" scaling statistics would include data from LATER
    in the outer training set than that inner fold's own cutoff -- a
    real (if mild) leakage-adjacent inconsistency, since it lets the
    preprocessing see future-relative information the inner CV is
    supposed to be blind to. With a Pipeline, GridSearchCV refits the
    scaler separately inside every individual inner training fold.

    Also scores the inner CV on MAE (scoring='neg_mean_absolute_error'),
    not the ElasticNetCV default of MSE -- this project declared MAE as
    the primary metric precisely because prices are heavy-tailed and
    can be negative, so hyperparameter selection should optimize the
    same criterion the model is actually judged on.

    TimeSeriesSplit operates on ROW ORDER, not timestamp values -- it
    has no idea what a timestamp even is. If the input weren't already
    chronologically sorted, "train-only inner CV" would be silently
    false despite every other safeguard being in place. Rather than
    rely on an incidental upstream property (the parquet happens to be
    sorted, slice_window() happens to preserve order), this explicitly
    sorts by ts_col and asserts strict monotonicity before handing rows
    to TimeSeriesSplit, and rejects duplicate timestamps outright.
    """
    predictor_cols = predictor_cols or ELASTICNET_PREDICTOR_COLS
    complete = (
        train_df.dropna(subset=predictor_cols + [target_col])
        .sort_values(ts_col)
        .reset_index(drop=True)
    )
    if len(complete) < 100:
        raise ValueError(
            f"Only {len(complete)} complete training rows after dropping NaNs -- "
            f"too few to fit a model reliably."
        )
    if complete[ts_col].duplicated().any():
        raise AssertionError(
            f"ElasticNet training data contains duplicate '{ts_col}' values -- "
            f"TimeSeriesSplit's row-order-based CV assumes one row per timestamp."
        )
    if not complete[ts_col].is_monotonic_increasing:
        raise AssertionError(
            "ElasticNet training rows are not chronologically ordered after sorting -- "
            "this should be unreachable; investigate the input data."
        )

    X = complete[predictor_cols].values
    y = complete[target_col].values

    search = build_elasticnet_search(
        inner_cv_splits=inner_cv_splits, alpha_grid=alpha_grid, l1_ratio_grid=l1_ratio_grid,
        search_profile=search_profile,
    )
    search.fit(X, y)

    best_alpha = search.best_params_["elasticnet__alpha"]
    best_l1_ratio = search.best_params_["elasticnet__l1_ratio"]
    logger.info(
        "GridSearchCV selected alpha=%.4g, l1_ratio=%.2f (inner train-only "
        "TimeSeriesSplit, %d folds, MAE-scored)", best_alpha, best_l1_ratio, inner_cv_splits,
    )
    logger.info("Fit ElasticNet on %d rows (dropped %d with missing predictor/target)",
                len(complete), len(train_df) - len(complete))

    best_pipeline = search.best_estimator_
    scaler = best_pipeline.named_steps["scaler"]
    model = best_pipeline.named_steps["elasticnet"]
    hyperparams = {"alpha": float(best_alpha), "l1_ratio": float(best_l1_ratio)}
    return model, scaler, predictor_cols, hyperparams


def predict_elasticnet(
    model: ElasticNet, scaler: StandardScaler, df: pd.DataFrame, predictor_cols: List[str]
) -> pd.Series:
    """Predict for rows with complete predictors; NaN elsewhere (never
    silently impute -- a row we can't safely predict for should show up
    as missing in the evaluation, not as a guessed number).
    """
    complete_mask = df[predictor_cols].notna().all(axis=1)
    preds = pd.Series(np.nan, index=df.index)
    if complete_mask.any():
        X = df.loc[complete_mask, predictor_cols].values
        X_scaled = scaler.transform(X)
        preds.loc[complete_mask] = model.predict(X_scaled)
    return preds


# ---------------------------------------------------------------------
# Full fold evaluation
# ---------------------------------------------------------------------
@dataclass
class FoldResult:
    fold_name: str
    overall_metrics: pd.DataFrame       # index: model name, columns: mae/rmse/median_ae/n -- ALL on the same common-row set
    coverage: pd.DataFrame              # raw val rows, usable-per-model rows, common comparison rows
    metrics_by_hour: Dict[str, pd.DataFrame] = field(default_factory=dict)
    metrics_by_train_quantile_regime: Dict[str, pd.DataFrame] = field(default_factory=dict)
    metrics_by_fixed_regime: Dict[str, pd.DataFrame] = field(default_factory=dict)
    predictions: pd.DataFrame = None    # per-row predictions for every model, on the common set
    elasticnet_coefficients: Dict[str, pd.DataFrame] = field(default_factory=dict)  # keyed "elasticnet_full" / "elasticnet_tier1"
    elasticnet_hyperparams: Dict[str, Dict[str, float]] = field(default_factory=dict)


FIXED_STRESS_BUCKETS = {
    "negative_price": lambda p: p < 0,
    "price_gt_200": lambda p: p > 200,
    "price_gt_500": lambda p: p > 500,
}


def evaluate_fold(df: pd.DataFrame, window: SplitWindow, search_profile: str = None) -> FoldResult:
    assert_split_is_chronological(window)  # re-check even though get_split_windows() already validated

    train_df, val_df = slice_window(df, window)
    train_df = baseline_predictions(train_df)
    val_df = baseline_predictions(val_df).copy()

    elasticnet_variants = {
        "elasticnet_full": ELASTICNET_PREDICTOR_COLS,
        "elasticnet_tier1": ELASTICNET_TIER1_PREDICTOR_COLS,
    }
    fitted = {}
    for variant_name, cols in elasticnet_variants.items():
        model, scaler, used_cols, hyperparams = fit_elasticnet(
            train_df, predictor_cols=cols, search_profile=search_profile
        )
        val_df[f"{variant_name}_pred"] = predict_elasticnet(model, scaler, val_df, used_cols)
        fitted[variant_name] = (model, scaler, used_cols, hyperparams)

    pred_cols = {
        "lag_24": "lag_24_pred",
        "lag_168": "lag_168_pred",
        "elasticnet_full": "elasticnet_full_pred",
        "elasticnet_tier1": "elasticnet_tier1_pred",
    }

    # Coverage report BEFORE restricting to a common set -- shows exactly
    # how many rows each model could individually score, and how many
    # are usable by all four, so nothing is hidden by the common-mask fix.
    raw_n = len(val_df)
    coverage_rows = {"raw_val_rows": raw_n}
    for name, pred_col in pred_cols.items():
        coverage_rows[f"{name}_usable_rows"] = int(val_df[[TARGET_COL, pred_col]].notna().all(axis=1).sum())

    # Common evaluation mask: every model is scored on EXACTLY the same
    # rows, so a reported improvement means the same validation instances
    # for every model, not an artifact of one model dropping harder or
    # easier rows via its own NaN pattern.
    common_mask = val_df[TARGET_COL].notna()
    for pred_col in pred_cols.values():
        common_mask &= val_df[pred_col].notna()
    common_val = val_df.loc[common_mask].copy()
    coverage_rows["common_comparison_rows"] = len(common_val)
    coverage = pd.DataFrame([coverage_rows])

    # Extreme-price flags: BOTH definitions, computed on the common set.
    # (a) training-quantile-based (spec section 19's original definition,
    #     regime-relative but can drift a lot depending on training years)
    # (b) fixed, fold-independent thresholds (<0, >200, >500 EUR/MWh) --
    #     directly comparable across folds, no validation-derived tuning.
    threshold = fit_extreme_regime_threshold(train_df[TARGET_COL])
    common_val["is_extreme_train_quantile"] = threshold.is_extreme(common_val[TARGET_COL])

    overall_rows = []
    by_hour = {}
    by_train_quantile_regime = {}
    by_fixed_regime = {}
    for name, pred_col in pred_cols.items():
        m = compute_metrics(common_val[TARGET_COL], common_val[pred_col])
        m["model"] = name
        overall_rows.append(m)

        by_hour[name] = compute_metrics_by_group(common_val, TARGET_COL, pred_col, "hour_local")
        by_train_quantile_regime[name] = compute_metrics_by_group(
            common_val, TARGET_COL, pred_col, "is_extreme_train_quantile"
        )

        fixed_rows = []
        for bucket_name, bucket_fn in FIXED_STRESS_BUCKETS.items():
            mask = bucket_fn(common_val[TARGET_COL])
            if mask.sum() == 0:
                continue
            fm = compute_metrics(common_val.loc[mask, TARGET_COL], common_val.loc[mask, pred_col])
            fm["bucket"] = bucket_name
            fixed_rows.append(fm)
        by_fixed_regime[name] = pd.DataFrame(fixed_rows).set_index("bucket") if fixed_rows else pd.DataFrame()

    overall = pd.DataFrame(overall_rows).set_index("model")

    # ElasticNet standardized coefficients for BOTH variants -- useful for
    # predictive interpretation (direction/relative magnitude across
    # folds), NOT independent causal effects: residual_load_forecast_mw,
    # renewables_forecast_mw, and renewable_share_forecast are
    # algebraically related (residual_load = load - renewables; share =
    # renewables / load), so ElasticNet can distribute weight across them
    # in ways that don't reflect a clean, separable causal contribution.
    coefficients = {}
    hyperparams_out = {}
    for variant_name, (model, scaler, used_cols, hyperparams) in fitted.items():
        coef_values = model.coef_ if hasattr(model, "coef_") else None
        if coef_values is not None:
            coefficients[variant_name] = pd.DataFrame({
                "feature": used_cols,
                "standardized_coefficient": coef_values,
                "abs_coefficient": np.abs(coef_values),
            }).sort_values("abs_coefficient", ascending=False)
        hyperparams_out[variant_name] = hyperparams

    predictions = common_val[["timestamp_utc", TARGET_COL] + list(pred_cols.values())].copy()

    return FoldResult(
        fold_name=window.name,
        overall_metrics=overall,
        coverage=coverage,
        metrics_by_hour=by_hour,
        metrics_by_train_quantile_regime=by_train_quantile_regime,
        metrics_by_fixed_regime=by_fixed_regime,
        predictions=predictions,
        elasticnet_coefficients=coefficients,
        elasticnet_hyperparams=hyperparams_out,
    )


def evaluate_all_folds(
    df: pd.DataFrame, windows: List[SplitWindow] = None, search_profile: str = None
) -> Dict[str, FoldResult]:
    windows = windows or get_split_windows()
    results = {}
    for w in windows:
        logger.info("Evaluating fold '%s': train %s->%s, val %s->%s (search_profile=%s)",
                    w.name, w.train_start, w.train_end, w.val_start, w.val_end,
                    search_profile or DEFAULT_SEARCH_PROFILE)
        results[w.name] = evaluate_fold(df, w, search_profile=search_profile)
    return results
