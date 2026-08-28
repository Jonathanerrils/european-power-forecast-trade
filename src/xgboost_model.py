"""XGBoost, evaluated against every previously-frozen model on the same
chronological splits and the same identical common-row set. Per spec
section 10, this is deliberately the LAST model built -- naive
baselines and the interpretable linear model (frozen as baseline_v1,
see README's "Baseline & ElasticNet Results") were established first,
and their numbers are not touched here.

XGBoost's job is specific, not generic: beat baseline_v1's frozen
row-weighted MAE (19.84), and in particular test whether it helps in
the regime_stress_test fold (where ElasticNet-full's median-hour
performance did not actually beat lag-24) and whether it narrows or
widens the Tier-1/Tier-2 dependency gap (23% retention in the newest
regime vs 40-55% in earlier folds). "XGBoost is more sophisticated" is
not itself a justification -- see spec section 4/H4 in README.

Predictor sets are DELIBERATELY IDENTICAL to ElasticNet's
(ELASTICNET_PREDICTOR_COLS / ELASTICNET_TIER1_PREDICTOR_COLS, imported
from src.models, not redefined here) so performance differences cannot
be attributed to XGBoost receiving additional information. The
comparison nevertheless includes each model's own frozen
model-selection procedure, not model family alone: ElasticNet retains
its previously frozen hourly TimeSeriesSplit inner CV, while XGBoost
uses delivery-day-aligned inner CV introduced before its first
real-data run (see below). The two models have identical information
sets and identical outer evaluation, but different inner tuning
protocols -- "XGBoost beat ElasticNet by X%" should be read as "the
XGBoost pipeline, as specified, beat the ElasticNet pipeline, as
specified," not as an isolated causal estimate of nonlinearity alone.

XGBoost's inner CV is aligned to Europe/Berlin DELIVERY DATES, not
arbitrary hourly rows (see make_delivery_day_cv). The simulated
decision point is D-1 11:45 -- at that moment, ALL hours of delivery
day D are being forecast at once, with none of D's realized prices
available yet. Ordinary hourly TimeSeriesSplit can place different
hours of the SAME delivery day on opposite sides of an inner-CV
boundary. This is NOT feature leakage (no predictor encodes same-day
identity, and every inner-train row is still chronologically before
every inner-val row -- no feature ever sees a later timestamp's
information) and it does NOT contaminate outer validation (no outer
row is ever used for inner training or tuning). But it is a real,
distinct problem: the fitted model's parameters ARE a channel from
earlier hours of D into predictions for later hours of D -- if an
inner-training fold contains realized targets from D 00:00-10:00 and
inner-validation starts at D 11:00, those earlier-in-day outcomes have
already influenced the fitted coefficients/tree structure, even though
at the real D-1 11:45 forecast origin none of those D targets existed
yet. This is best described as an inner-CV FORECAST-ORIGIN ALIGNMENT
problem, potentially amplified by within-day residual dependence
(correlated same-day residuals from unmodeled shocks -- weather, an
unplanned outage -- make hyperparameter selection mildly optimistic
when one day's hours are split across train/val), rather than as
"leakage" in the feature-contamination sense.

This is applied to XGBoost (not yet run against real data -- fixing
its design now is finishing the build, not reopening a decision) but
DELIBERATELY NOT retrofitted onto ElasticNet's frozen baseline_v1/v2
inner CV. Delivery-day alignment changes only the inner
model-selection procedure and never gives ElasticNet access to
outer-validation observations -- but it COULD select different
hyperparameters and therefore indirectly change outer-validation
predictions and MAE if retrofitted (no data contamination is not the
same claim as no effect on results). Because baseline_v1 and
baseline_v2 were already frozen under a documented hourly-CV
methodology (and this script's own verify_baseline_v1_reproduction()
guard exists specifically to detect exactly this kind of drift), they
are preserved rather than retrospectively redefined. The prior
baseline_v2 grid-extension experiment suggests limited sensitivity to
further weakening regularization -- it does NOT directly test
sensitivity to delivery-day-vs-hourly CV, which is a different axis of
variation entirely; it is cited here only as weak, indirect context,
not as evidence this specific change would have zero effect.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

from .models import (
    TARGET_COL,
    ELASTICNET_PREDICTOR_COLS,
    ELASTICNET_TIER1_PREDICTOR_COLS,
    compute_metrics,
    compute_metrics_by_group,
    fit_extreme_regime_threshold,
    baseline_predictions,
    fit_elasticnet,
    predict_elasticnet,
    FIXED_STRESS_BUCKETS,
)
from .splits import SplitWindow, get_split_windows, slice_window, assert_split_is_chronological

logger = logging.getLogger("power_forecast.xgboost_model")

# Predictor sets are the SAME as ElasticNet's -- see module docstring
# for why this must not be a different (e.g. richer) feature set.
XGBOOST_PREDICTOR_COLS: List[str] = ELASTICNET_PREDICTOR_COLS
XGBOOST_TIER1_PREDICTOR_COLS: List[str] = ELASTICNET_TIER1_PREDICTOR_COLS

# Small, pre-specified grid -- mirrors config.yaml's models.xgboost_param_grid
# exactly (frozen there before any model was built). Per spec section 9,
# "tune only a small parameter space using chronological validation" --
# not an open-ended search. 3*2*3*2 = 36 combinations.
XGBOOST_PARAM_GRID: Dict[str, list] = {
    "max_depth": [3, 4, 5],
    "n_estimators": [200, 400],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.8, 1.0],
}


def make_delivery_day_cv(timestamps: pd.Series, n_splits: int, local_tz: str = "Europe/Berlin") -> List[Tuple[np.ndarray, np.ndarray]]:
    """Inner-CV splits aligned to Europe/Berlin DELIVERY DATES, not
    arbitrary hourly rows. See module docstring for the full rationale:
    the simulated decision point (D-1 11:45) forecasts ALL hours of
    delivery day D at once, so no inner-CV split should place different
    hours of the same delivery day on opposite sides of a train/val
    boundary. Handles DST correctly by construction -- the unit being
    split is the local CALENDAR DATE, so a 23-hour spring day or a
    25-hour autumn day is still exactly one indivisible unit, however
    many hourly rows it actually contains.

    timestamps must correspond 1:1 with the rows to be split (same
    order, same length) -- typically complete[ts_col] after the
    chronological-sort/dedup guard in fit_xgboost() has already run.
    """
    ts = pd.to_datetime(timestamps, utc=True).reset_index(drop=True)
    local_days = ts.dt.tz_convert(local_tz).dt.date

    unique_days = np.array(sorted(pd.unique(local_days)))
    if len(unique_days) < n_splits + 1:
        raise ValueError(
            f"Only {len(unique_days)} unique delivery days available -- too few for "
            f"{n_splits} splits (need at least {n_splits + 1})."
        )

    day_splitter = TimeSeriesSplit(n_splits=n_splits)
    splits = []
    for train_day_idx, val_day_idx in day_splitter.split(unique_days):
        train_days = set(unique_days[train_day_idx])
        val_days = set(unique_days[val_day_idx])
        train_idx = np.flatnonzero(local_days.isin(train_days).values)
        val_idx = np.flatnonzero(local_days.isin(val_days).values)
        splits.append((train_idx, val_idx))
    return splits


def build_xgboost_search(
    inner_cv_splits: int = 3,
    param_grid: Dict[str, list] = None,
    cv=None,
) -> GridSearchCV:
    """Constructs the (unfit) GridSearchCV object for XGBoost. No
    scaler needed -- tree-based models are scale-invariant, unlike
    ElasticNet, so there's no analogous scaler-before-inner-CV leakage
    risk to guard against here.

    cv: pass a pre-computed list of (train_idx, val_idx) tuples (e.g.
    from make_delivery_day_cv) for delivery-day-aligned splits. Falls
    back to plain hourly-row TimeSeriesSplit if not provided -- used by
    tests that don't need day-alignment and want a simpler, faster path.
    """
    param_grid = param_grid or XGBOOST_PARAM_GRID
    model = XGBRegressor(
        objective="reg:absoluteerror",  # matches the project's declared primary metric (MAE), not squared error
        tree_method="hist",  # explicit, not left to XGBoost's automatic choice -- also materially faster for this grid size
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    if cv is None:
        cv = TimeSeriesSplit(n_splits=inner_cv_splits)
    return GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=1,  # model itself is already n_jobs=-1; avoid oversubscribing cores
        refit=True,
    )


def fit_xgboost(
    train_df: pd.DataFrame,
    predictor_cols: List[str] = None,
    target_col: str = TARGET_COL,
    ts_col: str = "timestamp_utc",
    inner_cv_splits: int = 3,
    param_grid: Dict[str, list] = None,
    day_aligned_cv: bool = True,
) -> Tuple[XGBRegressor, List[str], Dict[str, object]]:
    """Fit XGBoost on TRAIN data only, hyperparameters selected by an
    inner, train-only CV -- delivery-day-aligned by default
    (make_delivery_day_cv), not plain hourly TimeSeriesSplit. Same
    chronological discipline as fit_elasticnet(): explicit
    chronological sort, duplicate-timestamp rejection, monotonicity
    assertion before any CV splitter ever sees the rows.
    """
    predictor_cols = predictor_cols or XGBOOST_PREDICTOR_COLS
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
            f"XGBoost training data contains duplicate '{ts_col}' values -- "
            f"TimeSeriesSplit's row-order-based CV assumes one row per timestamp."
        )
    if not complete[ts_col].is_monotonic_increasing:
        raise AssertionError(
            "XGBoost training rows are not chronologically ordered after sorting -- "
            "this should be unreachable; investigate the input data."
        )

    X = complete[predictor_cols].values
    y = complete[target_col].values

    cv = make_delivery_day_cv(complete[ts_col], n_splits=inner_cv_splits) if day_aligned_cv else None
    search = build_xgboost_search(inner_cv_splits=inner_cv_splits, param_grid=param_grid, cv=cv)
    search.fit(X, y)

    logger.info("GridSearchCV selected XGBoost params=%s (inner train-only %s, "
                "%d folds, MAE-scored)", search.best_params_,
                "delivery-day-aligned CV" if day_aligned_cv else "hourly TimeSeriesSplit", inner_cv_splits)
    logger.info("Fit XGBoost on %d rows (dropped %d with missing predictor/target)",
                len(complete), len(train_df) - len(complete))

    return search.best_estimator_, predictor_cols, dict(search.best_params_)


def predict_xgboost(model: XGBRegressor, df: pd.DataFrame, predictor_cols: List[str]) -> pd.Series:
    """Predict for rows with complete predictors; NaN elsewhere (same
    policy as predict_elasticnet -- never silently impute).
    """
    complete_mask = df[predictor_cols].notna().all(axis=1)
    preds = pd.Series(np.nan, index=df.index)
    if complete_mask.any():
        X = df.loc[complete_mask, predictor_cols].values
        preds.loc[complete_mask] = model.predict(X)
    return preds


# ---------------------------------------------------------------------
# Full fold evaluation: all SIX models (lag-24, lag-168, ElasticNet
# full/Tier-1, XGBoost full/Tier-1) on one identical common-row set.
# ---------------------------------------------------------------------
@dataclass
class XGBoostFoldResult:
    fold_name: str
    overall_metrics: pd.DataFrame
    coverage: pd.DataFrame
    metrics_by_hour: Dict[str, pd.DataFrame] = field(default_factory=dict)
    metrics_by_train_quantile_regime: Dict[str, pd.DataFrame] = field(default_factory=dict)
    metrics_by_fixed_regime: Dict[str, pd.DataFrame] = field(default_factory=dict)
    predictions: pd.DataFrame = None
    elasticnet_coefficients: Dict[str, pd.DataFrame] = field(default_factory=dict)
    elasticnet_hyperparams: Dict[str, Dict[str, float]] = field(default_factory=dict)
    xgboost_feature_importances: Dict[str, pd.DataFrame] = field(default_factory=dict)
    xgboost_hyperparams: Dict[str, Dict[str, object]] = field(default_factory=dict)


def evaluate_fold_with_xgboost(
    df: pd.DataFrame,
    window: SplitWindow,
    elasticnet_search_profile: str = "v1",  # the ACCEPTED, frozen ElasticNet profile -- see README
    xgboost_param_grid: Dict[str, list] = None,  # override for testing/experimentation; None = XGBOOST_PARAM_GRID
    xgboost_inner_cv_splits: int = 3,
    xgboost_day_aligned_cv: bool = True,
) -> XGBoostFoldResult:
    assert_split_is_chronological(window)

    train_df, val_df = slice_window(df, window)
    train_df = baseline_predictions(train_df)
    val_df = baseline_predictions(val_df).copy()

    # --- ElasticNet full + Tier-1, using the FROZEN baseline_v1 profile ---
    elasticnet_variants = {
        "elasticnet_full": ELASTICNET_PREDICTOR_COLS,
        "elasticnet_tier1": ELASTICNET_TIER1_PREDICTOR_COLS,
    }
    en_fitted = {}
    en_coefficients = {}
    en_hyperparams = {}
    for variant_name, cols in elasticnet_variants.items():
        model, scaler, used_cols, hyperparams = fit_elasticnet(
            train_df, predictor_cols=cols, search_profile=elasticnet_search_profile
        )
        val_df[f"{variant_name}_pred"] = predict_elasticnet(model, scaler, val_df, used_cols)
        en_fitted[variant_name] = (model, scaler, used_cols, hyperparams)
        en_hyperparams[variant_name] = hyperparams
        coef_values = model.coef_ if hasattr(model, "coef_") else None
        if coef_values is not None:
            en_coefficients[variant_name] = pd.DataFrame({
                "feature": used_cols,
                "standardized_coefficient": coef_values,
                "abs_coefficient": np.abs(coef_values),
            }).sort_values("abs_coefficient", ascending=False)

    # --- XGBoost full + Tier-1, same predictor sets ---
    xgb_variants = {
        "xgboost_full": XGBOOST_PREDICTOR_COLS,
        "xgboost_tier1": XGBOOST_TIER1_PREDICTOR_COLS,
    }
    xgb_fitted = {}
    xgb_importances = {}
    xgb_hyperparams = {}
    for variant_name, cols in xgb_variants.items():
        model, used_cols, hyperparams = fit_xgboost(
            train_df, predictor_cols=cols,
            param_grid=xgboost_param_grid, inner_cv_splits=xgboost_inner_cv_splits,
            day_aligned_cv=xgboost_day_aligned_cv,
        )
        val_df[f"{variant_name}_pred"] = predict_xgboost(model, val_df, used_cols)
        xgb_fitted[variant_name] = (model, used_cols, hyperparams)
        xgb_hyperparams[variant_name] = hyperparams
        importances = model.feature_importances_
        xgb_importances[variant_name] = pd.DataFrame({
            "feature": used_cols,
            "importance": importances,
        }).sort_values("importance", ascending=False)

    pred_cols = {
        "lag_24": "lag_24_pred",
        "lag_168": "lag_168_pred",
        "elasticnet_full": "elasticnet_full_pred",
        "elasticnet_tier1": "elasticnet_tier1_pred",
        "xgboost_full": "xgboost_full_pred",
        "xgboost_tier1": "xgboost_tier1_pred",
    }

    # Coverage BEFORE the common mask -- shows exactly how many rows
    # each of the six models could individually score.
    raw_n = len(val_df)
    coverage_rows = {"raw_val_rows": raw_n}
    for name, pred_col in pred_cols.items():
        coverage_rows[f"{name}_usable_rows"] = int(val_df[[TARGET_COL, pred_col]].notna().all(axis=1).sum())

    # Common evaluation mask across ALL SIX models -- same discipline as
    # models.py::evaluate_fold. A reported "XGBoost beats ElasticNet by X%"
    # must mean the identical validation rows for both, not an artifact
    # of different NaN-dropping behavior between the two model families.
    common_mask = val_df[TARGET_COL].notna()
    for pred_col in pred_cols.values():
        common_mask &= val_df[pred_col].notna()
    common_val = val_df.loc[common_mask].copy()
    coverage_rows["common_comparison_rows"] = len(common_val)
    coverage = pd.DataFrame([coverage_rows])

    # Extreme-price flags: both definitions, same as models.py.
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
    predictions = common_val[["timestamp_utc", TARGET_COL] + list(pred_cols.values())].copy()

    return XGBoostFoldResult(
        fold_name=window.name,
        overall_metrics=overall,
        coverage=coverage,
        metrics_by_hour=by_hour,
        metrics_by_train_quantile_regime=by_train_quantile_regime,
        metrics_by_fixed_regime=by_fixed_regime,
        predictions=predictions,
        elasticnet_coefficients=en_coefficients,
        elasticnet_hyperparams=en_hyperparams,
        xgboost_feature_importances=xgb_importances,
        xgboost_hyperparams=xgb_hyperparams,
    )


def evaluate_all_folds_with_xgboost(
    df: pd.DataFrame,
    windows: List[SplitWindow] = None,
    elasticnet_search_profile: str = "v1",
    xgboost_param_grid: Dict[str, list] = None,
    xgboost_inner_cv_splits: int = 3,
    xgboost_day_aligned_cv: bool = True,
) -> Dict[str, XGBoostFoldResult]:
    windows = windows or get_split_windows()
    results = {}
    for w in windows:
        logger.info("Evaluating fold '%s': train %s->%s, val %s->%s (elasticnet_search_profile=%s)",
                    w.name, w.train_start, w.train_end, w.val_start, w.val_end, elasticnet_search_profile)
        results[w.name] = evaluate_fold_with_xgboost(
            df, w, elasticnet_search_profile=elasticnet_search_profile,
            xgboost_param_grid=xgboost_param_grid, xgboost_inner_cv_splits=xgboost_inner_cv_splits,
            xgboost_day_aligned_cv=xgboost_day_aligned_cv,
        )
    return results
