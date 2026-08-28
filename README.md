# European Day-Ahead Power Forecast-to-Trade & Tail-Risk System

**Status: Forecasting and uncertainty layers built, reproduced on corrected data, and verified. 237/237 tests passing.** The A03 parser bug is fixed and confirmed reproducible: `baseline_v1_a03fix` and `xgboost_v1_a03fix` re-verify the original findings on corrected data (XGBoost-full promoted, 17.2% row-weighted improvement over ElasticNet-full, exceeding the original pre-fix margin). Uncertainty quantification is selected and frozen: `uncertainty_selected_v1` (60-day rolling residual window), chosen via a pre-registered sensitivity experiment across `[60, 90, 120, 180, 365]` days, validated on a common-row-corrected comparison, with a Tier-1 robustness check and a mechanical-vs-structural regime diagnostic completed on top of it. EDA is versioned and reproducible going forward (`development_a03fix_v1`), though the pre-A03 EDA snapshot itself was lost to an earlier unversioned-output bug and cannot be recovered. **`auction_sequence == 1` is CONFIRMED** via an independent cross-check against SMARD (Germany's official market data platform) across all 8,833 disagreeing intervals in the corrected data — 100% consistent with sequence 1, 0% with sequence 2 — closing what had been the project's last open target-definition assumption; see "Auction sequence" below for the full result. The standing, still-open caveat is Tier-2 (wind/solar-derived) features' unproven point-in-time availability at the 11:45 D-1 decision cutoff — this is a real, load-bearing limitation on the model's edge, not a formality, and remains unresolved. Next: the economic/strategy layer, not yet started. See Limitations and Process notes for the full history, including several real bugs found and fixed along the way (not deleted from the record).**

## 1. Research Question

> Can publicly available day-ahead fundamentals improve hourly Germany-Luxembourg
> (DE-LU) electricity price forecasts over strong naive benchmarks, and does any
> forecasting improvement translate into economically useful decisions after
> transaction-cost assumptions, forecast uncertainty, and tail risk are considered?

## 2. Market Context

- Market: **DE-LU day-ahead bidding zone** only.
- Sample starts **2019-01-01** (clean post DE/AT bidding-zone-split sample), local delivery-day boundary.
- On **2025-10-01**, EU Single Day-Ahead Coupling moved from hourly to 15-minute
  market time units. This project stays hourly and aggregates post-cutover
  quarter-hour data by averaging (never summing) — see `src/clean.py`.

## What's built so far

| Step | File | Status |
|---|---|---|
| 1. Repo + config | `config.yaml`, `.env.example`, `.gitignore`, `requirements.txt` | done |
| 2. ENTSO-E ingestion | `src/entsoe_client.py` | done — exercised against live DE-LU data; date-range semantics fixed (see below) |
| 3. Time normalization | `src/clean.py` | done — UTC storage, Europe/Berlin local derivation, DST-safe, quarter-hour to hourly aggregation, cutover anchored to local delivery day; explicit range clipping added |
| 4. Final clean dataset | `src/clean.py::build_clean_dataset` | **historical artifact exists, not currently frozen** — the 66,455-row `delu_hourly.parquet` was generated before the A03 parser correction. It must be regenerated from the preserved raw-XML cache after the cache-only CurveType audit confirms scope and structural integrity. |
| 5. Leakage protection (D-1 cutoff) | `src/features.py`, `tests/test_no_leakage.py` | done — decision cutoff, fundamentals, calendar, price lags/rolling, structural leakage guard with explicit provenance (not name-inference) |
| Tests | `tests/` | **full-suite rerun required after cache-provenance audit tests** — last user-confirmed complete run before those additions: 148/148 passed |
| 6. EDA | `run_eda.py` | done — see EDA findings below; solar nighttime-only missingness confirmed against real data. **Measured against data that needs re-ingestion (see curveType bug above) — provisional until reproduced.** |
| 7. Chronological splits | `src/splits.py`, `tests/test_splits.py` | done — locked in before any model exists |
| 8-9. Baselines + ElasticNet | `src/models.py`, `run_models.py`, `tests/test_models.py` | historically frozen — `baseline_v1` accepted (rule applied to `baseline_v2`, retained v1: condition 1 met, condition 2 not met by -0.014%). **Result currently provisional pending corrected re-ingestion and reproduction; not yet re-verified against the fixed parser.** |
| 10. XGBoost | `src/xgboost_model.py`, `run_xgboost.py` | historically frozen — `xgboost_v1` PROMOTED per pre-registered rule (15.6% row-weighted MAE improvement over ElasticNet-full, threshold was 1.0%). Reproduction check passed against `baseline_v1` at the time. **Promotion currently provisional pending corrected re-ingestion and reproduction; not yet re-verified against the fixed parser.** Tier-1/Tier-2 dependency caveat carries over unchanged (worse in fold_1) but also needs reproduction. No further tuning until re-verified. |
| 10-11. Freeze + holdout | `src/splits.py` | not started |
| 12. Uncertainty intervals | `src/uncertainty.py` | not started |
| 13-17. Strategy + P&L | `src/strategy.py` | not started |
| 18-19. Tail risk | `src/risk.py` | not started |

### Real bugs found against live data (kept as a record, not deleted)

Each of these was found by inspecting actual ENTSO-E responses or actual regulation text, not assumed — and each has a regression test built directly from the real finding:

1. **Two parallel price products since 2019** (PT60M standard auction + PT15M quarter-hour auction) — colliding on timestamp with genuinely different prices. Fixed: `entsoe_client.select_price_resolution()`.
2. **2025-10-01 cutover anchored to UTC midnight instead of local delivery day** — misclassified the first two hours of the new regime. Fixed: `clean.local_delivery_date_to_utc()`, used consistently everywhere the cutover is checked.
3. **Two sequential auction runs per interval since 2025-10-01** (`classificationSequence_AttributeInstanceComponent.position` 1 vs 2), different prices, everything else identical. Fixed: `entsoe_client.select_primary_auction_sequence()` keeps sequence 1 — documented as an unverified assumption, not a confirmed fact; see Limitations.
4. **Load/wind/solar forecasts are quarter-hourly for the whole history**, not just post-2025-10-01 like price. Fixed: `clean.aggregate_to_hourly()` (unconditional) vs `aggregate_quarter_hour_to_hourly()` (price-specific, cutover-gated).
5. **DST fall-back day (25 local hours) breaks the "24h lag = previous calendar date" assumption** for ~7 rows/year. Fixed: rows where this can't be proven safe are explicitly set to `NaN`, not silently used.
6. **`save_ingestion_log()` opened the log file in append mode**, so every re-run of `run_ingestion.py` (routine during iterative debugging) added its own records on top of whatever was already there, rather than reflecting only the current run. Confirmed directly from a real uploaded `ingestion_log.jsonl`: 32 price-chunk entries, only 8 distinct — 3 stale copies from an old naive-UTC-boundary version of `get_default_date_range` (all `cache_hit=True`, three separate runs within ~12 hours), plus 1 from the corrected local-delivery-day-boundary version (`cache_hit=False`, a genuinely fresh fetch since the corrected boundaries produce a different cache key). This fully explains the earlier "~3x too many price rows" observation — `diagnostics/inspect_price_xml.py`'s `price_records[0]` was very likely grabbing a stale entry from a superseded run. **This is a logging/metadata bug, not a data-correctness bug**: `fetch_day_ahead_prices()` never reads from this log file, so `delu_hourly.parquet` built by any given `run_ingestion.py` execution always reflects that run's own correctly-computed boundaries regardless of what the log says. Fixed: `save_ingestion_log()` now overwrites (mode `"w"`), matching `delu_hourly.parquet` itself being overwritten wholesale each run rather than accumulated.
6. **Sample date boundaries didn't match what ENTSO-E actually returned.** `start_date: "2019-01-01"` was parsed as naive UTC midnight, but ENTSO-E returns whole local delivery-day periods — the first frozen row was `2018-12-31T23:00Z`, an hour before the declared start. Fixed: `get_default_date_range()` now uses `local_delivery_date_to_utc()` for both boundaries, and `build_clean_dataset()` explicitly clips to `[start_utc, end_utc)` as a final step regardless of what the upstream API returns.
7. **Rolling-feature leakage guard checked the wrong number.** `price_rolling_mean_168h`'s window is 168h, but its newest underlying data point is only ~24h back (it's built from an already-lagged series). The guard was inferring "168" from the column name and checking that — which happened to still pass today, but wouldn't have caught a future regression that removed the internal lag. Fixed: `add_price_lag_and_rolling_features()` now returns explicit provenance (`newest_source_lag_hours`), and `assert_information_set_valid()` requires it for rolling columns rather than guessing from the name.
8. **Missing wind forecast components were `fillna(0)`'d**, which isn't the same as an actual 0 MW forecast and could artificially inflate residual load. Fixed: both onshore and offshore must be present or the combined figure is `NaN` (`pandas.sum(min_count=2)`).
9. **`pivot_table(..., aggfunc="mean")` on wind/solar could silently average genuine duplicate (timestamp, psr_type) submissions into a value that never existed** — the same class of mistake already found and fixed for price. Fixed: explicit duplicate check raises before pivoting, rather than smoothing over it.

### Point-in-time gap found via the actual EU Transparency Regulation (important, not yet fully resolved)

Checked directly against **Commission Regulation (EU) No 543/2013**:

- **Article 6(2)(b):** day-ahead load forecast must be published no later than 2 hours before day-ahead gate closure (~noon D-1) — provably before our 11:45 D-1 decision cutoff. Load forecast is **Tier 1**: publication timing is supported by the regulatory deadline and is compatible with the simulated cutoff — not "individually vintage-verified," since exact historical revision vintage still can't be reconstructed for any Tier (see the standing limitation below).
- **Article 14(2)(d):** day-ahead wind/solar generation forecast must be published no later than 5 p.m., one day before delivery, with further updates through intraday trading — this deadline is after our 11:45 D-1 cutoff, not before it. Wind/solar is **Tier 2**: even the regulatory publication deadline does not establish availability by the simulated cutoff.

This means the blanket claim "day-ahead forecasts are safe because they're published before the auction" was true for load's *deadline* but not for wind/solar's. `src/features.py::FEATURE_AVAILABILITY_TIER` now tags every fundamentals-derived column with its tier. Wind, solar, and everything derived from them (residual load, renewable share) are Tier 2 — this doesn't mean they're leaking, it means we cannot currently prove they aren't, since ENTSO-E's historical API doesn't expose publication-vintage timestamps. This is a real, open methodological question, documented rather than hidden, and now backed by an actual empirical robustness test (`ELASTICNET_TIER1_PREDICTOR_COLS`, evaluated alongside the full model — see Baseline results below), not just a caveat.

## Pre-Registered Modelling Hypotheses (written before seeing any model results)

These are written down before `src/models.py` exists, based purely on the EDA findings above — so there's evidence they weren't invented after the fact to fit whatever the results turn out to be.

- **H1:** Lag-24 will be a strong benchmark — the target has pronounced intraday structure (fig3: ~€60/MWh average peak-to-trough swing, morning + evening double peak).
- **H2:** Lag-168 may match weekday/weekend patterns well but adapt more slowly to changing price regimes than lag-24.
- **H3:** Fundamentals (residual load especially) should add value beyond the naive lags — within-year Pearson/Spearman correlations of 0.58–0.88 are strong, even though the pooled correlation (0.45) looks only moderate because it's diluted by year-to-year level shifts.
- **H4:** XGBoost may outperform ElasticNet specifically at very low and very high residual load, where fig4 shows the relationship is nonlinear (convex at the top end) rather than a straight line.
- **H5:** Forecast errors should increase during extreme-price periods (>€200, >€500) even if average MAE looks good — the target has a long right tail (max €936.28, 0.87% of hours above €500) and a real negative-price floor (-€500, hit exactly at EPEX's actual auction floor).
- **H6:** Post-2025-10-01 performance may differ from earlier folds — the development sample has only ~2,209 rows (about 3 months) under the new 15-minute-MTU market design, while the *entire* 2026+ holdout is under that same new regime. This is exactly why `src/splits.py` includes a dedicated `regime_stress_test` fold (train on everything through Sep 2025, validate on Oct–Dec 2025) rather than only annual folds.

## Chronological Validation Architecture (`src/splits.py`)

Built and tested **before** `src/models.py`, so no baseline or model can shape how the evaluation framework itself works. No shuffling, ever — every split is checked structurally (`assert_split_is_chronological`), not just by construction.

```
Fold 1:              train 2019 → 2023-01-01,  validate 2023
Fold 2:              train 2019 → 2024-01-01,  validate 2024
Fold 3:              train 2019 → 2025-01-01,  validate Jan–Sep 2025
Regime stress test:  train 2019 → 2025-10-01,  validate Oct–Dec 2025

Final model:   train on all of 2019 → 2026-01-01
Final holdout: 2026-01-01 → latest complete month (untouched until the model is frozen)
```

## EDA Findings (development sample, 2019–2025, holdout excluded)

- **Target is strongly non-Gaussian and regime-changing.** Mean €94.86/MWh vs median €74.15/MWh (mean >> median: heavy right skew), std €93.81/MWh, range -€500 to €936.28. 3.31% of hours negative, 0.87% above €500. Confirms MAE (not MAPE — prices can be zero/negative) as the primary metric, RMSE as secondary (spikes matter), and justifies separate extreme-regime reporting later.
- **2022 was a completely different price regime** (mean €235.56/MWh vs €30–38/MWh in 2019–2020, a 6–8x shift) — but extremes aren't confined to 2022: the single highest observation (€936.28) is from December 2024, and the -€500 floor was hit in July 2023. The elevated *level* receded after 2022; extreme *events* didn't.
- **Residual load is a genuinely strong price driver, but regime-dependent.** Pooled Pearson/Spearman = 0.45/0.50, but every individual year is 0.58–0.88 — the pooled figure is diluted by year-to-year price-level shifts, not a weak relationship. The relationship is also visibly nonlinear (convex at high residual load — fig4), which is the concrete justification for trying XGBoost rather than stopping at ElasticNet.
- **Strong, non-trivial intraday and weekly structure.** ~€60/MWh average peak-to-trough per day (trough ~13:00, peak ~19:00, double-peaked morning+evening); weekday mean €106 vs Sunday €69 (~35% lower). Weekday and weekend intraday *shapes* differ, not just levels — motivated the `weekend_hour_sin/cos` interaction terms.
- **Negative residual load happens** (478 hours, 0.78% — forecast renewables exceeding forecast load), concentrated at the low end of the price distribution (near-zero median price at the lowest residual-load ventile). Real market mechanism, not a data artifact.
- **Solar's 40.5% missingness confirmed nighttime-only** against real data: zero missing rows in local hours 8–16, missingness concentrated in hours 0–7 and 17–23 with small transition counts at hours 7 and 17 (dawn/dusk). Validates the `fillna(0)` treatment in `add_fundamentals_features`.
- Full figures and tables: `outputs/eda/delu_features/development/`.

## Baseline & ElasticNet Results (steps 8-9)

**Four methodological corrections were made after reviewing the pipeline and early real-data results:**

1. **Common evaluation mask** — the first run scored each model on its own NaN-pattern-dependent row subset (slightly different `n` per model). Fixed: `evaluate_fold()` now computes one common mask (target + every model's prediction present) and scores all models on the *identical* validation rows per fold. `run_models.py`'s coverage report shows raw/per-model/common row counts explicitly, not just asserts they're equal.
2. **Scaler-before-inner-CV leakage.** The second run fit `StandardScaler` on the full outer-training set before `TimeSeriesSplit` divided it for hyperparameter selection — meaning each inner fold's scaling statistics could reflect data from later in the outer training set than that inner fold's own cutoff. Fixed: `fit_elasticnet()` now uses `Pipeline(scaler, elasticnet)` inside `GridSearchCV`, so the scaler is refit separately within every individual inner training fold. Hyperparameter selection is also now scored on MAE (`scoring="neg_mean_absolute_error"`), matching the project's declared primary metric, instead of the default MSE.
3. **Row-order assumption in `TimeSeriesSplit`.** `TimeSeriesSplit` splits by row *position*, not timestamp value — it has no idea what a timestamp even is. `fit_elasticnet()` previously relied on the incidental fact that the input happened to already be sorted. Fixed: now explicitly sorts by `timestamp_utc`, rejects duplicate timestamps, and asserts monotonicity before handing rows to the inner CV — proven by a test that fits on a deliberately shuffled copy of the same data and checks the selected hyperparameters are identical either way.
4. **Holdout boundary test used a naive UTC literal** (`2026-08-01T00:00:00Z`) instead of the canonical `local_delivery_date_to_utc("2026-08-01")` — a 2-hour discrepancy that would have silently included the first two local hours of August in the "holdout," confirmed against the real EDA manifest's `holdout_rows_excluded: 5087` (the UTC literal implies 5,089).

**Also added:** a Tier-1-only ElasticNet variant (`ELASTICNET_TIER1_PREDICTOR_COLS` — load forecast + price state + calendar, excluding the wind/solar-derived Tier-2 columns) evaluated alongside the full model, so the Tier-2 point-in-time gap becomes an actual empirical robustness test rather than just a caveat. Fixed, fold-independent stress buckets (`price < 0`, `> 200`, `> 500` EUR/MWh) are saved alongside the training-quantile-based extreme flag. Per-fold coefficients (both variants), per-fold hyperparameters (`{fold}_elasticnet_hyperparams.json`), per-row predictions, and a run manifest (sklearn/pandas versions, hyperparameter-selection method) are saved to `outputs/models/<input_stem>/<run_version>/`. This directory is now **immutable by construction** — `run_models.py` raises `FileExistsError` rather than overwriting if a `run_version` directory already has results in it; use a new version name (e.g. `baseline_v2`) for a fresh run.

**The numbers from the earlier pre-correction runs are no longer reported here.** They were provisional and are superseded by the corrected evaluation architecture described above.

### `baseline_v1` results (real data, corrected methodology, all four models on identical common rows per fold)

| Fold | lag-24 MAE | lag-168 MAE | ElasticNet-full MAE | ElasticNet-Tier1 MAE | Full vs lag-24 | Tier1 vs lag-24 |
|---|---:|---:|---:|---:|---:|---:|
| fold_1 (val 2023) | 27.15 | 33.45 | 19.51 | 22.94 | -28.1% | -15.5% |
| fold_2 (val 2024) | 27.66 | 32.54 | 20.90 | 24.55 | -24.4% | -11.2% |
| fold_3 (val Jan-Sep 2025) | 25.98 | 32.13 | 18.81 | 23.10 | -27.6% | -11.1% |
| regime_stress_test (val Oct-Dec 2025) | 25.76 | 34.72 | 20.03 | 24.44 | -22.2% | -5.1% |

**Row-weighted across all 25,790 common out-of-sample observations** (fold sizes 8627/8547/6409/2207): lag-168 = 32.93, lag-24 = 26.91, ElasticNet-Tier1 = 23.64, ElasticNet-full = 19.84 — a **26.3% weighted MAE reduction over lag-24** and **39.8% over lag-168** for the full model.

**H1 confirmed:** lag-24 beats lag-168 in every fold.

**H2 partially supported / consistent with the evidence, not cleanly rejected:** lag-168 underperformed lag-24 in every fold, consistent with the "adapts more slowly to changing price regimes" component of H2. But the separate claim in H2 — that lag-168 might match weekday/weekend structure well — was never independently tested by these aggregate fold metrics; poorer overall MAE doesn't address that part of the hypothesis one way or the other. Calling this "rejected" (as an earlier version of this README did) overstated what the aggregate numbers actually show.

**H3 confirmed, with corrected (smaller but still solid and consistent) magnitudes:** ElasticNet-full beats lag-24 by 22–28% MAE in every fold, including the regime-stress test. The earlier (buggy) run's 27–36% figures overstated this somewhat — the corrected numbers are more modest but the direction and consistency hold up.

**Most important finding — the Tier-2 dependency GROWS in the newest regime, not shrinks:**

| Fold | Tier-1 retains this % of Full model's advantage over lag-24 |
|---|---:|
| fold_1 | 55% |
| fold_2 | 46% |
| fold_3 | 40% |
| regime_stress_test | **23%** |

Tier-1 (load forecast + price state + calendar only) retains less than a quarter of the full model's advantage in the one fold that's actually under the new 2025-10-01 market design — the same regime the entire 2026 holdout lives in. Tier-1's regulatory publication timing is compatible with the 11:45 D-1 cutoff (see Point-in-time gap above), though — as with everything in this dataset — exact historical revision vintage remains unreconstructed, so "compatible" is not the same claim as "individually vintage-verified." This is the opposite of reassuring: the model's edge increasingly depends on the wind/solar-derived Tier-2 features precisely where we can least prove their point-in-time availability. This is a real, load-bearing limitation, not a footnote — worth stating plainly before any holdout evaluation.

**The regime-stress fold's advantage is concentrated in the hard hours, not the typical ones.** In `regime_stress_test`, ElasticNet-full clearly wins on MAE (20.03 vs 25.76) and RMSE (28.62 vs 40.25), but its **median absolute error is actually slightly worse than lag-24** (15.07 vs 14.80). Combined with the fixed-bucket results below (large wins in `negative_price` and `price_gt_200`), this means the full model's Q4-2025 advantage comes disproportionately from getting the unusual, economically consequential hours right — not from shaving a bit off every ordinary hour. That's arguably a more useful property for a trading application than a uniformly lower median error would be, but it's a real behavioral difference from the earlier folds and worth naming explicitly rather than letting the single MAE number imply uniform improvement.

**A prior finding does not replicate under the corrected methodology.** The earlier (pre-fix) run suggested lag-24 was competitive with or beat ElasticNet specifically in the extreme-price regime, in 2 of 4 folds. Under the corrected common-mask/tuned-hyperparameter evaluation, **ElasticNet-full now wins the `negative_price` and `price_gt_200` fixed buckets in every single fold** — a reversal, by roughly 27–46% MAE (negative price) and 17–29% MAE (>€200). The one exception: `price_gt_500` (n=1, 15, 1, 0 across the four folds) — **results above €500/MWh are descriptive only; event counts are too small for stable comparative inference**, and should not be read as a pattern (lag-24 happens to win the one fold with a usable sample size, fold_2 at n=15, but that single comparison cannot establish anything general). The earlier "extreme-regime weakness" was very likely an artifact of the bugs that have since been fixed, not a genuine property of the linear model.

**Grid-boundary robustness check.** The minimum alpha (`0.001`, the exact edge of `baseline_v1`'s 15-value search grid) was selected in 6 of 8 model fits; the minimum `l1_ratio` (`0.1`) was selected in 3 of 8 fits (fold_2 full, stress full, stress Tier-1). Both dimensions showing boundary evidence is why both are extended below, not just alpha.

**Decision made before seeing any `baseline_v2` results:** the v2 search profile (`BASELINE_V2_ALPHA_GRID`, `src/models.py::ELASTICNET_SEARCH_PROFILES["v2"]`) extends v1's original 15-value grid (`BASELINE_V1_ALPHA_GRID`) as a genuine *union* (`{1e-5, 3e-5, 1e-4, 3e-4} ∪ BASELINE_V1_ALPHA_GRID`, verified by a regression test) plus `l1_ratio=0.01` — not replaced with a different discretization, so the experiment cleanly answers "does allowing smaller alpha/l1_ratio help" rather than confounding that with "does a different grid help." Both profiles are named, fixed module constants (`ELASTICNET_SEARCH_PROFILES["v1"]`/`["v2"]`), not a single mutable default — `run_models.py` now takes an explicit `search_profile` CLI argument and records it (plus the exact grid values used) in the run manifest, so which experiment produced a given result is a checkable fact, not inferred from a folder name. `baseline_v1` is preserved as-is, not overwritten, consistent with the immutable-evidence design; this check runs as `baseline_v2`.

**Acceptance rule, fixed before running `baseline_v2`:**
```
Accept baseline_v2 as the new frozen benchmark only if BOTH:
  1. ElasticNet-full selects at least one newly-available grid value
     (alpha < 0.001, or l1_ratio = 0.01) in at least one development fold, AND
  2. Row-weighted ElasticNet-full development MAE improves by >= 0.5%
     versus baseline_v1 ElasticNet-full.

Evaluated on the FULL model specifically (the primary forecasting
benchmark) -- Tier-1's grid selection or MAE change does not by itself
determine promotion, since Tier-1 is the robustness check, not the
benchmark being tuned.

Otherwise: retain baseline_v1. No baseline_v3 — the grid-boundary
question is answered either way once v2 runs.
```

### `baseline_v2` ran — rule applied, `baseline_v1` retained

| Condition | Result |
|---|---|
| 1. Newly-available grid value selected by ElasticNet-full in ≥1 fold | **Met** — `alpha=1e-05` (below v1's 0.001 minimum) in fold_1/2/3, `l1_ratio=0.01` (below v1's 0.1 minimum) in the regime-stress fold |
| 2. Row-weighted ElasticNet-full MAE improves ≥0.5% vs v1 | **Not met** — v1: 19.8407, v2: 19.8436 (v2 is **0.014% worse**, not better) |

**Decision: `baseline_v1` retained as the frozen linear benchmark. No `baseline_v2` promotion. No `baseline_v3`.**

This is the rule working as intended: condition 1 passed easily — the search genuinely wanted weaker regularization everywhere it could get it — but the practical effect was negligible, changing row-weighted MAE by about €0.003/MWh, well inside noise. The `baseline_v1` grid-boundary hit was real, but it turned out not to matter: ElasticNet was already close enough to unregularized (near-OLS) behavior that extending the search further doesn't change the answer. A useful negative result, not a wasted check — and exactly the kind of finding this project's "What Failed" philosophy exists to record rather than bury.

**Secondary finding, documented rather than acted on:** the `v2` grid search produced numerous `ConvergenceWarning`s from `sklearn`'s coordinate descent solver, concentrated at the new low-alpha candidates (`alpha=1e-5` is ~100x smaller than v1's minimum). This has a mundane, expected explanation — near-zero regularization combined with the already-documented collinearity among `residual_load_forecast_mw`/`renewables_forecast_mw`/`renewable_share_forecast` is a textbook slow-convergence case for coordinate descent, not evidence of a correctness bug. **Deliberately not addressed by raising `max_iter` or re-tuning the grid**: doing so now, after seeing an unwelcome/noisy-looking result, would be exactly the kind of post-hoc rationalized tuning the pre-registered "no `baseline_v3`" stopping rule exists to prevent. The warnings don't change the decision either way — `baseline_v2` already loses decisively on the pre-registered MAE criterion regardless of whether every candidate fully converged.

**The linear-modelling stage (steps 8–9) is now closed.** `baseline_v1` is the accepted ElasticNet benchmark going forward; no further ElasticNet code or hyperparameter changes. `baseline_v2/` is preserved on disk, not deleted — it's evidence the robustness question was actually tested and that the pre-registered stopping rule was followed, not skipped.

```
baseline_v1: ACCEPTED AND FROZEN
baseline_v2: COMPLETED, NOT PROMOTED
ElasticNet development: CLOSED
```

**Important caveat on what this rule can and can't establish:** both `baseline_v1` and `baseline_v2` were compared using the same 2019–2025 development folds that were also used to design the feature set and evaluation architecture. Choosing between v1 and v2 based on development-fold MAE is a legitimate, disciplined way to close out a specific, pre-identified question (the grid-boundary check) — but the resulting number is not an untouched estimate of "how good is model selection here." That's what the 2026 holdout is for, and it stays untouched through this decision, through XGBoost development, and until the entire forecasting system (including this model-selection step) is frozen.

**Coefficients are economically sensible and show a real regime shift in Q4 2025.** `price_lag_24h` is by far the dominant standardized predictor in every fold, but its magnitude *decreases* steadily from fold_1 (+61.2) through the regime-stress fold (+27.0) as regularization strengthens — consistent with the model relying comparatively more on fundamentals and less on pure persistence in the newest regime. `renewables_forecast_mw` carries a negative sign wherever it appears in the top 5 (more forecast renewables → lower price, consistent with merit-order economics); `residual_load_forecast_mw` carries a positive sign wherever it appears (more residual load → higher price, and it enters the regime-stress fold's top 5 specifically). Which of the two shows up in a given fold's top 5 varies — expected collinearity between algebraically related predictors (see the standing coefficient-interpretation caveat below), not an inconsistency.

Full per-hour, per-regime, per-fold coefficient, hyperparameter, and per-row prediction tables: `outputs/models/delu_features/baseline_v1/` (and `baseline_v2/`, preserved as evidence the check was run).

## XGBoost (step 10)

Built following the same discipline as the ElasticNet stage:

- **Predictor sets are deliberately identical to ElasticNet's** (`XGBOOST_PREDICTOR_COLS = ELASTICNET_PREDICTOR_COLS`, `XGBOOST_TIER1_PREDICTOR_COLS = ELASTICNET_TIER1_PREDICTOR_COLS`, imported not redefined), so performance differences cannot be attributed to XGBoost receiving additional information. **The comparison nevertheless includes each model's own frozen model-selection procedure, not model family alone**: ElasticNet retains its previously frozen hourly `TimeSeriesSplit` inner CV, while XGBoost uses delivery-day-aligned inner CV introduced before its first real-data run (see below). The two models have identical information sets and identical outer evaluation, but different inner tuning protocols — "XGBoost beat ElasticNet by X%" should be read as "the XGBoost pipeline, as specified, beat the ElasticNet pipeline, as specified," not as an isolated causal estimate of nonlinearity alone. Raw (non-cyclic) calendar columns aren't included yet; trees don't strictly need cyclic encoding, but adding them would be a separate, deliberate follow-up experiment, not silently baked in here.
- **Small, pre-specified hyperparameter grid** matching `config.yaml::models.xgboost_param_grid` exactly (max_depth × n_estimators × learning_rate × subsample = 36 combinations) — frozen there before any model was built, per spec section 9's "tune only a small parameter space." `objective="reg:absoluteerror"` and `tree_method="hist"` are both set explicitly (not left to XGBoost's automatic choice) but are not additional tuned hyperparameters.
- **Chronological discipline enforced, not assumed**: `fit_xgboost()` sorts by timestamp, rejects duplicates, asserts monotonicity before any CV splitter sees the rows — identical guard to `fit_elasticnet()`.
- **All six models — lag-24, lag-168, ElasticNet-full, ElasticNet-Tier1, XGBoost-full, XGBoost-Tier1 — evaluated on one identical common-row set per fold**, via `evaluate_fold_with_xgboost()`. ElasticNet uses the **accepted `baseline_v1` profile** by default (`elasticnet_search_profile="v1"`) — that decision is closed, not reopened here.
- **The canonical experiment name `xgboost_v1` is structurally locked to ElasticNet profile `v1`** (`CANONICAL_XGBOOST_RUN_PROFILES` in `run_xgboost.py`, same class of guard as `run_models.py`'s `baseline_v1`→`v1` lock). A directory literally named `xgboost_v1` cannot silently compare against the rejected ElasticNet-v2 profile — passing `v2` for that run_version raises, not just warns.
- **`baseline_v1` reproduction check runs before anything is written.** `run_xgboost.py` refits ElasticNet rather than reading old prediction files (methodologically fine since v1 is frozen, fitting is deterministic, and predictor sets are unchanged) — but since `baseline_v1` is supposed to be immutable evidence, `verify_baseline_v1_reproduction()` compares each fold's re-fitted ElasticNet-full MAE against the documented `baseline_v1` numbers (0.1% relative tolerance — tightened from an initial 0.3%, since the promotion threshold itself is only 1.0% and a loose comparator tolerance would eat into that decision margin) and raises `RuntimeError` *before* writing any output if they don't match. Protects against future code drift silently invalidating the whole comparison.
- **XGBoost's inner CV is aligned to Europe/Berlin delivery dates, not arbitrary hourly rows** (`make_delivery_day_cv`). The simulated decision point is D-1 11:45 — at that moment, all hours of delivery day D are being forecast at once, with none of D's realized prices available yet. Plain hourly `TimeSeriesSplit` can place different hours of the same delivery day on opposite sides of an inner-CV boundary. **This is not feature leakage** (no predictor encodes same-day identity, and every inner-train row is still chronologically before every inner-val row — no feature ever sees a later timestamp's information) **and it does not contaminate outer validation** (no outer row is ever used for inner training or tuning). But it is a real, distinct problem: the fitted model's *parameters* are a channel from earlier hours of D into predictions for later hours of D — if an inner-training fold contains realized targets from `D 00:00–10:00` and inner-validation starts at `D 11:00`, those earlier-in-day outcomes have already influenced the fitted coefficients/tree structure, even though at the real D-1 11:45 forecast origin none of those D targets existed yet. This is best described as an **inner-CV forecast-origin alignment problem**, potentially amplified by within-day residual dependence (correlated same-day residuals from unmodeled shocks — weather, an unplanned outage — make hyperparameter selection mildly optimistic), rather than as "leakage" in the feature-contamination sense. `make_delivery_day_cv` groups by local calendar date first, then applies `TimeSeriesSplit` to the *dates*, so a 23-hour spring-DST day or a 25-hour autumn-DST day is still exactly one indivisible unit regardless of its actual row count (verified by dedicated tests, including the real 2024-03-31 and 2024-10-27 DST transition dates).

  **Deliberately not retrofitted onto ElasticNet's frozen `baseline_v1`/`baseline_v2`.** Delivery-day alignment changes only the inner model-selection procedure and never gives ElasticNet access to outer-validation observations — but it *could* select different hyperparameters and therefore indirectly change outer-validation predictions and MAE if retrofitted (no data contamination is not the same claim as no effect on results). Because `baseline_v1` and `baseline_v2` were already frozen under a documented hourly-CV methodology — and `verify_baseline_v1_reproduction()` exists specifically to detect exactly this kind of drift — they are preserved rather than retrospectively redefined. The prior `baseline_v2` grid-extension experiment suggests limited sensitivity to further weakening regularization; it does **not** directly test sensitivity to delivery-day-vs-hourly CV, a different axis of variation entirely, and is cited here only as weak, indirect context. XGBoost, not yet run, gets the more rigorous design from the start — that's finishing the build, not reopening a decision.
- Coverage report, both stress-bucket definitions (fixed + training-quantile), ElasticNet coefficients, and XGBoost feature importances all saved, same pattern as `run_models.py`. The run manifest records `xgboost_day_aligned_cv`, `xgboost_cv_timezone`, and `xgboost_cv_unit` explicitly, alongside a note on `elasticnet_hyperparameter_selection` documenting the deliberate hourly/day-aligned asymmetry.
- Output scoped and immutable: `outputs/models/<input_stem>/<run_version>/`, `FileExistsError` if the directory already has results.

**21 tests passing** (`tests/test_xgboost_model.py`, `tests/test_run_xgboost.py`), including a regression guard that the predictor sets stay identical to ElasticNet's, a check that the param grid matches `config.yaml` by actually loading it (not comparing against a second hardcoded copy), the canonical-name lock (`xgboost_v1` + `v2` → rejected), delivery-day CV correctness (no date split across train/val, spring/autumn DST days verified to stay whole), and the same shuffled-input/duplicate-timestamp guards proven for ElasticNet.

### Promotion rule, pre-registered before the first real-data run

```
Promote XGBoost-full to the primary point-forecast model only if its
row-weighted development MAE improves by >= 1.0% versus frozen
ElasticNet-full baseline_v1 (19.8407). That means XGBoost-full must
achieve approximately 19.6423 or lower.

Stress-period, RMSE, and extreme-bucket results are important
diagnostics but do NOT override the primary MAE rule.

Tier-1 XGBoost is a point-in-time robustness diagnostic, same role as
Tier-1 ElasticNet -- it does not itself determine promotion.

No XGBoost retuning after seeing xgboost_v1 results.
```

`run_xgboost.py` applies this rule automatically and writes the decision to `promotion_rule_result.json` alongside the run.

**Feature-importance caveat (stated before results arrive, not after):** XGBoost's saved `feature_importances_` are gain-based descriptive model attribution, not causal or economically independent contribution claims. The Full predictor set contains algebraically related variables (`load_forecast_mw`, `renewables_forecast_mw`, `residual_load_forecast_mw`, `renewable_share_forecast`) — tree importance, like ElasticNet's standardized coefficients, can be redistributed among correlated predictors. Report as "XGBoost gain-based feature importance ranks X highest," not "X drives Y% of the prediction."

### `xgboost_v1` ran — rule applied, XGBoost-full PROMOTED

**Reproduction check passed** — re-fitted ElasticNet-full matched the frozen `baseline_v1` numbers to 6 decimal places (deterministic fit, same code, no drift).

| Fold | lag-24 MAE | ElasticNet-full MAE | XGBoost-full MAE |
|---|---:|---:|---:|
| fold_1 (2023) | 27.15 | 19.51 | **17.06** |
| fold_2 (2024) | 27.66 | 20.90 | **16.59** |
| fold_3 (Jan-Sep 2025) | 25.98 | 18.81 | **16.88** |
| regime_stress_test (Oct-Dec 2025) | 25.76 | 20.03 | **15.72** |

**Row-weighted: ElasticNet-full = 19.8407, XGBoost-full = 16.7442 — a 15.607% improvement, decisively above the 1.0% threshold.**

```
Condition 1: reproduction check passes           PASSED
Condition 2: row-weighted MAE improves >= 1.0%    15.607% >= 1.0% -- MET

DECISION: PROMOTE_XGBOOST_FULL
```

**H4 is strongly supported at the model level, but the specific residual-load-extreme mechanism has not been directly tested.** H4 originally predicted XGBoost's advantage would be greatest at *very low and very high residual load*. The stress buckets actually measured here are stratified by *price regime* (negative price, >€200, >€500) — economically related to residual-load extremes, but not the same variable, and no table here bins error against residual-load quantiles directly. What's genuinely confirmed: the nonlinear XGBoost pipeline decisively outperforms ElasticNet, including during negative-price and other difficult regimes — improvement over ElasticNet-full in the `negative_price` bucket ranges 2.6%–63.8% across folds, largest in `regime_stress_test` (17.59 → 6.37 MAE), and `residual_load_forecast_mw` is consistently highly ranked in the Full model's gain-based importance (third overall in `regime_stress_test`, behind only the two price lags) — consistent with, but not direct confirmation of, the proposed mechanism.

**The strongest single finding may be Q4 2025's median-hour result, not the negative-price extremes.** ElasticNet-full's `regime_stress_test` improvement was tail-heavy — its median AE (15.07) never actually beat lag-24's (14.80). XGBoost-full's median AE there is **10.78**, a genuine, broad improvement: 21.5% better than ElasticNet-full, 39.0% better than lag-24 (both verified independently). This matters more than it might first appear, because the entire untouched 2026 holdout sits in this same post-October-2025 regime — a model that only wins on hard hours in this regime would be a weaker foundation for the holdout than one that wins broadly.

**The Tier-1/Tier-2 dependency problem does not improve with XGBoost — in the earliest fold it's worse.**

| Fold | XGBoost Tier-1 retention | ElasticNet Tier-1 retention |
|---|---:|---:|
| fold_1 | 35.4% | 55.2% |
| fold_2 | 43.5% | 46.0% |
| fold_3 | 41.5% | 40.1% |
| regime_stress_test | 25.8% | 23.0% |

XGBoost-full's larger accuracy gain leans just as heavily — sometimes more heavily (fold_1) — on the wind/solar-derived Tier-2 features whose point-in-time availability at the 11:45 D-1 cutoff cannot be proven. Promoting XGBoost does not make this open question smaller; it inherits it unchanged.

**Hyperparameter boundary noted, deliberately not acted on.** `learning_rate=0.1` (the grid's maximum) was selected in all 4 folds, and `n_estimators=400` (also the grid maximum) in 3 of 4 — real boundary evidence, same class of observation as `baseline_v1`'s alpha hitting its grid edge. Unlike the ElasticNet case, this is **not being investigated with a `xgboost_v2` grid extension**: the pre-registered question was whether this specific, frozen challenger materially beat the accepted linear benchmark, and it did by more than 15x the required margin. Searching higher learning rates, more trees, or deeper trees now would add methodological risk (chasing a strong result to make it marginally stronger) for essentially no scientific value, and would violate the same "no retuning after seeing results" discipline applied throughout this project.

**No XGBoost retuning after this result, per the pre-registered rule.** `xgboost_v1` is the accepted challenger. Full per-hour, per-regime, per-fold predictions, coefficients, and feature importances: `outputs/models/delu_features/xgboost_v1/`.

## Setup

```bash
git clone <this-repo>
cd european-power-forecast-trade
pip install -r requirements.txt

# Register a free token at https://transparency.entsoe.eu/
cp .env.example .env
# edit .env and set ENTSOE_TOKEN=your-token
export ENTSOE_TOKEN=your-token   # PowerShell: $env:ENTSOE_TOKEN = "your-token"

python -m pytest tests/ -v        # rerun after audit-test additions; record the actual collected/passed count
python scripts/audit_price_curve_types.py   # CACHE-ONLY whole-history audit; no ENTSO-E token/network needed
# Only after the audit manifest says complete=true with zero missing/failed cache chunks:
python run_ingestion.py           # reparse the preserved raw XML cache with the corrected A03 decoder
python run_features.py            # build the feature matrix
python run_eda.py                 # EDA figures + tables
python run_models.py delu_features.parquet baseline_v1 v1   # the accepted, frozen ElasticNet benchmark
python run_xgboost.py delu_features.parquet xgboost_v1       # adds XGBoost full + Tier-1; all six models on one common-row set
# baseline_v2 was a completed sensitivity check (grid-boundary robustness) -- rejected by the
# pre-registered rule, preserved on disk at outputs/models/delu_features/baseline_v2/, not rerun here.
```

## Point-in-Time Information Set

Decision cutoff is fixed at 11:45 Europe/Berlin on D-1, before the day-ahead
auction for delivery day D clears (`src/features.py::decision_cutoff_utc`, DST-safe).

Three different safety arguments apply, checked structurally where possible —
not just assumed from how the code was written:

- **Load forecast at (D, h):** Tier 1 — publication timing supported by the regulatory deadline (EU Reg. 543/2013 Art. 6(2)(b), published >=2h before gate closure) and compatible with our 11:45 cutoff. Not "individually vintage-verified" — see the standing limitation on forecast revision vintages below.
- **Wind/solar forecast at (D, h), and everything derived from them:** Tier 2 — used as features, but even the regulatory deadline doesn't establish availability by the cutoff (Art. 14(2)(d) only requires publication by 17:00 D-1).
- **Price at (D', h) for D' < D:** safe once D' is a calendar date strictly before D, regardless of hour — day-ahead price for D' is fixed and published on D'-1, before D' has even started. A >=24h lag always crosses into an earlier calendar date; checked directly via explicit construction provenance, not inferred from column names.
- **Price at (D, h) itself:** never a feature — it's the target.

Realised load and realised generation are never ingested at all (only day-ahead
forecasts — see `entsoe_client.fetch_load_forecast` / `fetch_wind_solar_forecast`),
so that leakage vector is closed structurally.

## Data Quality and Leakage Controls

- `src/clean.py::dedupe_timestamps` rejects a suspiciously high duplicate rate rather than silently dropping it (catches missing upstream selection logic, as happened with the price-product collision).
- `src/clean.py::clip_to_range` explicitly enforces the declared sample boundary as a final step, regardless of what the upstream API returns.
- `src/entsoe_client.py` logs full ingestion metadata to `data/raw/ingestion_log.jsonl` for every request.
- Two DST-transition tests confirm the local-time derivation produces a 23-hour spring day and a 25-hour autumn day without losing or duplicating any underlying UTC rows.
- `src/features.py::assert_information_set_valid` requires explicit construction provenance for rolling features rather than inferring safety from a column name.
- **Limitation (documented up front):** even for Tier 1 (load), we don't have historical forecast revision vintages, only ENTSO-E's current publication. Where a vintage can't be reconstructed exactly, the published forecast is used as a forecast-time proxy.

## Limitations

- Historical backtest is not live trading.
- No market impact, bid acceptance, or order-book dynamics are modelled.
- Gas, EUA prices, interconnector constraints, and outages are excluded from v1.
- Forecast vintages are a proxy, not a guaranteed reconstruction of what a trader actually saw before the D-1 cutoff.
- **CONFIRMED AND FIXED: a real target-construction bug in `entsoe_client.py::_parse_timeseries()` was silently undercounting real published prices as missing data.** Post-2025-10-01 DE-LU day-ahead prices use ENTSO-E's `curveType=A03` ("variable sized blocks"), verified directly against ENTSO-E's own official specification ("The Introduction of Different Time Series Possibilities (CurveType) within ENTSO-E Electronic Documents," v1.4, section 4.3): only positions where the price *changes* are published as explicit XML `<Point>` elements; an unpublished position means "same value as the previous block," not "missing." The parser fix (`_expand_period_points`) reconstructs the full quarter-hourly series by forward-filling, verified against ENTSO-E's own worked numerical example from the spec as a golden test case, and distinguishes this from genuine gaps (which the spec represents as temporally disjoint `Period` elements, not an omitted position within one continuous `Period`). Also fails closed (raises `ValueError`) rather than silently guessing on: a missing A01 position, a missing A03 position 1 (an earlier version of the fix still fabricated a false "forward-filled" value here), duplicate Point positions, out-of-range positions, Points with no discoverable value, and Periods with non-positive or non-resolution-divisible duration.

  **This makes the `auction_sequence` audit's headline numbers unreliable — not necessarily inflated in a known direction.** The old audit's disagreement statistic was computed only over rows where both sequences had an explicit value; A03 under-parsing removed some rows from that comparison entirely rather than biasing it in a predictable direction. The sequence-2-only count (56) is more plausibly inflated by the bug specifically (a position missing only because it wasn't a change-point would show up as "sequence 1 absent"), but the overall disagreement rate (96.8%) could move up, down, or barely change once recomputed. Both figures — and `manual_epex_verification_template.csv`'s discriminating-interval sample — must be re-measured on re-ingested data, not adjusted by assumption.

  **CONFIRMED via the full historical structural audit (`scripts/audit_price_curve_types.py`, cache-only, 8/8 production chunks found and parsed, `complete: true`, zero missing/failed chunks): `curveType=A03` is used throughout the ENTIRE price history, not just post-cutover.** Pre-2025-10-01 PT60M data was also affected by the undercounting bug — 266 of 59,159 effective PT60M intervals (~0.45%) were reconstructed (forward-filled) rather than explicit XML publications, spread across every year 2019–2025. The affected fraction is small, but since these prices feed lag-24/48/168 and rolling-price features, the effect can propagate beyond the 266 directly-affected rows — the feature matrix and all model results still require reproduction. **Also confirmed: zero structural gaps or overlaps anywhere in the audited history** (`disjoint_gap_count`/`overlapping_period_count` both 0 in every year/resolution group) — this was specifically an A03 decoding problem, not damaged source data.

  **Do not delete the ENTSO-E cache.** `_request()` returns the raw cached XML directly on a cache hit, and `_fetch_generic()` calls `_parse_timeseries()` fresh on every fetch regardless of cache hit/miss — so simply re-running ingestion against the *existing, untouched* cache reparses the exact same historical documents with the corrected parser. Deleting the cache would instead trigger fresh API calls that could return a different publication/revision state, destroying the provenance link between "the data that produced every result in this README" and "the data being re-parsed." A genuinely fresh-API sensitivity check, if wanted later, belongs in a separate cache directory, not as a replacement for reparsing the audit source.
- **RESOLVED — `auction_sequence == 1` is CONFIRMED as the correct DE-LU day-ahead reference price.** Cross-checked against SMARD (Bundesnetzagentur's official German electricity market transparency platform, an independently-retrieved reference not sourced through the same ENTSO-E Transparency Platform XML pipeline this project's data comes from — see `scripts/verify_against_smard.py`) for **every one of the 8,833 disagreeing intervals** in the corrected (post-A03-fix, `a03fix_v1`) audit across the entire Oct 1 – Dec 31 2025 window — not a manual sample, the complete set. Result: **8,833 of 8,833 (100%) consistent with sequence 1; zero consistent with sequence 2 to the exclusion of sequence 1; zero consistent with neither.** (3 intervals initially classified "ambiguous" on a 0.01 EUR/MWh tolerance were inspected directly: in every one, SMARD's price is an *exact*, bit-for-bit match to sequence 1, e.g. 118.71 = 118.71; sequence 2 only cleared the tolerance because it happened to sit exactly one cent away — a tolerance-boundary artifact, not genuine ambiguity.) Per the pre-registered hard rule: SMARD consistently matches sequence 1 → **assumption closed.**

  This also means the corrected-data auction-sequence disagreement statistics are now known precisely, superseding the pre-fix figures below: **8,833 of at most 8,836 possible intervals disagree (~99.97%)** — a *higher* rate than the original pre-fix 96.8%, because the old buggy parser's dropped positions were simply incomparable rather than biased toward "agree." **`sequence2_only_intervals.csv` no longer exists on the corrected data** — the original 56-interval "sequence 1 entirely missing" problem was confirmed to be purely an A03 parsing artifact, fully resolved by the parser fix itself, independent of the SMARD check.

  **Every point-forecast, uncertainty, and robustness result in this README that depends on the post-cutover target price** (`baseline_v1_a03fix`, `xgboost_v1_a03fix`, `uncertainty_selected_v1`, the Tier-1 robustness check, the regime diagnostic) **now rests on a confirmed-correct target definition, not merely a well-supported assumption.** Nothing needs to be rerun because of this — the target was already `auction_sequence == 1` throughout; this closes out what had been an open caveat on every one of those results.

  <details>
  <summary>Historical record of the investigation (click to expand) — preserved per this project's practice of not deleting superseded findings</summary>

  - *(pre-fix, provisional, superseded by the confirmed figures above)* 96.8% of ALL PT15M intervals across the entire Oct 1 – Dec 31 2025 window disagreed between sequence 1 and sequence 2 (8,553 of 8,836 possible intervals). Median gap €7.04/MWh, mean €11.28/MWh, max €278.69/MWh (2025-10-01 18:15 local: €200.00 vs €478.69).
  - *(pre-fix, provisional, superseded)* 56 intervals across 27 dates had sequence 2 present with sequence 1 entirely absent (`sequence2_only_intervals.csv`) — confirmed above to be purely an A03 parsing artifact.
  - **Strong internal structural evidence (independently verified against the real audit output, still valid): sequence 1's *resolution* transitions from PT60M to PT15M exactly at the SDAC MTU cutover; sequence 2 is PT15M throughout the entire history.** Checked directly against the full structural-periods file (5,538 Period rows): pre-cutover, sequence 1 = PT60M and sequence 2 = PT15M with zero exceptions (2,465/2,465 each); post-cutover, both are PT15M (304/304 each). This made sequence 1 considerably less arbitrary even before the SMARD check confirmed it directly.
  - **XML metadata inspection (`scripts/inspect_sequence_xml_attributes.py`) ruled out "two different auction products/mechanisms" as the explanation** — `businessType`, `auction.type`, `contract_MarketAgreement.type`, `curveType`, both domains, currency, and price unit are all identical between sequence 1 and sequence 2.
  - **ENTSO-E's official schema defines `classificationSequence` as distinguishing multiple auctions published within the same auction category and contract type — it does NOT itself establish that position 1 is "primary."** This is why the external SMARD check above was necessary rather than inferring primacy from the schema alone.
  </details>
- **Wind/solar forecasts (and residual load, renewable share, derived from them) are not proven point-in-time safe at the 11:45 D-1 decision cutoff** — see "Point-in-time gap" above. Tagged as Tier 2 in `FEATURE_AVAILABILITY_TIER`. The empirical robustness check (`baseline_v1`) shows this is **not a minor caveat**: Tier-1-only ElasticNet retains just 23% of the full model's advantage over lag-24 in the regime-stress fold (the one fold under the same 2025-10-01+ market design as the entire 2026 holdout), versus 40–55% in earlier folds. The model's edge increasingly depends on the features we can least prove are point-in-time safe, in exactly the regime the holdout will test. This should be weighed explicitly before treating final holdout results as conclusive.



**The same gap was checked for and found in two more scripts.** `verify_auction_sequence.py` and `audit_price_curve_types.py` both had the identical fixed, unversioned, unguarded output path (`outputs/auction_sequence_verification/`, `outputs/curve_type_audit/`). `inspect_sequence_xml_attributes.py` saves nothing to disk, so it wasn't affected. Both fixed the same way: `run_version` is now mandatory (`python scripts/verify_auction_sequence.py <run_version> [n_sample_dates]`, `python scripts/audit_price_curve_types.py <run_version> [start] [end]`), each writing to a new versioned subdirectory rather than the old fixed path.

**Unlike the EDA case, this was caught before the loss happened** — if these two scripts hadn't been re-run since the A03 fix, the pre-fix audit outputs (the ones the 56-sequence-2-only and 96.8%-disagreement figures came from) are still sitting at the old unversioned paths, untouched, since the new versioned runs write to a different location entirely. **Re-running these against the corrected data now gives a genuine pre/post-A03 comparison** for the auction-sequence audit — something EDA lost the chance to do.

## EDA versioning (a real gap found while trying to do the pre/post-A03 comparison)

Attempting the pre-A03 → post-A03 EDA comparison surfaced a real problem: `run_eda.py`'s output directory was never versioned — a fixed path (`outputs/eda/<stem>/<scope>/`), overwritten on every run with no guard. Since EDA had already been re-run once on the corrected (post-A03-fix) data, **the original pre-A03 EDA output no longer exists** — there is nothing left to diff against. Unlike the ingestion-log append bug (where the fix could be applied and old entries simply ignored going forward), this loss is not recoverable: the pre-A03 structured EDA data was never preserved anywhere.

Fixed going forward: `run_eda.py` now requires an explicit `run_version` (matching `run_models.py`'s pattern) and is `FileExistsError`-guarded against overwriting a previous run. `scripts/compare_eda_runs.py` diffs two versioned EDA runs' summary CSVs directly — tested end-to-end with synthetic data (two runs differing by a known, deliberate price shift), and it correctly surfaced that exact shift across every relevant table while showing zero spurious difference elsewhere. It can't be used retroactively for the A03 transition specifically, but it's ready for the next time this project's data changes and a real comparison is needed.

**The current EDA numbers stand as the current record, now properly versioned as `development_a03fix_v1`** (`outputs/eda/delu_features/development/development_a03fix_v1/`). Confirmed to exactly reproduce the earlier, pre-versioning EDA output on every figure spot-checked (count=61,368; mean=94.746001; std=93.794805; 2,051 negative-price hours; 5,869 above €200; 530 above €500; 59,159/2,209 pre/post-cutover split) — expected, since the underlying `delu_features.parquet` hasn't changed since that run; this is a determinism confirmation, not new information. What's new is that this snapshot is now durable: `FileExistsError`-guarded, so it can't be silently overwritten by a future run the way the pre-A03 version was. There is still no verified "before" figure to contrast it against; treat any pre-A03 numbers mentioned elsewhere in this README's history/investigation-log sections as approximate, prose-only recollections, not verified structured data.

## Process notes (lessons from real recurring mistakes, not hypothetical)

Four failure patterns actually happened more than once in this project's development and are now guarded against mechanically rather than left to memory:

1. **A corrected claim recurring in a different file.** The "classificationSequence is absent before 2025-10-01" claim was proven false once, then independently resurfaced in a test docstring and in this README from copies that hadn't seen the fix. Guard: `python scripts/check_stale_claims.py` greps the whole repo for a small registry of specific, previously-real false claims (not general style). Deliberately does not include anything that needs context to judge as wrong (a plain substring match would false-positive on legitimate correct usage) — only exact statements that are wrong under every framing.
2. **An orphaned test body silently merged into the previous test's function.** An edit replaced a test's `def` line but left its old body behind; with no `def` at column 0 between them, Python treated it as more statements appended to the prior test, not a syntax error. The logic kept running and passing, just under the wrong test's name — `pytest --collect-only` showed a normal, unremarkable count throughout, so test count alone never revealed it. Guard: `python scripts/check_orphaned_test_bodies.py` walks each test file's AST for a string-literal expression statement that isn't a function's first statement (real docstrings always are) — the specific, verified shape of this exact bug.
3. **An ad-hoc diagnostic script reimplementing part of the production selection pipeline.** `diagnose_price_duplicates.py` applied `select_price_resolution()` but never `select_primary_auction_sequence()`, unlike real `fetch_day_ahead_prices()` — producing a "duplicate timestamps" finding that was really just a missing selection step, not new corruption. No automated guard for this one (it's a design/review discipline, not a mechanical pattern) — but any new diagnostic script touching price data should call `fetch_day_ahead_prices()` itself, or explicitly apply both selection steps in production order, rather than reimplementing pieces.
4. **A log file accumulating stale entries across re-runs with no explicit versioning policy.** `save_ingestion_log()` opened in append mode; three re-runs during debugging each added their own records, and reading the log naively (as `diagnostics/inspect_price_xml.py` did) saw ~4x the real chunk count. Fixed to overwrite. This project now uses two deliberate, distinct patterns for generated artifacts — new code adding either should say explicitly which one applies:
   - **Current-state artifacts** (`delu_hourly.parquet`, `ingestion_log.jsonl`): represent "what the most recent run produced." Overwritten each run. No versioning needed because there's only ever one meaningful current answer.
   - **Frozen evidence artifacts** (`outputs/models/*/baseline_v1/`, `xgboost_v1/`): represent a specific, citable experimental result. Guarded by `FileExistsError` against silent overwrite — a new run needs an explicitly new `run_version` name, so old evidence is never silently lost.
5. **Unequal comparison populations, the same failure mode twice in two different layers.** First for point models (fixed early: `common_comparison_rows`, all six models scored on identical rows). Then again for uncertainty window candidates (`run_uncertainty_sensitivity.py`): different `window_days` have different warm-up lengths by construction, so without correction a shorter window is partly scored on early data a longer window never gets evaluated on at all — confounding "which window is better" with "which period got included." Fixed the same way: `find_common_evaluation_start()` + `common_start` restrict every candidate to an identical row set, with an explicit assertion (`len(n_values) == 1`) rather than an implicit hope. **The recurrence itself is the lesson**: a fix applied once, in one layer, doesn't automatically propagate to the next layer that has the same shape of problem — this needs to be checked deliberately every time a new comparison is built, not assumed solved because it was solved before.

**The general rule, going forward**: whenever a design decision would otherwise be justified with "it should be okay because..." — stop, and turn it into one of three things: a **test** (if it's a checkable property), a **documented assumption** (if it's genuinely unverifiable right now, like `auction_sequence == 1`), or a **documented limitation** (if it's a known, accepted gap, like Tier-2 point-in-time availability). "Should be fine" is not one of the three options.

Run `python scripts/run_all_checks.py` (checks 1 and 2 above) before treating any batch of file changes as final, in addition to `python -m pytest tests/ -v`. Run `python scripts/generate_status.py` to get a real, generated test-count/check-status snippet rather than hand-typing a number here — this README's test count has gone stale from manual editing more than once; the generator can't fix the surrounding prose, but it makes the count itself a fact instead of a guess.

## Uncertainty quantification

**Selected: `uncertainty_selected_v1`, `window_days=60`.** Chosen by a pre-registered sensitivity experiment (`run_uncertainty_sensitivity.py`, candidates `[60, 90, 120, 180, 365]` fixed before any real-data run) and validated twice: once on the available-row comparison, then again on a common-row-corrected comparison (every candidate scored on an identical n=24,114 rows, after finding this confound was real — see `find_common_evaluation_start()`). **The ranking held under the stricter, corrected methodology** — 60 days won both times, so this isn't an artifact of unequal row counts. No fold exceeds the pre-registered 3-point calibration-deviation flag threshold (worst is `fold_3` at 2.07 points); `regime_stress_test`, the fold closest to the eventual 2026 holdout, is the *best*-calibrated of the four at 0.53 points off nominal. `config.yaml`'s `residual_window_days` now reflects this selected value; `run_uncertainty.py` also accepts an explicit `window_days` CLI argument so a consequential frozen run's provenance is traceable directly from its invocation, not only from a config default that could later be edited.

`uncertainty_v2` (`window_days=180`, the original, provisional value) is kept, not deleted — see below for what it revealed.

**Tier-1 robustness check** (`uncertainty_tier1_robustness_v1`): the frozen 60-day specification applied to XGBoost Tier-1's predictions, no retuning. Tier-1 achieves good pooled calibration (80.55% vs. 80% nominal) but becomes mildly overconservative in `regime_stress_test` specifically — empirical coverage there reaches 83.05%, which by the same 3-point deviation threshold used elsewhere in this project (`FOLD_CALIBRATION_FLAG_THRESHOLD`) is itself flagged (3.05 points off nominal, just over the line). This comes at a real cost: intervals 55.1% wider than Full's, and a 45.5% worse interval score — Tier-1's uncertainty quantification is technically adequate but substantially less sharp/informative, mirroring and extending the already-established point-accuracy gap between Full and Tier-1 into the uncertainty layer.

**Regime diagnostic** (`uncertainty_tier1_regime_diagnostic_v1`, descriptive only, no promotion decision): investigated whether Tier-1's lower downside-tail miss rate in `regime_stress_test` (vs. Full) is mechanical (wider intervals alone) or structural (genuinely different signed forecast errors), by decomposing point-forecast and envelope-width effects separately via a point×envelope cross on real data. **Result: mechanical, decisively — the envelope-width effect alone (+0.0915 reduction in lower-tail miss rate) is large and positive; the point-forecast effect alone is large and *negative* (−0.1205, i.e. Tier-1's point forecasts are actually worse in the tail, mean of worst 10% of residuals −61.5 vs. Full's −37.8) — the observed net improvement is the wider envelope more than compensating for a genuinely worse point forecast, not evidence that Tier-1 predicts the downside better.** The two effects do not combine additively (interaction term +0.080, larger than either isolated effect) — a real nonlinearity, not a modeling artifact.

Rolling-window empirical-residual-quantile intervals (`src/uncertainty.py`, `run_uncertainty.py`) built on top of `xgboost_v1_a03fix`'s pooled out-of-sample predictions across all four contiguous folds (config: quantiles `[0.1, 0.5, 0.9]`, `residual_window_days=180`). `uncertainty_v2` is the frozen reference run — **preserved as-is, not touched, not re-tuned**.

**Pooled calibration is excellent**: 79.58% empirical coverage on the nominal 80% interval (n=25,218), tail fractions 9.98%/10.44% against a 10%/10% nominal split.

**But per-fold calibration is NOT stable, and the direction of miscalibration changes across folds — this is a real, documented open limitation, not resolved by the good pooled number:**

| Fold | Coverage | Below lower | Above upper | Direction |
|---|---:|---:|---:|---|
| fold_1 | 80.36% | 7.89% | 11.75% | upside misses dominate |
| fold_2 | 78.38% | 12.34% | 9.28% | downside misses dominate |
| fold_3 | 81.15% | 8.38% | 10.47% | slight upside dominance |
| regime_stress_test | **77.03%** | **12.60%** | **10.38%** | downside misses dominate |

`regime_stress_test` — the fold closest to the eventual 2026 holdout — has the lowest coverage of the four (2.97-point shortfall, ≈278 of 2,207 observations falling below the lower bound vs. ≈229 above the upper bound). **The correct characterization is "regime-sensitive/time-varying calibration instability," not a permanent downside bias** — the miscalibration direction genuinely oscillates fold to fold (fold_1 and fold_3 skew upside, fold_2 and regime_stress_test skew downside), and the good pooled number is partly the different folds' errors cancelling rather than genuine stability. A naive independent-Bernoulli standard error puts `regime_stress_test`'s shortfall at ≈3.5σ, but electricity-market interval misses cluster in time (a single price event can cause several consecutive misses), so the effective sample size is smaller than the raw row count and this overstates confidence — **worth taking seriously, not treated as statistically proven** without a correlation-aware test (e.g. block bootstrap) this project doesn't yet have.

**What has NOT been shown**: *why* `regime_stress_test` under-covers. "The 180-day window is still dominated by calmer preceding months and hasn't adapted to the new regime" is a plausible, untested hypothesis — not yet distinguished from the alternative (a few clustered extreme-price events, which a shorter window wouldn't necessarily fix). Rolling 30-day coverage over 2023–2025 would distinguish "slow drift" from "event clustering," and hasn't been built yet.

**Scope boundary for future work**: this interval quantifies *price forecast error* (P − P̂), not strategy P&L. VaR/ES for the eventual trading layer should be built on the realized loss distribution after position sizing and transaction costs, not by naively extending this same residual-quantile machinery to small tail probabilities — a 180-day hourly window has only ~4,320 observations, meaning a 1% tail has only ~43 effective observations, too few for a reliable extreme-quantile estimate.

**Deliberately not done in response to this finding**: changing `residual_window_days` based on having seen the 77.03% result. That would be exactly the kind of post-result tuning this project has consistently avoided elsewhere (the ElasticNet `baseline_v2` stopping rule, XGBoost's no-retuning-after-`xgboost_v1` rule). A proper, pre-registered sensitivity experiment across several candidate windows — chosen before looking at any result, scored on calibration, sharpness, and tail balance together via a proper scoring rule (interval/Winkler score), not "closest to 80% coverage" — is the correct next step if this is revisited; see `run_uncertainty_sensitivity.py`.

## Next steps

**Forecasting and uncertainty layers are complete** (see Status above) — items 1–6 below, previously listed as pending, are done: A03 fix reproduced, EDA versioned and rerun, `baseline_v1_a03fix`/`xgboost_v1_a03fix` re-verified with XGBoost re-promoted, uncertainty selected via a pre-registered sensitivity experiment, and the auction-sequence assumption confirmed via an independent SMARD cross-check (not the originally-planned manual EPEX check, which turned out to be blocked behind a paid tier — see "Auction sequence" above for why SMARD was used instead and why it's a valid substitute).

What's actually left, in order:

1. **Economic contract specification** (`docs/economic_contract_v1.md`) — decision time, information set, instrument, position, entry/execution price, settlement price, costs, position limits, P&L formula, all written down *before* any strategy code, per this project's established pre-registration discipline.
2. **Strategy layer**: pre-registered candidate policies (benchmark, point-forecast, uncertainty-aware, Tier-1), parameters estimated on training data only, a common-row-set guard analogous to `common_comparison_rows`/`find_common_evaluation_start`, hourly *and* daily P&L, and the Full-vs-Tier-1 comparison carried through into economic terms, not just point MAE.
3. **Tail-risk layer**: loss defined as `-P&L` (not reusing `uncertainty_selected_v1`'s price-residual intervals, which are a different quantity), a small pre-specified risk model set, point-in-time-safe VaR/ES with its own leakage test, and block-aware backtesting given how much this project's own uncertainty-coverage work already showed misses cluster in time.
4. **Freeze protocol**: a hashed final specification manifest, a literal holdout lock (ordinary scripts refuse timestamps ≥ 2026-01-01), and a full dry run on development data before the holdout is ever touched.
5. **The 2026 holdout, exactly once** — reported honestly regardless of outcome, no retuning afterward.
