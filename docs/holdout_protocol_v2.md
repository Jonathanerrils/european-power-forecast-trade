# Holdout Protocol v2 — What Happens When 2026 Is Opened

**Status: FROZEN PRE-HOLDOUT PROTOCOL, finalized before any 2026 outcome metric
exposure.** 2026 outcome data have not been used for model selection, tuning,
forecast-performance evaluation, strategy evaluation, uncertainty selection, or economic
interpretation. This document exists because the 2026 holdout can only be opened once.
Every choice that would otherwise get made *during* the holdout run — which model,
which retraining rule, which mask, which success criterion — is made here instead,
before any 2026 outcome has been seen by anyone. Per this project's established
discipline: a decision made after seeing a result is not a finding, it's a
rationalization. This document is the mechanism that prevents that for the single most
consequential run in the project.

Changing this document after the holdout has been opened is not permitted. If a genuine
error is found in this protocol itself before the holdout run, fix it and note the
correction explicitly with a timestamp — but once the first 2026 metric has been
computed and seen (Section 11's precise trigger — not merely "a script has executed,"
since a run that crashes before producing a metric is not an exposure event, per
Section 1), this document is as frozen as the `economic_contract_v2.md` numbers it
depends on.

## Changelog from v1 (before any holdout run — ordinary pre-freeze correction, not a violation of the "once" rule)

`v1` was reviewed and found to contain three factual errors and one imprecise phrase,
corrected here before commit/tag, exactly per Section 11's "before the first metric is
computed" rule:

1. **Section 2/3 (v1)** claimed Full and Tier-1 share "identical frozen hyperparameters."
   False — the actual saved artifacts show fold-level Full and Tier-1 hyperparameters
   genuinely differ (e.g. fold 1: `learning_rate` 0.10 vs 0.03, `subsample` 0.8 vs 1.0),
   because `fit_xgboost()` searches each information set independently. Corrected below.
2. **Section 4 (v1)** specified `latest_residual_quantile_offsets()` as the holdout
   uncertainty function. That function's window is computed per *hourly* timestamp,
   which can let an afternoon hour of day D see residuals from earlier hours of that
   *same* day D — safe for one-step-ahead hourly evaluation, not safe for this
   project's whole-day D-1 decision. Corrected to
   `residual_quantile_offsets_for_delivery_day()` / `compute_delivery_day_residual_quantiles()`,
   which restrict history to `delivery_date < D` and give one identical snapshot per
   day — the same correction already applied to and verified in the development chain
   (`economic_contract_v2.md`).
3. **Section 8 (v1)** used subjective language ("most," "visibly smaller," "qualitatively
   different") in its success criteria. Replaced with a fully mechanical rule below.
4. **Section 1 (v1)** said 2026 "has never been touched." Imprecise — this project's
   dataset already contains 2026 rows for structural ingestion/coverage validation.
   Corrected to the precise claim below.
5. **The v2 headline Status line and line 14's freeze-trigger wording** still repeated
   v1's imprecise "2026 has never been touched" / "has actually executed" phrasing even
   though the detailed sections below them had already been corrected — an internal
   inconsistency within v2 itself, caught and fixed before commit.

Additionally, while making these edits, a `str_replace` mistake briefly duplicated
Section 7's "Development runners must never be pointed at 2026 data directly" paragraph
and deleted the section header entirely for one intermediate draft — caught by
diffing the section-header list before finalizing, not left in the committed version.
This is exactly the class of editing error this project's `check_orphaned_test_bodies.py`
was built to catch in code; the same discipline (verify structure after every edit,
don't assume a replace worked) applies to this document too.

A second review pass found four further precision issues, also corrected before commit:

6. **Section 4's final bullet** still said the holdout uses "whichever the frozen
   development-corrected sensitivity run selected" — looser than Section 2's exact
   lineage binding. Replaced with the literal frozen values.
7. **Section 4's "no historical residual at all"** was slightly misleading — 2026
   forecasts have substantial development residual history behind them, just no
   realized residual of their own before delivery. Corrected.
8. **Section 6 incorrectly suggested `verify_auction_sequence.py`** for a July 2026
   completeness check. That script deliberately stops at `2026-01-01` specifically to
   avoid inspecting 2026 price outcomes before the holdout; its 2025 result is already
   the sequence-1 evidence. Replaced with a description of a purely structural audit
   (timestamps, interval counts, gaps/duplicates, PT15 structure, sequence-1 presence,
   raw-cache completeness) that makes no price-value comparison.
9. **Section 9's header and a sample-size fraction** used "before real 2026 data is
   touched" (imprecise, given structural ingestion already occurred) and "a sixth the
   size of development" (212/1,081 ≈ 19.6%, closer to a fifth). Both corrected.
10. **Section 6's `HOLDOUT_END` boundary moved from proposed to frozen** (2026-09-11)
    after three independent structural completeness checks against July 2026 data all
    passed: raw cache completeness (0 missing/failed chunks), raw PT15M sequence-1
    structural integrity (2,976/2,976 expected intervals, 0 gaps, 0 duplicates), and
    processed hourly dataset completeness (744/744 expected rows, 0 gaps, 0 duplicates,
    0 missing target values). All expected counts were independently recomputed from
    first principles before being accepted, not read off the tooling's own claim alone.
    See Section 6 for the full record.

## 1. What "opening the holdout" means, precisely

The holdout is **exposed exactly once through one frozen orchestration pipeline**.
Internal components may execute multiple functions, scripts, or subprocesses — a crash
before any 2026 metric is computed and rerunning the pipeline is not a "second look" —
but **no 2026 metric may be inspected until all prerequisite validation and provenance
gates have passed**. Before that exposure:

- Every item in this document is fixed.
- A full dry run has been completed against synthetic or `regime_stress_test`-shaped
  data (see Section 9) and passed without needing any code change.
- `docs/economic_contract_v2.md` remains unmodified since its own freeze.

After that exposure:
- The result is reported, in full, regardless of outcome.
- No parameter, model, or rule in this document may be changed and re-run against 2026
  data to "improve" the result. A genuine bug found *after* opening the holdout follows
  the rule in Section 11, not a silent redo.

**Precise claim about prior 2026 contact, not the absolute one v1 made**: 2026 outcome
data have not been used for model selection, tuning, forecast-performance evaluation,
strategy evaluation, uncertainty selection, or economic interpretation. Structural
ingestion and coverage validation (confirming the raw data exists and is well-formed)
does not constitute holdout metric exposure and has already occurred as part of this
project's data pipeline. This is the accurate claim; "2026 has never been touched" was
not.

## 2. Frozen model artifacts — exact run versions, no substitutions

| Component | Exact artifact | Notes |
|---|---|---|
| Full point forecast | `XGBOOST_PREDICTOR_COLS`, refit on the final training window per Section 3 | Not reused from any individual development fold's fitted model. |
| Tier-1 point forecast | `XGBOOST_TIER1_PREDICTOR_COLS`, refit **independently** on the same final training window per Section 3 | Independently searched, not "identical hyperparameters to Full" (v1's error — see Changelog). |
| Naive benchmark | `lag_24` | Unchanged definition. |
| Uncertainty | Delivery-day-safe method, exact frozen lineage: `uncertainty_window_sensitivity_v3_dayorigin` mechanically selected **60 days** → `uncertainty_selected_v2_dayorigin` (`window_days=60, min_periods_days=15, quantiles=[0.1, 0.5, 0.9], information_set_method="delivery_day_safe"`) | This exact frozen artifact, not "whichever is current" — development is already frozen (tag `corrected-development-v2-dayorigin`), so there is no ambiguity left to resolve at holdout time. |
| Strategy rules | `docs/economic_contract_v2.md`, S0–S5, exactly as specified | No new strategy variants introduced for the holdout. |
| Economic parameters | `η_rt=0.85, c=€10` (primary); full pre-registered `3×3` grid (`η_rt ∈ {0.70, 0.85, 0.92}`, `c ∈ {5, 10, 15}`) | Identical to development. |
| Risk method | Empirical historical VaR/ES at `{95%, 99%}`, per `run_strategy_tailrisk.py`'s frozen, bug-fixed implementation | Same method. Sample-size caveat below is new to this document, not a method change. |

## 3. Final model refit rule — independent selection per information set, on all development data

The Full and Tier-1 models used for the holdout are each **refit once, independently, on
the entire development period** — not reused from any individual fold's fitted model,
and not sharing a single set of hyperparameters between them (v1's error — see
Changelog). Training window: `src/splits.py`'s `get_final_train_window()`, `2019-01-01`
(inclusive) to `2026-01-01` (exclusive) — all of `fold_1` through `regime_stress_test`'s
data combined.

**Explicit rule**: Full and Tier-1 each run the **same already-frozen XGBoost
model-selection procedure** used throughout development — the unchanged 36-combination
`XGBOOST_PARAM_GRID`, MAE scoring, three-fold delivery-day-aligned inner CV
(`day_aligned_cv=True`, called explicitly, not relying on `fit_xgboost()`'s default),
identical to `fit_xgboost()`'s existing behavior — applied independently to each
information set, since Full and Tier-1 use different predictor sets and are expected to
select different hyperparameters, exactly as they did in every development fold. This is
**not new experimental tuning**: the selection *procedure* was frozen before the
holdout; this step applies that already-frozen algorithm to the final training window,
once per information set. No grid expansion, alternative metric, extra search, feature
change, or manual selection of one of the old fold-level parameter sets is permitted.
Both fitted final models and their selected hyperparameters are saved and frozen before
the holdout runner touches them.

## 4. Uncertainty procedure for the holdout — the corrected delivery-day-safe path

Development-sample uncertainty (S3, S5) is built via
`compute_delivery_day_residual_quantiles()` — every hour of delivery day D uses residual
information ONLY from delivery days strictly before D, never an earlier hour of D
itself (`economic_contract_v2.md` §3, hostile-tested in `src/uncertainty.py`).

**2026 forecasts have development residual history behind them, but no realized
residual of their own before delivery** — a 2026 prediction's own residual is not known
until the corresponding actual price clears, even though the residual history it draws
its trailing window from (development, and later revealed holdout days) is substantial.
The holdout runner must use `residual_quantile_offsets_for_delivery_day()` (the
single-day primitive), called **once per holdout delivery day D**, with the residual
history extended incrementally:

**Exactly which residuals initialize this history, stated precisely to prevent a subtle
but real error**: the initial holdout residual histories (Full and Tier-1, separately)
are the corrected out-of-sample development residuals already generated from the
cross-validated `xgboost_v1_a03fix` fold predictions — the same residual series
`uncertainty_selected_v2_dayorigin` was built from. They are **not** in-sample residuals
from the newly-refit final 2019–2025 models (Section 3). This distinction matters: the
final models are refit on *all* development data, so evaluating them on that same data
would produce artificially small, in-sample residuals — using those to estimate
uncertainty quantiles would understate the true forecast error and miscalibrate the
holdout intervals. After delivery day D is revealed (its actual prices clear), the
**final holdout Full/Tier-1 models'** residuals for D — genuinely out-of-sample, since
D is holdout data the final models never trained on — are appended to their respective
histories, for use starting D+1. The two residual series (development OOS residuals,
then holdout-model OOS residuals from D+1 onward) are concatenated into one continuous
history; the source of a residual (development vs. revealed holdout) does not need to
be tracked separately once appended, only that day D's own residual is correctly
excluded until after D is revealed.

- Before predicting day D, the eligible residual history is: all development OOS
  residuals, plus every already-**revealed** holdout day's residual (i.e. holdout days
  strictly before D whose actual prices have already cleared).
- Day D's own residual is never in that history when D itself is being predicted — it
  only becomes eligible for D+1 and later, once D's actual prices clear.
- This is a loop the holdout runner must implement explicitly, day by day, in
  chronological order — it is not something a single call with a longer series achieves
  (the corrected function computes one fixed snapshot per call; see its docstring).
- The window_days, min_periods_days, and quantiles used are the literal frozen values
  from Section 2's exact lineage — `window_days=60`, `min_periods_days=15`,
  `quantiles=[0.1, 0.5, 0.9]`, `information_set_method="delivery_day_safe"` — not
  re-selected here, and not referenced indirectly through "whichever run is current."

## 5. Common evaluation mask for the holdout

Built the *same way* as development — `build_common_evaluation_days()`
(`run_strategy_backtest.py`), reused verbatim, applied to holdout-period data instead of
development-period data. A holdout day is included only if every strategy's required
inputs (Full/Tier-1 point forecasts, Full/Tier-1 uncertainty bounds, `lag_24`) are
complete for every hour of that day — identical standard to development, no relaxation
because the sample is smaller.

## 6. Holdout boundary and sample size — frozen before outcome exposure

The holdout runs from `2026-01-01` to a fixed, now-verified exclusive end boundary —
`get_holdout_window()` deliberately has no unsafe default for this exact reason
(`src/splits.py`) — the future `run_final_holdout.py` must supply the frozen endpoint
below explicitly, not infer it from `data.end_date: null` in `config.yaml`, which is a
`run_uncertainty.py`/`run_xgboost.py`-era convenience default for "latest complete month
at ordinary development run time," not a holdout-safe value.

**FROZEN boundary**:

```text
HOLDOUT_START = 2026-01-01 (Europe/Berlin), inclusive
HOLDOUT_END   = 2026-08-01 (Europe/Berlin), exclusive
```

giving 212 nominal calendar days before the common-day completeness mask (Section 5) is
applied.

**Structural completeness verification — performed, passed, recorded here as required
before this boundary could move from proposed to frozen.** Per this section's own
requirement, `verify_auction_sequence.py` was correctly NOT used (it deliberately stops
at `2026-01-01` to avoid any 2026 price-outcome inspection; its 2025 result remains the
sequence-1 evidence). Three independent structural checks were run instead, each
purely structural — no price value or auction-sequence-value comparison, only presence,
counts, gaps, and duplicates:

1. **Raw cache completeness** (`audit_price_curve_types.py`, cache-only, no network
   fetch): `audit_end = 2026-07-31T22:00:00+00:00` (exactly `2026-08-01T00:00`
   Europe/Berlin — the correct upper boundary), `expected_cache_chunks = 8`,
   `successful_cache_chunks = 8`, `missing_cache_chunks = 0`, `parse_failed_chunks = 0`,
   `complete = true`.
2. **Raw PT15M sequence-1 structural audit**, direct inspection of
   `curve_type_structural_periods.csv` for July 2026 (`2026-06-30T22:00:00Z` to
   `2026-07-31T22:00:00Z`), filtered to `resolution == "PT15M"` and
   `auction_sequence == 1`, expanded to individual 15-minute interval starts: 2,976
   expected (31 days × 96 intervals/day, no DST transition in July), 2,976 expanded,
   2,976 unique, 0 duplicates, 0 non-15-minute gaps, first interval
   `2026-06-30T22:00:00Z`, last interval `2026-07-31T21:45:00Z` — both boundaries
   independently confirmed correct (translate to `00:00`/`23:45` Europe/Berlin on the
   correct calendar days). Expanded count equaling unique count with zero gaps confirms
   auction sequence 1 alone provides complete, non-overlapping, gap-free coverage of
   the entire month — not merely present on average, but present for every interval.
3. **Processed hourly dataset completeness** (`data/processed/delu_hourly.parquet`,
   the actual dataset any real run reads from), same window: 744 expected hourly rows
   (31 × 24), 744 observed, 744 unique, 0 duplicates, 0 non-1-hour gaps, 0 missing
   `price_eur_mwh` values, boundaries independently confirmed correct.

All three checks independently verified (expected counts recomputed from first
principles, not merely read off the tool's own claim) before being accepted here. This
satisfies every element this section originally required: raw-cache completeness,
timestamp continuity, expected interval counts per delivery day, gaps/duplicates, PT15
sub-interval structure, and confirmation that auction sequence 1 is present for every
interval — none of it inferred from an unverified assertion.

**The 212-day holdout is roughly a fifth of the 1,081-day development sample**
(212/1,081 ≈ 19.6%) — this is now the exact, known figure, not an estimate.

**Explicit consequence, now known exactly rather than estimated as a range**: at 212
days, the empirical tail for **99% VaR/ES contains only ~2 observations** (212 × 0.01 =
2.12) and must be reported as illustrative only, never as a pass/fail criterion (also
see Section 8 — VaR/ES is explicitly excluded from the primary criterion regardless of
precision). **95% VaR/ES rests on ~10–11 tail observations** (212 × 0.05 = 10.6) — still
thin relative to development's ~54, reportable, but not treated with the same
confidence.

## 7. Reporting

The full 11-section structure from `run_strategy_report.py` (provenance, primary table,
four named deltas, `3×3` grid, uncertainty decomposition, drawdown, oracle/value
capture, concentration, regime split, stress split, limitations) is produced for the
holdout using the same, unmodified functions — a `run_final_holdout_report.py` importing
and reusing this project's existing reporting functions, not reimplementing them. The
holdout report is saved under a name unambiguously distinct from every development
artifact (e.g. `outputs/strategy/delu_features/holdout_v1/`,
`outputs/risk/delu_features/holdout_tailrisk_v1/`) and is **never** merged, averaged, or
otherwise combined with development-sample numbers into a single reported figure.

**Secondary point-forecast accuracy table, reported alongside the economic results but
never affecting the Section 8 classification**: identical holdout evaluation rows,
`lag_24`, `lag_168`, XGBoost Full, and XGBoost Tier-1 (`lag_168` is an existing,
already-established benchmark in this project — `src/models.py`'s `lag_168_pred`
column, built from `price_lag_168h`, the same way `lag_24_pred` already is), reporting
MAE, RMSE, and MedianAE. This answers a genuinely separate research question — does the
richer feature set improve raw forecast accuracy, independent of what the trading
strategy does with that forecast — and is reported purely descriptively. No tuning, no
model selection, and no influence on the primary economic confirmation/partial/failure
outcome follows from this table.

**Development runners must never be pointed at 2026 data directly.**
`src/strategy.assert_no_holdout_access()`, the decomposition provenance validator's
`holdout_used=False` requirement, and the tail-risk provenance gate all deliberately and
permanently forbid 2026 — this is a feature to preserve, not a constraint to route
around. The holdout requires its own, separate orchestration layer that reuses the
proven pure functions (`build_common_evaluation_days`, the strategy kernel, the
decomposition/report/tailrisk computation functions) without inheriting the development
runners' holdout-refusal wrapper:

```text
run_final_model_fit.py        -- Section 3's independent Full+Tier-1 refit
run_final_holdout.py          -- Section 4's day-by-day uncertainty + strategy decisions
run_final_holdout_report.py   -- Section 7's reporting, reused functions, plus the
                                  point-forecast accuracy table above
run_final_holdout_tailrisk.py -- VaR/ES on holdout daily P&L, same frozen method

tests/test_run_final_model_fit.py
tests/test_run_final_holdout.py
```

These do not yet exist and must be built and dry-run (Section 9) before Section 1's
exposure event.

## 8. Success / partial confirmation / failure — mechanical, defined now

**Primary holdout criterion, and only this determines confirmation/partial/failure**:

```text
Delta_forecast = S2 - S1 at eta_rt=0.85, c=10 (the primary cell)

CONFIRMATION:
    Delta_forecast > 0 at the primary cell
    AND Delta_forecast > 0 in all 9/9 pre-registered sensitivity cells.

PARTIAL CONFIRMATION:
    Delta_forecast > 0 at the primary cell
    BUT fewer than 9/9 sensitivity cells are positive.

FAILURE:
    Delta_forecast <= 0 at the primary cell.
```

No discretionary judgment enters this classification. It does not depend on sample size,
margin, or any secondary result.

**Secondary findings are reported, never redefine the primary outcome**:

```text
Tier-1 replication:      count and report signs of S4-S2 across the 9 cells.
Uncertainty replication:  count and report signs of S3-S2 and S5-S3 across the 9 cells.
Tail-risk (95%/99% VaR/ES): reported descriptively; explicitly NOT part of the primary
                             pass/fail criterion, and NOT overridden by a low-precision
                             flag at 99% (Section 6) -- reported as-is, flagged as-is.
```

A holdout sample about a fifth the size of development (212/1,081 ≈ 19.6%) cannot be
expected to reproduce a point estimate precisely even under a correctly-generalizing
model — this is exactly why the secondary findings are reported as descriptive evidence,
not folded into a composite score that could obscure which specific thing did or didn't
replicate.

**A "failure" result is not treated as something to explain away.** If it occurs, the
honest report is that the development-sample finding did not generalize under the exact
frozen conditions tested — the value of a pre-registered holdout is precisely that this
conclusion is allowed to stand, not quietly reinterpreted after the fact.

## 9. Dry run required before the real 2026 holdout evaluation produces any outcome metric

The entire pipeline described in Sections 3–7, including the new `run_final_*.py` suite,
must be executed once against `regime_stress_test`-shaped or synthetic data standing in
for "the holdout," and every structural check (common-mask construction, oracle
dominance, reconciliation assertions, provenance gates, the day-by-day uncertainty loop
in Section 4) must pass, **before** the same code is pointed at real 2026 data. This
finds implementation bugs against data whose answer is already known, not against the
one holdout run that matters.

## 10. What cannot change once the first 2026 result is seen

Everything in Sections 2–8: model artifacts and the independent-refit rule, the
uncertainty procedure, the common-mask construction, `S0`–`S5` definitions, `η_rt`/`c`
(primary and grid), the VaR/ES method and confidence levels, and the mechanical
success/partial/failure rule in Section 8. None of these may be adjusted and the holdout
re-run to produce a more favorable outcome. The holdout is opened exactly once
(Section 1).

## 11. If a genuine bug is found during or after the holdout run

- **Before the first 2026 metric is computed and seen**: fix the bug, re-run. This is
  ordinary debugging, not a violation of the "once" rule, since no result has been
  observed yet. (This is exactly the category this document's own v1→v2 correction
  falls into — see Changelog.)
- **After the first 2026 metric has been seen**: any correction is explicitly labeled
  **"post-holdout corrective analysis"** in every report that uses it, with the original
  (possibly bug-affected) result preserved alongside it, not silently replaced. The
  distinction between "the pre-registered result" and "a corrected re-analysis" must
  remain visible to a reader, permanently.

## 12. Standing limitation carried into the holdout, unresolved

**Tier-2 (wind/solar-forecast-derived) feature point-in-time availability at the 11:45
Europe/Berlin D-1 decision cutoff remains unproven**, exactly as documented throughout
this project's development-sample work. This is not resolved by reaching the holdout
stage — if anything, it matters more here: a live 2026 forecasting system would need
Tier-2 features to actually be available at the moment of decision, and this project has
never independently confirmed that timing holds in practice for the data source used.
The holdout evaluates the *model's* performance under the assumption that the historical
data's timestamps reflect genuine point-in-time availability — an assumption inherited
from development, not newly validated here.
