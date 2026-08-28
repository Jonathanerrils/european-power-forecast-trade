# Economic Contract v1 — Stylised Single-Cycle DE-LU Day-Ahead Storage Scheduling

**Status: FROZEN.** This document specifies the economic decision problem, P&L formula,
strategy candidates, and evaluation protocol *before* any strategy code is written or
any P&L number is computed. Per this project's established discipline (see README
"Process notes"), a design decision justified by "it should be okay because..." gets
converted into a test, a documented assumption, or a documented limitation — not left
implicit. This document is that conversion for the strategy layer. Changing anything
in this document after strategy code exists and has been run once requires an explicit,
labeled revision (`economic_contract_v2.md`), not a silent edit — the same discipline
already applied to `baseline_v1`→`v2`, `uncertainty_v2`→`uncertainty_selected_v1`, and
the `[60, 90, 120, 180, 365]` window sensitivity experiment.

**Revision note**: an internal design review before any code was written caught two
consequential issues (the post-cutover execution price wasn't actually tradable as
specified, and the cost convention was ambiguous by a factor that would have shifted
real trade/no-trade decisions) and several precision improvements. All are incorporated
below — this is the corrected, current specification.

## Why this framing, not a generic long/short trading rule

This project forecasts the DE-LU day-ahead clearing price itself — there is no second,
independently-validated market price (an intraday continuous price, a neighboring
bidding zone's price) to buy at one and sell at another. A generic
`P&L = P_actual - P_forecast` rule would be economically meaningless: it doesn't
correspond to a transaction anyone could actually execute.

A stylised storage operator sidesteps this cleanly: it buys and sells in the *same*
single day-ahead auction, at two different hours of the *same* delivery day. The
decision (which hour to buy, which hour to sell) is made D-1 from forecasts; the
economic outcome (P&L) is realized from the actual clearing prices once the auction
settles. This gives a genuine forecast → decision → realized-outcome chain without
inventing a second price series, and it exercises something none of the simpler
"decision-value" framings do: the model's ability to get the *relative intraday shape*
of the 24 (or 23, or 25) hourly prices right, not just their level or direction.

## Why single-cycle, not continuous multi-hour battery dispatch

A full continuous state-of-charge battery model (capacity, charge/discharge power
limits, continuous SOC trajectory, initial/terminal SOC constraints, degradation curve)
introduces 7–8 simultaneously-frozen physical parameters before any result exists — more
novel, unvalidated assumptions in one step than anywhere else in this project. Every one
of those parameters is a place a "mathematically attractive but not economically
attainable" mistake can hide, and a place post-hoc tuning against P&L could creep in
(discovering "a 2-hour battery looks best," then testing 1.5 hours, then 2.25 — exactly
the post-result-tuning pattern this project has repeatedly refused elsewhere).

**v1 is deliberately the simplest version that is still economically meaningful**: at
most one charge event and one later discharge event per delivery day, fixed 1 MWh
nominal quantity. This cuts the free-parameter count to a small, explicit set (round-trip
efficiency, degradation cost — see below) while keeping a genuine position and a genuine
realized payoff. A continuous multi-cycle battery is an explicitly deferred `v2`
extension, to be built only if it answers a *new* scientific question after `v1`
produces a clean, interpretable result — not automatically triggered by `v1` being
profitable, and not because more complexity looks more rigorous.

**No separately parameterized battery capacity or power-rate model.** The stylised unit
is not "capacity-less" — it is assumed capable of exactly the single fixed 1-MWh charge
and subsequent efficiency-adjusted discharge specified below. This normalizes the unit
of capacity out of the experiment rather than denying the operator has any capability at
all.

## Frozen specification

| Element | Frozen definition |
|---|---|
| Market | DE-LU day-ahead electricity market |
| Decision time | 11:45 Europe/Berlin, D-1 (matches `config.yaml::decision.cutoff_local_time`, already enforced by `src/features.py`'s leakage guard) |
| Information set | Exactly the frozen model's predictor columns — `XGBOOST_PREDICTOR_COLS` for the Full strategies, `XGBOOST_TIER1_PREDICTOR_COLS` for the Tier-1 strategies. No new features. |
| Point forecast | Frozen `xgboost_v1_a03fix` (`xgboost_full_pred` / `xgboost_tier1_pred`) |
| Uncertainty | Frozen `uncertainty_selected_v1` — 60-day rolling residual window, `q10`/`q50`/`q90`. **Provenance rule, development vs. holdout**: for the 2023–2025 development backtest, consume the already-saved OOS `quantile_forecasts.csv` from `uncertainty_selected_v1` directly (computed via `compute_rolling_residual_quantiles` across the full historical residual series) — do not regenerate it from current code, even though regeneration would be deterministic; reading the frozen artifact is stronger provenance than re-deriving old evidence. `latest_residual_quantile_offsets()` (`src/uncertainty.py`) is reserved for the genuinely different case of attaching an interval to a brand-new forecast with no historical residual yet — i.e. the eventual 2026 holdout, not development backtesting. |
| Decision unit | One local Europe/Berlin delivery day. **23, 24, or 25 hourly intervals as they actually occur** (spring/autumn DST) — never `range(24)`. The candidate hour set for day D is `H_D = {all actual delivery intervals belonging to local date D}`, matching `add_local_time_columns`'s existing `delivery_date` derivation. |
| Economic agent | Stylised, price-taking storage operator: purchases 1 MWh nominal in one hourly interval, releases the efficiency-adjusted energy in one later interval of the same delivery day. |
| Action | At most one charge/discharge pair `(i, j)` per delivery day, `i < j`, or **no trade**. `i < j` is a hard, tested invariant — a battery starting the day empty cannot discharge before it charges. |
| Chronology constraint | `t_sell > t_buy`, always, or no position. Not "closest to break-even" — a strict, enforced ordering. |
| Quantity convention | 1 MWh nominal charged at hour `i`; `η_rt` MWh discharged at hour `j`. |

### Execution price — tradable interpretation, including post-October-2025

**Pre-cutover (hourly market design):** 1 MWh purchased/sold at the realized hourly
DE-LU day-ahead clearing price for that interval — directly tradable as-is.

**Post-cutover (15-minute MTU, from 2025-10-01):** there is no longer a single traded
hourly product. An "hourly" strategy position is defined as **equal-energy baskets
across the four constituent 15-minute MTUs, with each leg's basket sized to that leg's
own quantity** — this was left implicit in an earlier draft and is stated explicitly
now, since leaving it ambiguous invites exactly the bug it's meant to prevent (a naive
implementation selling 1 MWh instead of `η_rt` MWh on the discharge leg):

```
Charge hour i:     0.25 MWh in each of the 4 quarter-hours    (sums to 1 MWh)
Discharge hour j:  η_rt/4 MWh in each of the 4 quarter-hours  (sums to η_rt MWh)
```

The cost/revenue of executing each basket is:

```
charge cost    = Sum(q=1 to 4) 0.25 * P_q       = (P_1+P_2+P_3+P_4)/4          = P̄_i
discharge rev. = Sum(q=1 to 4) (η_rt/4) * P_q    = η_rt * (P_1+P_2+P_3+P_4)/4   = η_rt * P̄_j
```

which is *exactly* the arithmetic-mean hourly price this project's own pipeline already
computes (`aggregate_quarter_hour_to_hourly`, spec section 4), scaled by `η_rt` on the
discharge leg only. This means the existing hourly price series already used throughout
this project's forecasting and uncertainty layers is, post-cutover, precisely the
effective EUR/MWh price of a real, executable equal-quarter-hour basket — not a
fabricated aggregate — and the P&L formula below (`η_rt · P̄_j − P̄_i − C`) is unchanged
by this clarification; only the physical basket sizing needed stating explicitly so
`strategy.py` cannot accidentally implement both legs as four × 0.25 MWh. This keeps the
entire hourly
research architecture intact while making the post-cutover execution claim genuinely
tradable, not just descriptively convenient.

### Costs — two distinct categories, not blended into one symbol

The literature evidence found for this contract is specifically about **battery
degradation cost**, not exchange/clearing/brokerage fees — a different, unsourced cost
category. These are kept separate rather than combined under one number:

```
C_total = C_degradation + C_market
```

**`C_market = 0` in v1, with an explicit limitation**: exchange membership, clearing,
brokerage, and other market-access charges are excluded from `v1`; the cost sensitivity
below represents battery cycling/degradation only. This is a stated scope limit, not a
silently-assumed-away cost.

**Degradation cost convention — per MWh of total throughput, not a flat per-cycle fee.**
The source evidence (He et al. 2021, cited in a power-system-dispatch-optimization
paper: $7–15/MWh throughput marginal degradation cost) measures cost proportional to
total energy moved through the battery, not a flat fee per completed cycle. With a
1-MWh charge and `η_rt`-MWh discharge, total throughput per cycle is `(1 + η_rt)` MWh:

```
C_degradation = c * (1 + η_rt)
```

where `c` is the frozen per-MWh-throughput rate below. **This matters concretely**: at
`c = EUR 10/MWh` and `η_rt = 0.85`, `C_degradation = EUR 18.50`, not EUR 10 — an 85%
difference from treating `c` as a flat per-cycle fee, large enough to flip marginal
trade/no-trade decisions. Using the throughput convention consistently avoids that
ambiguity.

| Parameter | Base | Low | High | Source / status |
|---|---:|---:|---:|---|
| Round-trip efficiency `η_rt` | **0.85** | 0.70 | 0.92 | Base: consistent with NREL's own 2024 Annual Technology Baseline, which states "the 2024 ATB assumes a round-trip efficiency of 85%" (`atb.nrel.gov/electricity/2024/utility-scale_battery_storage`, attributed to Cole and Karmakar 2023) — used here as the fixed reference case. **Low (0.70) is a deliberately conservative stress case**, not this project's estimate of typical field performance — informed by a source distinguishing manufacturer-claimed 85%+ nameplate efficiency from real-world figures once inverter, thermal-management, and auxiliary losses are counted, but not independently verified to the same standard as the base case. **High (0.92) is a high-efficiency sensitivity case**, an upper-range value within the commonly-cited 85–95% range for modern lithium-ion systems, not itself independently sourced to one specific study. |
| Degradation cost per MWh throughput `c` | **EUR 10** | EUR 5 | EUR 15 | Approximates the $7–15/MWh throughput marginal-degradation-cost range reported by He et al. (2021). Source figures are USD; treated as approximately EUR-equivalent given typical recent USD/EUR exchange rates — a rough cross-currency approximation, not claimed more precise than the underlying degradation-cost estimate itself. The underlying literature also notes optimal degradation cost can be time-varying and context-dependent — these three values are **pre-specified sensitivity assumptions informed by the literature, not claimed physical constants.** |

**Sensitivity protocol, frozen now, not left implicit**: the primary reported result
uses `η_rt = 0.85`, `c = EUR 10` (base/base). The full `3 x 3 = 9`-combination factorial
grid is computed and reported as a robustness table — **no combination in that grid is
ever used for strategy selection, promotion, or choosing which result to report as
primary.** Nine combinations, computed once, alongside the primary result, not
cherry-picked afterward.

| Element | Frozen definition |
|---|---|
| Execution/settlement price | Realized DE-LU day-ahead clearing price (hourly pre-cutover; equal-quarter-hour basket post-cutover, see above) — **the same target price this whole project has been forecasting**, not a fabricated separate execution price. |
| Forecast used for execution? | **No.** Forecast determines the `(i, j)` pair only. P&L is calculated exclusively from realized prices. |
| Market impact | Excluded — the operator is assumed small/price-taking. Explicit limitation, not silently ignored. |
| Bid acceptance | Assumed executed at the clearing price under the stated price-taking assumption. Explicit limitation. |
| Negative prices | **Not clamped, no special-casing.** Real, valid outcomes already observed in this project's data (e.g. the EUR -500/MWh 2023-07-02 hour) — being paid to charge and later paid again to discharge is economically real in European day-ahead markets, and the P&L formula already handles it correctly. |
| Development window | `2023-01-01` through `2025-12-31` (`local_delivery_date_to_utc` boundaries) — **not** the full 2019–2025 history. This is a hard constraint, not a simplification: `uncertainty_selected_v1`'s residual series only exists across this exact contiguous four-fold range (the earliest point any genuinely out-of-sample XGBoost prediction exists is `fold_1`'s validation start). Uncertainty-aware strategies cannot be evaluated before 2023-01-01, so **no strategy is evaluated before that date either** — comparing a point strategy on a longer window than an uncertainty-aware one would violate the common-comparison-day requirement below. |
| 2026 | Prohibited. No strategy code may read timestamps `>= local_delivery_date_to_utc("2026-01-01")`. |

## P&L formula, frozen before any result is seen

For delivery day `D` with candidate charge hour `i` and discharge hour `j` (`i < j`,
both in `H_D`), and total degradation cost `C(η_rt, c) = c * (1 + η_rt)`:

**Decision (uses only forecasts, made D-1 at 11:45):**

```
Point-forecast strategies:
  (i*, j*) = argmax over i<j in H_D of [ η_rt * P_hat_j - P_hat_i - C ]
  position = trade if that maximum > 0, else no_trade

Uncertainty-aware strategies:
  L_t = P_hat_t + q_offset_0.10,t     (lower price bound: point forecast PLUS the
                                        frozen 10th-percentile residual offset)
  U_t = P_hat_t + q_offset_0.90,t     (upper price bound: point forecast PLUS the
                                        frozen 90th-percentile residual offset)
  S_conservative(i,j) = η_rt * L_j - U_i - C
    (pessimistic bounds: worst plausible sell price, worst plausible buy price --
    reuses the frozen q10/q90 OFFSETS directly, no new tunable "uncertainty aversion"
    coefficient)
  (i*, j*) = argmax over i<j in H_D of S_conservative(i,j)
  position = trade if that maximum > 0, else no_trade
```

**Explicit, unambiguous naming, because this is exactly the kind of gap a real
implementation bug hides in.** `compute_rolling_residual_quantiles()` and
`latest_residual_quantile_offsets()` (`src/uncertainty.py`) both return *residual
offsets* (small values, e.g. ±5–30 EUR/MWh) — not absolute price bounds. `L_t` and `U_t`
above are explicitly defined as the point forecast *plus* that offset; `S_conservative`
is written entirely in terms of `L_t`/`U_t`, never the raw offset. A superficially
plausible but catastrophically wrong implementation would use the raw offset directly
(`η_rt * q_offset_0.10,j - q_offset_0.90,i`, without adding the point forecast) — small
numbers that look like they could be prices but aren't. This gets its own required test
below (test 9).

**`S_conservative` is a decision score, not a calibrated confidence interval for the
spread.** `q_offset_0.10` and `q_offset_0.90` are *marginal* per-hour forecast-error
offsets; the forecast errors at the buy hour and sell hour are very plausibly correlated
(adjacent hours, same underlying model, same day's conditions), so `L_j`/`U_i`'s
difference does not have a straightforward jointly-calibrated 80% probability
interpretation. It remains a useful, parameter-free, economically conservative rule — it
is just not claimed to be more statistically precise than it is. Building a joint
spread-distribution model to "fix" this is explicitly out of scope for `v1` — it would
defeat the point of keeping
this layer simple.

**Realized P&L, gross and net, both reported (uses only actual settled prices, never
forecasts):**

```
if no_trade:
    Π_D_gross = 0
    Π_D_net   = 0
if trade:
    Π_D_gross = η_rt * P_actual_j* - P_actual_i*
    Π_D_net   = Π_D_gross - C(η_rt, c)
```

Reporting both separates "was there no real arbitrage opportunity" from "an opportunity
existed but degradation cost erased it." Daily net P&L (`Π_D_net`, generally written
`Π_D` elsewhere in this document) is the principal economic/risk series; hourly detail
exists only inside the two-leg trade itself. This naturally handles 23/24/25-hour DST
days without special casing, the same way `add_local_time_columns` already does
everywhere else in this project.

## Pre-registered strategy candidates — exactly these six, no more

An earlier draft bundled a Tier-1 point rule and a Tier-1 uncertainty-aware rule into
one "S4" — but these are genuinely different decision rules that can select different
`(i, j)` pairs on different days. Splitting them out is not complexity creep; it makes
an already-implied robustness comparison explicit, with zero new parameters.

| Strategy | Point forecast source | Decision rule | Purpose |
|---|---|---|---|
| **S0** | none | never trade, `Π = 0` always | Absolute floor — is *any* operation better than doing nothing |
| **S1** | `lag_24` (naive) | point-forecast rule above | **The real forecast-value benchmark.** Isolates the arbitrage value already present in typical diurnal price shape from anything XGBoost specifically contributes |
| **S2** | `xgboost_full_pred` | point-forecast rule above | Primary frozen point-forecast strategy |
| **S3** | `xgboost_full_pred` + `uncertainty_selected_v1` | uncertainty-aware rule above | Does the frozen 60-day uncertainty layer add tradeable value beyond the point forecast alone |
| **S4** | `xgboost_tier1_pred` | point-forecast rule above, **identical** `η_rt`/`c` to S2, no separate tuning | Tier-1 information-set robustness, point-forecast side |
| **S5** | `xgboost_tier1_pred` + Tier-1 `uncertainty_selected_v1` analog | uncertainty-aware rule above, **identical** `η_rt`/`c` to S3, no separate tuning | Tier-1 information-set robustness, uncertainty-aware side |

**Exactly six. No additional strategies without a new, explicitly labeled
pre-registration.**

**Named, pre-registered economic comparisons** — these are the actual reported results,
not raw P&L in isolation:

```
ΔΠ_forecast      = Π_S2 - Π_S1     (value of XGBoost over naive lag-24 -- PRIMARY result)
ΔΠ_uncertainty   = Π_S3 - Π_S2     (value of uncertainty-aware trade selection)
ΔΠ_Tier1         = Π_S4 - Π_S2     (cost of the more defensible information set, point side)
ΔΠ_Tier1,U       = Π_S5 - Π_S3     (cost of the more defensible information set, uncertainty side)
```

A storage asset earns some arbitrage value purely from the predictable diurnal shape of
day-ahead prices — `lag_24` alone will likely capture a meaningful share of that.
Reporting `Π_S2` alone without subtracting `Π_S1` would overstate what the forecasting
model specifically contributes.

## Ex-post oracle — diagnostic only, architecturally unreachable from live strategy code

```
Π_D_oracle = max( 0, max over i<j in H_D of [ η_rt * P_actual_j - P_actual_i - C ] )
```

The `max(0, ...)` floor matters: without it, a perfect-foresight benchmark could be
forced to report a loss on a day where every possible `(i, j)` pair is unprofitable —
nonsensical for an upper bound, since a perfect-foresight agent would simply choose not
to trade that day, exactly like every executable strategy above already can.

Uses **realized** prices to choose `(i, j)` — this is not a strategy, it can never be
executed D-1, and it must never influence any strategy's decision or any model-selection
step. Reported only as:

```
Value Capture Ratio (strategy s) = Sum_D Π_D_s / Sum_D Π_D_oracle
```

summed over the **same common evaluation days** used for every other cross-strategy
comparison (see below) — **not** conditioned on whether strategy `s` itself chose to
trade that day. Conditioning the oracle's denominator on the strategy's own trading
days would recreate the unequal-comparison-population problem in a new disguise.

**Enforced architecturally, not just by convention**: the oracle computation lives in a
module/function namespace that live strategy-generation code cannot import from (e.g. a
separate `oracle.py` never imported by `strategy.py`), and every report referencing it
is labeled **"ex-post oracle diagnostic, not an executable strategy"** — never bare
"performance."

**Dominance property, verified (not just asserted) before being relied on**: for every
executable strategy `s`, every day `D`, at the same `(η_rt, c)`,
`Π_D_s <= Π_D_oracle` must always hold — the oracle searches every valid `(i, j)` pair
(plus no-trade) using the same realized prices any strategy's own realized P&L is
computed from, so no strategy can ever beat it on its own terms. This was checked
numerically (20,000 randomized scenarios, including deliberately suboptimal strategy
choices) before being written down as a guaranteed invariant, not assumed from the
formula alone. A corollary: `VCR_s <= 1` whenever the oracle-sum denominator is
positive (a strategy can still show negative VCR by losing money; it cannot exceed 1).
This becomes test 8 below — one of the strongest, cheapest checks available, since a
violation implicates pair construction, cost application, efficiency scaling,
chronology, or the oracle itself.

## Required tests, built before any P&L is computed

1. **Actual-price perturbation (leakage).** Corrupt every realized price on day `D` to
   an absurd value. The recomputed `(i*, j*)` for `D` must be bit-identical to the
   original — decisions come from forecasts, never from realized outcomes. Recomputed
   `Π_D` must differ (P&L legitimately depends on realized prices; the *decision* must
   not).
2. **Future-day perturbation.** Corrupt forecasts and/or actual prices on `D+1`. The
   decision for `D` must be unchanged — no day's schedule may depend on a later day's
   information.
3. **No reverse-time cycle.** `j* > i*` whenever a trade is made — always, no
   exceptions.
4. **No forced trade.** If every candidate `(i, j)` pair's predicted net spread is
   `<= 0`, `position = no_trade`. The strategy must never be forced to open a position
   it predicts will lose money.
5. **Oracle unreachability.** A static/import-level check that no strategy-generation
   code path can call the oracle function — this is an architecture test, not a
   numerical one.
6. **Oracle floor.** A day where every `(i, j)` pair is unprofitable under realized
   prices must produce `Π_D_oracle = 0`, never a negative value.
7. **Common comparison days.** S0–S5 (and the oracle) are compared only on delivery
   days where every required input (forecast for all six variants, `uncertainty_selected_v1`
   coverage — Full and Tier-1 — for S3/S5) is available — a `common_strategy_evaluation_days`
   set, built the same way as `common_comparison_rows` (point models) and
   `find_common_evaluation_start` (uncertainty windows). This exact class of bug has
   recurred twice already in this project; this is the third layer it must be prevented
   in, not assumed solved because it was solved before.
8. **Oracle dominance.** For every strategy `s` and every day `D`, `Π_D_s <= Π_D_oracle`
   at the same `(η_rt, c)` — verified numerically above (20,000 randomized trials, zero
   violations) before being trusted as an invariant. A violation anywhere implicates
   pair construction, cost application, efficiency scaling, chronology enforcement, or
   the oracle calculation itself — one of the highest-value, cheapest tests available.
9. **Q-bound semantics.** `L_t`/`U_t` must equal `P̂_t + q_offset` — not the raw offset
   alone. A dedicated test constructs a case with known point forecast and known offset
   and asserts the strategy score is computed from the summed price bound, catching
   exactly the "superficially plausible but wrong" bug described above.
10. **15-minute basket quantities.** Post-cutover, assert the charge leg sums to
    exactly 1 MWh across its four quarter-hours and the discharge leg sums to exactly
    `η_rt` MWh — not 1 MWh on both legs.
11. **Holdout rejection.** Any code path attempting to read a timestamp
    `>= local_delivery_date_to_utc("2026-01-01")` raises, the same class of guard
    already required elsewhere in this project's holdout-protection discipline.

## Reporting requirements

Report at both **daily** (`Π_D`, the principal series) and, for diagnosis only,
**hourly** (`Π_D` decomposed into its two legs) resolution, at `η_rt = 0.85, c = EUR 10`
(primary) plus the full `3x3` sensitivity grid (robustness, never used for selection).
At minimum, for every strategy:

| Category | Measures |
|---|---|
| Economics | Gross P&L, net P&L (both, per the formula above), the four named `ΔΠ` comparisons (primary results) |
| Activity | Trading days vs. no-trade days, MWh cycled |
| Reliability | **Two distinct rates, not one**: `trade hit rate = #{trades with Π_D > 0} / #{days the strategy traded}` (conditional success, given it acted) and `profitable-day rate = #{D: Π_D > 0} / #{common evaluation days}` (unconditional, reflects selectivity too). A strategy that trades rarely but wins most of those trades can look deceptively strong on hit rate alone while abstaining most days — report both, always together. **If a strategy trades zero days under some sensitivity scenario, trade hit rate is reported as `N/A`, never `0%`** — a `0/0` ratio is undefined, not evidence of failure; profitable-day rate remains a well-defined `0%` in that case since its denominator is the common evaluation-day count, not the trade count. |
| Distribution | Mean, median, SD of `Π_D` |
| Downside | Worst day, worst 5 days |
| Drawdown | Maximum drawdown (on cumulative `Π_D`) |
| Concentration | % of total P&L from best 1%/5%/10% of days — **reported only when total net P&L > 0; otherwise `N/A`, plus the absolute EUR contribution of the best 1%/5%/10% days.** A percentage-of-total is meaningless or actively misleading when the total is zero or negative. |
| Regime | Pre- vs. post-2025-10-01 breakdown |
| Stress | **Negative-price day**: any delivery interval on day `D` has actual price `< 0`. **`>200`/`>500` day**: any delivery interval on `D` exceeds that threshold. Defined by the day's realized *market regime*, independent of which interval (if any) the strategy actually selected — this identifies whether the day was economically extreme, separately from whether the strategy's chosen leg happened to land on the extreme interval, which can be diagnosed separately if needed. |
| Information | Full (`S2`/`S3`) vs. Tier-1 (`S4`/`S5`) |
| Costs | Full `3x3` `(η_rt, c)` grid, reported separately — **never lower the cost assumption after seeing an unprofitable result** |
| Oracle | Value capture ratio, explicitly labeled diagnostic |

**Exclusion/coverage report, same discipline as the point-model and uncertainty
layers**: before reporting any comparison, show

```
raw delivery days
days Full forecast available
days Tier-1 forecast available
days lag-24 available
days uncertainty (Full) available
days uncertainty (Tier-1) available
days excluded for incomplete intraday vector
common evaluation days
```

with reason codes for exclusions (DST-related incompleteness, missing forecasts, etc.)
— common population isn't enough on its own; show what was excluded and why, the same
way `run_models.py`'s coverage report already does for the point-forecast stage.

Do not rely on Sharpe or similar risk-adjusted return metrics until capital/exposure is
explicitly defined — with a fixed 1 MWh nominal position and no capital-sizing decision
in `v1`, such a metric doesn't yet have a well-defined denominator.

## Explicitly deferred to a later, separately-frozen document

- Continuous multi-hour/multi-cycle battery dispatch (`v2`, only if `v1` motivates a
  genuinely new question)
- Any tunable "uncertainty aversion" coefficient beyond the parameter-free `q10`/`q90`
  conservative rule above, and any joint/multivariate spread-distribution model
- Battery physical-parameter sensitivity beyond the pre-registered `3x3`
  efficiency/degradation-cost grid
- Market-access cost (`C_market`) sourcing and inclusion — `v1` sets it to zero with an
  explicit limitation, not silently
- Tail-risk / VaR / ES on the resulting `Π_D` series — this is `L_D = -Π_D`, built
  *after* this layer produces a real loss series, not on price-forecast residuals
  (`uncertainty_selected_v1` is a different quantity and must not be relabeled as risk
  on strategy P&L; see README "Uncertainty quantification" scope-boundary note)
