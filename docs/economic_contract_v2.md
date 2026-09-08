# Economic Contract v2 — Delivery-Day Forecast-Origin Correction

## Status

This document supersedes `economic_contract_v1.md` only with respect to the information set used to construct empirical-residual uncertainty bounds.

The economic decision rules, strategy definitions, point forecasts, cost assumptions, robustness grid, and economic evaluation methodology remain unchanged.

`economic_contract_v1.md` is preserved unchanged as a historical record of the original preregistration and development analysis.

This correction is specified and frozen before inspecting any corrected development uncertainty, strategy, decomposition, or tail-risk results, and before any 2026 holdout performance or economic result is evaluated.

---

## 1. Reason for the correction

The original development uncertainty implementation constructed rolling empirical residual quantiles separately at each hourly timestamp.

That implementation is valid for a one-step-ahead hourly forecasting problem, because each forecast at time \(t\) may use information available strictly before \(t\).

It is not aligned with this project's economic decision problem.

For delivery day \(D\), the complete day-ahead schedule is assumed to be formed at the common forecast origin of D-1 11:45 Europe/Berlin. Therefore no realized price or forecast residual from any hour of delivery day \(D\) is available when any decision for \(D\) is made.

Under the corrected information set, all uncertainty bounds for delivery day \(D\) must therefore be based exclusively on residual information from delivery dates strictly earlier than \(D\).

This is a forecast-origin correction, not a post-result change to the economic strategy.

---

## 2. Components that remain frozen and unchanged

The following elements are not changed by this correction:

* Target: DE-LU day-ahead hourly electricity price.
* Frozen point-forecast development run: `xgboost_v1_a03fix`.
* XGBoost Full point forecasts.
* XGBoost Tier-1 point forecasts.
* Lag-24 benchmark forecasts.
* S0-S5 strategy definitions.
* Single-cycle intraday storage scheduling rule.
* Requirement that charging precedes discharge: i < j.
* Round-trip efficiency convention.
* Base round-trip efficiency: eta_rt = 0.85
* Base degradation-cost parameter: c = 10
* Market transaction-cost assumption: C_market = 0
* Degradation-cost definition: C_degradation = c(1 + eta_rt)
* Total cost: C = C_degradation + C_market
* Point-strategy score: eta_rt * P_hat_j - P_hat_i - C
* Realized P&L: Pi_D = eta_rt * P_actual_j - P_actual_i - C
* Primary robustness grid: eta_rt in {0.70, 0.85, 0.92} and c in {5, 10, 15}, all nine combinations.
* Economic oracle definition and oracle-dominance validation.
* Empirical VaR/ES methodology.
* Development/holdout separation.
* 2026 remains excluded from development model selection, uncertainty selection, strategy evaluation, and economic interpretation.

No point model is retuned as part of this correction.

---

## 3. Corrected uncertainty information set

For every delivery day D, define a single residual-information snapshot using observations whose local delivery date satisfies:

```
delivery_date < D
```

No residual from delivery day D may influence any uncertainty bound for any hour of D.

All hours belonging to the same delivery day use the same empirical residual-quantile offsets.

If the point prediction for hour h of delivery day D is P_hat(D,h), the uncertainty bound at residual quantile q is:

```
Q(D,h,q) = P_hat(D,h) + r_D(q)
```

where r_D(q) is calculated once for delivery day D from eligible prior-day residual history.

Thus absolute hourly uncertainty bounds may differ because hourly point forecasts differ, but the residual offset applied within a delivery day is identical.

Delivery days are defined using Europe/Berlin local calendar dates. The procedure therefore naturally accommodates 23-hour and 25-hour DST delivery days and does not assume that every delivery day contains 24 observations.

---

## 4. Frozen quantiles

The empirical residual quantiles remain: q = {0.10, 0.50, 0.90}

The nominal central interval remains 80%.

No additional quantiles may be introduced as part of the correction.

---

## 5. Frozen window sensitivity experiment

The uncertainty-window candidate set remains exactly: [60, 90, 120, 180, 365]

No candidate may be added, removed, or replaced after corrected results are observed.

For a candidate window of w delivery days:

```
min_periods_days = max(1, floor(w / 4))
```

The comparison must use the corrected delivery-day-safe information set for every candidate.

---

## 6. Common-row comparison rule

Window selection must be performed on an identical evaluation population for every candidate.

The common evaluation start is the latest valid warm-up boundary required by any candidate.

Only rows available to every candidate from that point onward enter the selection comparison.

Available-row results may be reported diagnostically but must not determine the winner.

---

## 7. Mechanical selection rule

The primary window-selection criterion remains the empirical interval/Winkler score calculated on the common-row table.

Lower is better.

The selected specification is:

> the candidate with the lowest common-row interval score.

The fold-level calibration diagnostic remains a reporting requirement.

A fold is flagged when empirical coverage differs from nominal coverage by more than 3 percentage points.

The calibration flag is diagnostic only.

It cannot override, veto, promote, or otherwise alter the mechanically selected window.

If two candidates have exactly equal interval scores, the smaller `window_days` value wins.

No discretionary post-result selection is permitted.

---

## 8. Selected Full uncertainty run

After the sensitivity experiment finishes, the consequential Full uncertainty run must obtain the selected window directly from the saved sensitivity-run manifest.

The selected window must not be manually retyped.

The selected uncertainty run must verify that:

* its parent sensitivity run used `information_set_method = "delivery_day_safe"`;
* its parent sensitivity run used the requested frozen XGBoost run;
* its quantiles are inherited from that sensitivity run;
* its selected window equals the mechanically recorded sensitivity winner.

The resulting manifest must preserve that lineage.

---

## 9. Tier-1 uncertainty robustness

Tier-1 uncertainty is a robustness evaluation, not an independent model-selection experiment.

Tier-1 must inherit the Full-selected uncertainty specification unchanged:

* same `window_days`;
* same `min_periods_days`;
* same quantiles;
* same delivery-day-safe information-set method.

No Tier-1-specific retuning or second window-selection experiment is permitted.

The Tier-1 runner must explicitly name the corrected Full uncertainty parent and verify that both use the same frozen XGBoost development run.

---

## 10. Uncertainty-aware economic strategies

The uncertainty-aware strategy rule remains unchanged.

For lower and upper price bounds L and U:

```
Score_U(i,j) = eta_rt * L_j - U_i - C
```

The strategy selects:

```
(i*, j*) = argmax over i<j of Score_U(i,j)
```

and abstains when the maximum score is less than or equal to zero.

The information-set correction changes only how the empirical residual offsets underlying L and U are estimated.

It does not change the economic rule itself.

---

## 11. Strategy definitions

The frozen strategies remain:

* **S0:** no trade.
* **S1:** lag-24 point forecast.
* **S2:** XGBoost Full point forecast.
* **S3:** XGBoost Full point forecast plus corrected Full uncertainty bounds.
* **S4:** XGBoost Tier-1 point forecast.
* **S5:** XGBoost Tier-1 point forecast plus corrected Tier-1 uncertainty bounds.

The primary economic contrasts remain:

```
Delta_forecast    = S2 - S1
Delta_uncertainty = S3 - S2
Delta_Tier1       = S4 - S2
Delta_Tier1,U     = S5 - S3
```

---

## 12. Corrected development rerun scope

The frozen point forecasts themselves are not regenerated.

The following uncertainty-dependent evidence must be regenerated under the corrected information set:

* uncertainty window sensitivity;
* selected Full uncertainty bounds;
* Tier-1 uncertainty robustness;
* Tier-1 regime diagnostic;
* common strategy evaluation population;
* strategy backtest;
* 3x3 strategy sensitivity;
* strategy decomposition;
* strategy report;
* empirical strategy tail-risk summaries.

S1, S2, and S4 point forecasts remain unchanged.

Their aggregate economic totals may nevertheless change if the corrected uncertainty warm-up changes the common strategy evaluation-day population.

Such a difference is an evaluation-population effect, not a point-model change.

---

## 13. Historical evidence preservation

Original pre-correction outputs must not be overwritten or deleted.

They remain part of the project's historical audit trail.

Corrected artifacts must use new version names.

Pre-correction uncertainty-dependent results are to be described as superseded for delivery-day forecast-origin alignment, not retroactively erased.

---

## 14. Holdout prohibition

No 2026 observation may be used by the corrected development uncertainty or strategy runners.

Development runners must fail closed if timestamps belonging to the 2026 holdout are encountered.

The correction must be fully implemented, tested, committed, and frozen before inspecting any corrected real development result.

The final 2026 holdout remains a separate evaluation event governed by the separately frozen holdout protocol.

---

## 15. Interpretation discipline

The corrected window sensitivity experiment is a replay of the already-defined uncertainty method under the correct information set.

The candidate set, scoring criterion, quantiles, minimum-history rule, economic rules, and point models are not changed in response to corrected performance.

Whatever window wins mechanically under the rule above is accepted.

Subsequent economic results are reported as corrected development evidence and are not used to redesign the frozen methodology before the 2026 holdout.

---

## 16. Status at freeze

At the time this contract is frozen:

* the correction has been motivated by an information-set audit;
* implementation and hostile tests have been completed;
* the corrected real window-sensitivity result has not yet been inspected;
* no corrected Full uncertainty winner has yet been promoted;
* no corrected Tier-1 uncertainty result has yet been interpreted;
* no corrected uncertainty-aware strategy result has yet been interpreted;
* no 2026 holdout performance or economic result has been used.

This chronology is intentional and is part of the project's reproducibility record.

---

## 17. Verification note (added before commit, not part of the original preregistration text)

Every mechanical rule stated above was checked directly against the actual implementation before this document was accepted as accurate, not assumed correct from being well-written:

* Section 5's `min_periods_days = max(1, floor(w/4))` matches `run_uncertainty.py`'s `max(1, window_days // 4)` exactly.
* Section 7's tie-break rule matches `run_uncertainty_sensitivity.py`'s `sort_values(["interval_score", "window_days"], kind="mergesort")` exactly.
* Section 8's lineage requirements match `run_uncertainty.py`'s `load_sensitivity_winner()`, which reads `lowest_interval_score_window_days` directly from the sensitivity manifest and raises on an `xgboost_run_version` or `information_set_method` mismatch.
* Section 9's Tier-1 lineage requirement matches `run_uncertainty_tier1_robustness.py`'s `load_frozen_uncertainty_spec()`, which raises if the parent's `xgboost_run_version` doesn't match the one requested.

`economic_contract_v1.md` was not modified in the process of writing or freezing this document.
