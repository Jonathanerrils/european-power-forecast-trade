# European Day-Ahead Power Forecast-to-Trade & Tail-Risk System

An end-to-end research pipeline for **Germany-Luxembourg (DE-LU) day-ahead electricity prices**, from ENTSO-E ingestion and leakage-safe forecasting through uncertainty quantification, a stylized forecast-to-trade layer, tail-risk analysis, and a single frozen 2026 holdout.

## Current status

**The project is complete through the frozen 2026 holdout.** Forecasting, uncertainty, strategy, sensitivity, holdout, and tail-risk layers have all been implemented and executed. The 2026 holdout was opened under the pre-specified rules in [\`docs/holdout_protocol_v2.md\`](docs/holdout_protocol_v2.md) and the resulting evidence is preserved in versioned output directories.

The most important unresolved methodological limitation remains unchanged: **wind/solar-derived Tier-2 features cannot be proven point-in-time available at the simulated 11:45 D-1 decision cutoff from the historical ENTSO-E API alone.** Tier-1 results therefore remain the information-availability robustness check.

| Layer | Current state | Main evidence |
|---|---|---|
| ENTSO-E ingestion + cleaning | Complete | \`src/entsoe_client.py\`, \`src/clean.py\` |
| Leakage-safe feature engineering | Complete | \`src/features.py\` |
| Versioned EDA | Complete | \`development_a03fix_v1\` |
| ElasticNet benchmark | Frozen | \`baseline_v1_a03fix\` |
| XGBoost challenger | Promoted on corrected development data | \`xgboost_v1_a03fix\` |
| Uncertainty | Frozen, delivery-day-safe, 60-day residual window | \`uncertainty_selected_v2_dayorigin\` |
| Economic strategy | Complete | \`strategy_report_v2_dayorigin\` |
| Development tail risk | Complete | \`strategy_tailrisk_v2_dayorigin\` |
| Final 2026 holdout | Exposed once and preserved | \`holdout_v1\`, \`holdout_report_v1\` |
| Holdout sensitivity | Complete | \`holdout_sensitivity_v1\` |
| Holdout tail risk | Complete | \`holdout_tailrisk_v1\` |

README refreshed: **2026-09-18**.

## Research question

> Can publicly available day-ahead fundamentals improve hourly DE-LU electricity-price forecasts over strong naive benchmarks, and does any forecasting improvement translate into economically useful decisions after forecast uncertainty, execution assumptions, and tail risk are considered?

## Market and decision setup

- **Market:** Germany-Luxembourg (DE-LU) day-ahead bidding zone.
- **Sample start:** 2019-01-01, after the DE/AT bidding-zone split.
- **Storage timezone:** UTC; local delivery-day logic: Europe/Berlin.
- **Simulated decision cutoff:** 11:45 Europe/Berlin on D-1.
- **2025-10-01 market-design change:** Single Day-Ahead Coupling moved to 15-minute market time units. The project remains hourly and averages the four quarter-hours into an hourly price after the cutover.
- **Development period:** 2019-01-01 through 2025-12-31.
- **Frozen holdout:** 2026-01-01 inclusive through 2026-08-01 exclusive, i.e. delivery dates through 2026-07-31.

The target is the DE-LU day-ahead price. The project uses price lags, calendar structure, load forecasts, wind/solar forecasts, residual-load variables, and related engineered predictors subject to the information-set rules in \`src/features.py\`.

## Final 2026 holdout results

The final holdout contains **5,087 identical point-evaluation rows across 212 delivery days**. The final model was frozen before outcome exposure; the protocol commit is \`a2d1f9a79022ca1e2731d52803061825038c69f5\` and the final model-fit commit is \`54a7c2e77122134eca31a09798cc1614df1c634c\`.

### Point-forecast performance

| Model | MAE (EUR/MWh) | RMSE | Median AE | n |
|---|---:|---:|---:|---:|
| Lag-24 | 29.329 | 46.855 | 17.063 | 5,087 |
| Lag-168 | 36.014 | 56.598 | 22.938 | 5,087 |
| XGBoost Full | **17.512** | **29.869** | **11.992** | 5,087 |
| XGBoost Tier-1 | 24.361 | 38.414 | 15.770 | 5,087 |

The Full XGBoost model therefore retained a large point-forecast advantage on the untouched 2026 holdout. The Tier-1 model also beat the naive benchmarks, but the gap between Full and Tier-1 remains material, which reinforces the importance of the unresolved Tier-2 information-timing limitation.

### Frozen strategy definitions

The economic layer is a **stylized, price-taking, single-cycle daily storage scheduling problem**, not a production trading engine.

- **S0:** no trade.
- **S1:** Lag-24 point forecast.
- **S2:** XGBoost Full point forecast.
- **S3:** XGBoost Full + delivery-day-safe uncertainty bounds.
- **S4:** XGBoost Tier-1 point forecast.
- **S5:** XGBoost Tier-1 + delivery-day-safe uncertainty bounds.

The primary economic parameters were frozen before the holdout:

- round-trip efficiency \(\eta_{rt}=0.85\);
- degradation parameter \(c=10\);
- market transaction-cost term \(C_{market}=0\);
- one charge/discharge pair per delivery day, with charge preceding discharge.

For a point forecast, the selected pair maximizes:

\[
\eta_{rt}\hat P_j-\hat P_i-C
\]

and realized daily P&L is:

\[
\Pi_D=\eta_{rt}P_j-P_i-C.
\]

See [\`docs/economic_contract_v2.md\`](docs/economic_contract_v2.md) for the exact frozen contract.

### Holdout economics

| Strategy | Trading days | Net P&L (EUR) | Profitable-day rate | Trade hit rate | Worst day (EUR) |
|---|---:|---:|---:|---:|---:|
| S0 | 0 | 0.00 | 0.0% | N/A | 0.00 |
| S1 | 208 | 18,564.42 | 84.0% | 85.6% | -67.28 |
| S2 | 210 | **20,002.02** | **91.0%** | 91.9% | -17.55 |
| S3 | 160 | 19,130.01 | 75.0% | **99.4%** | **-0.83** |
| S4 | 205 | 19,218.41 | 86.8% | 89.8% | -31.14 |
| S5 | 117 | 14,721.23 | 54.7% | 99.1% | -5.98 |

The four pre-specified economic contrasts at the primary parameter setting were:

- **Forecast value:** S2 - S1 = **+EUR 1,437.60**.
- **Uncertainty value:** S3 - S2 = **-EUR 872.01**.
- **Tier-1 cost:** S4 - S2 = **-EUR 783.61**.
- **Tier-1 uncertainty cost:** S5 - S3 = **-EUR 4,408.77**.

The forecast-value result was positive in **all 9/9 cells** of the frozen \(3\times3\) sensitivity grid over \(\eta_{rt}\in\{0.70,0.85,0.92\}\) and \(c\in\{5,10,15\}\).

The uncertainty-aware Full strategy illustrates an important trade-off rather than an automatic improvement: S3 gave up EUR 872.01 of total net P&L relative to S2, but its worst day improved from EUR -17.55 to EUR -0.83 and its trade hit rate rose from 91.9% to 99.4%. The decomposition attributes the S2→S3 gap to **EUR 952.46 of profits forgone versus EUR 80.44 of losses avoided**.

These are results under the frozen research contract. They are **not claims about investable return, Sharpe ratio, capital efficiency, market impact, or deployable trading profitability**.

## Tail-risk layer

Daily loss is defined as:

\[
L_{D,s}=-\Pi_{D,s}.
\]

The project reports empirical historical VaR/ES at 95% and 99% for S0-S5. On the 212-day holdout, tail estimates are explicitly flagged as low precision: the nominal tail counts are only about 10.6 observations at 95% and 2.12 at 99%.

For S2, the holdout estimates are:

| Confidence | VaR (EUR) | ES (EUR) |
|---|---:|---:|
| 95% | 2.07 | 7.07 |
| 99% | 8.35 | 13.45 |

The full table is preserved in [\`outputs/risk/delu_features/holdout_tailrisk_v1/var_es_results.csv\`](outputs/risk/delu_features/holdout_tailrisk_v1/var_es_results.csv).

## Development validation architecture

Model selection was chronological and fixed before final holdout exposure.

| Fold | Training window | Validation window |
|---|---|---|
| fold_1 | 2019 → 2022 | 2023 |
| fold_2 | 2019 → 2023 | 2024 |
| fold_3 | 2019 → 2024 | Jan-Sep 2025 |
| regime_stress_test | 2019 → Sep 2025 | Oct-Dec 2025 |

The final Full and Tier-1 XGBoost models were then independently refit on all development data through 2025-12-31 using the already-frozen model-selection procedure.

On the corrected development data, \`xgboost_v1_a03fix\` re-established the XGBoost promotion result: the Full model improved row-weighted MAE by **17.2% versus the frozen ElasticNet Full benchmark**.

## Uncertainty quantification

The selected uncertainty method uses empirical residual quantiles \(q=\{0.10,0.50,0.90\}\), giving a nominal central 80% interval.

A pre-registered sensitivity experiment compared residual windows of:

\[
[60,\ 90,\ 120,\ 180,\ 365] \text{ days}
\]

on a common evaluation population using interval/Winkler score. The mechanically selected window was **60 days**.

The original hourly residual implementation was later corrected to match the actual D-1 whole-day decision problem. Under the current method, every hour of delivery day \(D\) uses residual information from delivery dates strictly earlier than \(D\). No realized residual from any hour of \(D\) can affect an uncertainty bound used to make a decision for \(D\).

Authoritative corrected artifacts:

- \`uncertainty_window_sensitivity_v3_dayorigin\`
- \`uncertainty_selected_v2_dayorigin\`
- \`strategy_backtest_v3_dayorigin\`
- \`strategy_report_v2_dayorigin\`
- \`strategy_tailrisk_v2_dayorigin\`

## Information-availability tiers

The information set is deliberately split into two tiers.

**Tier 1** contains variables whose regulatory publication timing is compatible with the 11:45 D-1 cutoff, most importantly the day-ahead load forecast plus price-state and calendar information.

**Tier 2** contains wind/solar forecasts and variables derived from them, including residual load and renewable share. Commission Regulation (EU) No 543/2013 does not establish that these forecasts had to be published by 11:45 D-1, and the historical ENTSO-E API does not provide the publication-vintage timestamps needed to prove exact historical availability.

This does **not** prove that Tier-2 features leak. It means the project cannot prove that they do not. The Full/Tier-1 comparison therefore remains a central robustness analysis rather than a cosmetic sensitivity check.

## Auction-sequence validation

Post-2025-10-01 ENTSO-E price responses contain two sequential auction runs for the same PT15M intervals. The production pipeline keeps \`classificationSequence == 1\`.

That choice is no longer an unsupported assumption. It was independently cross-checked against SMARD over the corrected data: among **8,833 intervals where sequence 1 and sequence 2 disagreed materially, SMARD was consistent with sequence 1 in 100% of cases and sequence 2 in 0%**.

The structural audit also showed that sequence 1 is PT60M before the 15-minute SDAC cutover and PT15M after it, while sequence 2 is PT15M throughout.

## Data and pipeline integrity

This project keeps a record of bugs that materially changed the methodology rather than deleting them from the narrative. Important corrections include:

1. Selecting the correct parallel price product instead of silently mixing PT60M and PT15M products.
2. Anchoring the 2025-10-01 market-design cutover to the Europe/Berlin delivery day, not UTC midnight.
3. Explicitly selecting the verified primary auction sequence.
4. Aggregating load/wind/solar quarter-hour forecasts correctly across the full history.
5. Making 24-hour lag logic DST-safe on 23-hour and 25-hour local days.
6. Fixing ingestion logs that previously appended stale records across reruns.
7. Converting configured sample dates to local-delivery-day UTC boundaries and clipping the final dataset explicitly.
8. Replacing name-based rolling-feature leakage inference with explicit provenance.
9. Preventing missing wind components from being converted to false zeros.
10. Rejecting duplicate wind/solar submissions before pivoting instead of silently averaging them.
11. Correcting the A03 parser issue and reproducing the main development results on corrected data.
12. Correcting uncertainty from an hourly-information-set formulation to a delivery-day-safe D-1 formulation.

Regression tests and repository checks were added around these failure modes where they can be mechanically tested.

## Repository layout

| Path | Purpose |
|---|---|
| \`config.yaml\` | Operational defaults and pointers to authoritative frozen specs |
| \`src/entsoe_client.py\` | ENTSO-E retrieval, price-product and auction-sequence handling |
| \`src/clean.py\` | UTC/local normalization, DST-safe aggregation, range clipping |
| \`src/features.py\` | Feature engineering, information tiers, leakage checks |
| \`src/splits.py\` | Chronological development folds and final-train window |
| \`src/models.py\` | Naive benchmarks + ElasticNet |
| \`src/xgboost_model.py\` | XGBoost training and delivery-day-aligned inner CV |
| \`src/uncertainty.py\` | Delivery-day-safe empirical residual intervals |
| \`src/strategy.py\` | Frozen economic decision rules |
| \`src/oracle.py\` | Oracle benchmark used for strategy diagnostics |
| \`run_*.py\` | Reproducible pipeline stages and frozen experiment runners |
| \`docs/economic_contract_v2.md\` | Authoritative corrected economic contract |
| \`docs/holdout_protocol_v2.md\` | Authoritative pre-holdout protocol |
| \`outputs/\` | Versioned experimental evidence |
| \`tests/\` | Regression and methodology tests |
| \`scripts/\` | Audits, repository checks, status generation, validation utilities |

## Setup and verification

Create a Python environment and install the frozen dependency set:

~~~bash
python -m venv .venv
# activate the environment for your platform
pip install -r requirements-lock.txt
~~~

Copy \`.env.example\` to \`.env\` and set your ENTSO-E token:

~~~text
ENTSOE_TOKEN=your_token_here
~~~

Run the tests and repository integrity checks:

~~~bash
python -m pytest tests/ -v
python scripts/run_all_checks.py
python scripts/generate_status.py
~~~

The main pipeline entry points are:

- \`run_ingestion.py\`
- \`run_features.py\`
- \`run_eda.py\`
- \`run_models.py\`
- \`run_xgboost.py\`
- \`run_uncertainty_sensitivity.py\`
- \`run_uncertainty.py\`
- \`run_strategy_backtest.py\`
- \`run_strategy_sensitivity.py\`
- \`run_strategy_decomposition.py\`
- \`run_strategy_report.py\`
- \`run_strategy_tailrisk.py\`
- \`run_final_model_fit.py\`
- \`run_final_holdout.py\`
- \`run_final_holdout_sensitivity.py\`
- \`run_final_holdout_report.py\`
- \`run_final_holdout_tailrisk.py\`

For consequential frozen runs, use the exact run versions and lineage documented in the manifests and protocol files rather than inventing a new directory name or silently changing parameters.

**Important:** the 2026 holdout has already been exposed. It must not be reused as a tuning or model-selection set. Any future experiment should be treated as a new study with a new pre-specified evaluation period.

## Key final evidence

- [Economic Contract v2](docs/economic_contract_v2.md)
- [Holdout Protocol v2](docs/holdout_protocol_v2.md)
- [Final holdout report](outputs/holdout/delu_features/holdout_report_v1/holdout_report.txt)
- [Holdout report manifest](outputs/holdout/delu_features/holdout_report_v1/holdout_report_manifest.json)
- [Point-forecast holdout metrics](outputs/holdout/delu_features/holdout_report_v1/point_forecast_metrics.csv)
- [Primary strategy table](outputs/holdout/delu_features/holdout_report_v1/primary_strategy_table.csv)
- [Frozen 3x3 sensitivity grid](outputs/holdout/delu_features/holdout_report_v1/sensitivity_grid.csv)
- [Holdout VaR/ES](outputs/risk/delu_features/holdout_tailrisk_v1/var_es_results.csv)

## Limitations

1. **Tier-2 publication vintage remains unresolved.** Wind/solar-derived predictors are economically important but not historically vintage-verified at 11:45 D-1.
2. **The economic layer is stylized.** It assumes a price-taking single-cycle storage schedule and no market impact.
3. **Market transaction costs are set to zero** in the frozen contract; degradation cost is modeled separately.
4. **There is no capital or exposure model.** The project therefore makes no ROI, Sharpe, annualized-return, leverage, or capital-efficiency claim.
5. **The final holdout covers only 2026-01-01 through 2026-07-31 delivery dates.**
6. **Holdout tail-risk estimates are sample-limited**, especially at 99%.
7. **Post-2025-10-01 quarter-hour prices are aggregated to hourly means.** The project does not model native 15-minute trading opportunities.
8. Historical public-market data can be revised; exact publication-vintage reconstruction is incomplete for some fundamentals.

## Future work

Any future extension should be treated as a new pre-specified experiment rather than a reinterpretation of the completed holdout. Natural next studies include:

- obtaining a source with historical publication-vintage timestamps for wind/solar forecasts;
- building a native 15-minute post-cutover forecasting and strategy system;
- adding physical storage constraints, capital requirements, market impact, and realistic transaction-cost models;
- validating the frozen approach on a later untouched period or another bidding zone.

---

This repository is a research system designed around reproducibility, chronological evaluation, explicit assumptions, frozen evidence, and documented failure modes. The central rule remains: if something can be tested, test it; if it cannot be verified, state the assumption or limitation explicitly.
