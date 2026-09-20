# Replication guide for the Energy Economics manuscript

This document maps the manuscript **Forecast-to-Trade under Information Constraints: Frozen Out-of-Sample Evidence from the Germany--Luxembourg Day-Ahead Electricity Market** to the code and machine-readable evidence preserved in this repository.

## Evidential status

The January--July 2026 holdout has already been exposed. The original model-selection, uncertainty, economic-contract and nine-cell sign-consistency rules were specified and frozen before holdout exposure. Analyses under `outputs/posthoc/` were introduced after exposure and are explicitly supplementary.

The frozen protocol commit is:

`a2d1f9a79022ca1e2731d52803061825038c69f5`

The final Full and Tier-1 XGBoost refit commit is:

`54a7c2e77122134eca31a09798cc1614df1c634c`

## Software environment

Create a clean Python environment and install the locked dependencies:

```bash
python -m venv .venv
# activate .venv for your operating system
pip install -r requirements-lock.txt
```

Run the test and repository-integrity suite:

```bash
python -m pytest tests/ -v
python scripts/run_all_checks.py
```

## Data access and exact-reproduction boundary

The project uses public ENTSO-E Transparency Platform data. Fresh source retrieval requires an ENTSO-E API token:

```text
ENTSOE_TOKEN=your_token_here
```

Raw and general processed source-data directories are not committed because they contain large retrieved datasets. Public market databases can also be revised after the original retrieval date. Therefore:

1. **Exact published holdout results** can be independently audited from the preserved machine-readable holdout inputs and outputs in this repository.
2. **Development outputs and model-selection evidence** are preserved in versioned `outputs/` directories.
3. **A byte-for-byte reconstruction of the original entire processed feature dataset from a fresh API pull is not presently guaranteed**, because historical source revisions and the untracked full processed feature file can differ from the original research snapshot.

Before journal submission, the exact derived feature file used for the final manuscript should be deposited in a stable research archive, subject to ENTSO-E redistribution terms, together with its SHA-256 checksum and this repository release DOI. This is the principal remaining replication-package gate.

## Frozen holdout: exact machine-readable inputs

Primary hourly holdout inputs:

`outputs/holdout/delu_features/holdout_v1/hourly_holdout_inputs.csv`

Daily strategy results:

`outputs/holdout/delu_features/holdout_v1/per_day_results.csv`

Point-forecast table:

`outputs/holdout/delu_features/holdout_report_v1/point_forecast_metrics.csv`

Primary strategy table:

`outputs/holdout/delu_features/holdout_report_v1/primary_strategy_table.csv`

Nine-cell sensitivity grid:

`outputs/holdout/delu_features/holdout_report_v1/sensitivity_grid.csv`

Daily sensitivity results:

`outputs/holdout/delu_features/holdout_sensitivity_v1/per_day_sensitivity_results.csv`

Tail-risk outputs:

`outputs/risk/delu_features/holdout_tailrisk_v1/`

## Development evidence

Corrected XGBoost development lineage:

`outputs/models/delu_features/xgboost_v1_a03fix/`

Corrected ElasticNet benchmark lineage:

`outputs/models/delu_features/baseline_v1_a03fix/`

Corrected EDA:

`outputs/eda/delu_features/development/development_a03fix_v1/`

Selected uncertainty outputs:

`outputs/uncertainty/delu_features/uncertainty_selected_v2_dayorigin/`

Development strategy report:

`outputs/strategy/delu_features/strategy_report_v2_dayorigin/`

Development tail risk:

`outputs/risk/delu_features/strategy_tailrisk_v2_dayorigin/`

Final fitted XGBoost models and manifests:

`outputs/final_model_fit/delu_features/final_model_fit_v1/`

## Manuscript result map

| Manuscript result | Primary machine-readable source |
|---|---|
| Development MAE table | `outputs/models/delu_features/xgboost_v1_a03fix/` |
| Development uncertainty-window table | corrected uncertainty sensitivity/selection outputs |
| Development strategy table | `strategy_report_v2_dayorigin` |
| Development VaR/ES table | `strategy_tailrisk_v2_dayorigin` |
| Holdout MAE/RMSE table | `holdout_report_v1/point_forecast_metrics.csv` |
| Holdout S0--S5 strategy table | `holdout_report_v1/primary_strategy_table.csv` |
| Frozen 3x3 S2-S1 sensitivity table | `holdout_report_v1/sensitivity_grid.csv` |
| Holdout daily P&L figure | `holdout_v1/per_day_results.csv` |
| Holdout VaR/ES | `holdout_tailrisk_v1` |
| Post-hoc benchmark table | `outputs/posthoc/robustness_v1/posthoc_benchmark_summary.csv` |
| Post-hoc block-bootstrap table | `outputs/posthoc/robustness_v1/posthoc_block_bootstrap.csv` |
| Post-hoc monthly S2-S1 decomposition | `outputs/posthoc/robustness_v1/posthoc_monthly_s2_s1.csv` |
| Post-hoc interval calibration | `outputs/posthoc/robustness_v1/posthoc_calibration_*.csv` |

## Post-hoc robustness reproduction

Run:

```bash
python scripts/posthoc_robustness_v1.py
```

This produces the supplementary benchmark, 7-day moving-block bootstrap, monthly-decomposition and calibration outputs under:

`outputs/posthoc/robustness_v1/`

These analyses were introduced after holdout exposure and must not be described as part of the original frozen protocol.

## Primary pipeline entry points

The project is modular rather than controlled by one monolithic runner. The principal stages are:

```text
run_ingestion.py
run_features.py
run_eda.py
run_models.py
run_xgboost.py
run_uncertainty_sensitivity.py
run_uncertainty.py
run_strategy_backtest.py
run_strategy_sensitivity.py
run_strategy_decomposition.py
run_strategy_report.py
run_strategy_tailrisk.py
run_final_model_fit.py
run_final_holdout.py
run_final_holdout_sensitivity.py
run_final_holdout_report.py
run_final_holdout_tailrisk.py
```

For exact frozen-lineage reproduction, use the manifests and version names already stored under `outputs/`; do not silently replace versioned output directories or alter parameters.

## Manuscript figures

Figures 1--4 are based on corrected development EDA outputs.

Figures 5--9 are manuscript presentation figures. Figures 7--9 are numerical visualizations of preserved machine-readable holdout outputs; Figure 5 is a research-workflow schematic and Figure 6 is a chronological-design schematic.

## Important interpretation notes

- Reported strategy P&L is denominated in euros for a normalized 1-MWh charge opportunity per delivery day under the stylized single-cycle contract.
- Full renewable-derived predictors have unresolved exact 11:45 historical publication vintages.
- Tier-1 is the conservative point-in-time robustness specification.
- The uncertainty-aware S3/S5 rule is algebraically a day-varying abstention threshold because residual offsets are common across hours within a day.
- The nine frozen sensitivity cells are related scenarios, not independent statistical replications.
- Post-hoc analyses do not redefine the frozen primary sign-consistency criterion.
