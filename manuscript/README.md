# Overleaf-ready manuscript

This directory contains the publication manuscript for the DE-LU forecast-to-trade study.

## Compile

Set `manuscript/main.tex` as the main document and compile with pdfLaTeX + BibTeX.

The current first-choice target is **Energy Economics**. The draft uses Elsevier's `elsarticle` class, so retargeting later remains straightforward if editorial fit or reviewer feedback suggests another outlet.

## Overleaf workflow

1. Import or Git-sync the repository in Overleaf.
2. Use the branch `manuscript/overleaf-draft-v1`.
3. Set `manuscript/main.tex` as the main file.
4. The manuscript references EDA figures already stored in the repository.
5. Replace placeholder authors, affiliations, funding and acknowledgements before submission.
6. Adapt the AI-assistance declaration to the chosen journal's current policy.

## Scientific source of truth

The manuscript is grounded in these frozen artifacts:

- `xgboost_v1_a03fix`
- `uncertainty_selected_v2_dayorigin`
- `strategy_report_v2_dayorigin`
- `strategy_tailrisk_v2_dayorigin`
- `holdout_v1`
- `holdout_report_v1`
- `holdout_sensitivity_v1`
- `holdout_tailrisk_v1`

Pre-holdout protocol commit:
`a2d1f9a79022ca1e2731d52803061825038c69f5`

Final model-fit commit:
`54a7c2e77122134eca31a09798cc1614df1c634c`

Do not alter a scientific result merely to improve the manuscript narrative.
