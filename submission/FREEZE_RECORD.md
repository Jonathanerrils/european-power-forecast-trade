# Energy Economics submission freeze

Date: 2026-09-20

## Manuscript

Title: Forecast-to-Trade under Information Constraints: Frozen Out-of-Sample Evidence from the Germany--Luxembourg Day-Ahead Electricity Market

Author: Jonathan Anthonio Nii Pedro Nelson

Affiliation: Kwame Nkrumah University of Science and Technology (KNUST), Kumasi, Ghana

Target journal: Energy Economics

## Repository freeze

Merged pull request: #3

Main merge commit:

`c757157634cae79014345fa29c74d5aa09bd6d95`

Submission freeze branch:

`submission/energy-economics-2026-09-20`

Pre-holdout protocol commit:

`a2d1f9a79022ca1e2731d52803061825038c69f5`

Final Full and Tier-1 model-fit commit:

`54a7c2e77122134eca31a09798cc1614df1c634c`

## Approved compiled PDF

Approved Overleaf compile: 46 pages

SHA-256:

`ac0456f7c135446a2de51274520b651d4af760d92e0a322ded957ec34d80d8d6`

The PDF was visually reviewed after the final author-name, acknowledgements,
negative-euro typography, PNG-figure and Figure-6 layout corrections.

## Evidential boundary

The January--July 2026 holdout has already been exposed. No further tuning of
models, information sets, uncertainty rules, thresholds or economic decision
rules should use this holdout. Any redesigned method must be frozen before
evaluation on a new untouched period.

Post-hoc analyses under `outputs/posthoc/robustness_v1/` remain supplementary
and do not redefine the frozen primary sign-consistency criterion.

## Remaining archival tasks

- Create a formal GitHub release/tag from this submission freeze.
- Archive the release on Zenodo or an equivalent repository and record the DOI.
- Archive the exact final derived feature dataset, subject to ENTSO-E terms,
  and record its SHA-256 checksum.
- Verify the replication instructions in a clean environment before journal
  submission.
