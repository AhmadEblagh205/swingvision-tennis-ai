# Model card — experimental next-match logistic model

## Intended use

Research whether a player's own prior session statistics contain incremental signal about the outcome of a future match.

## Not intended for

- gambling or betting;
- opponent scouting claims without validated opponent data;
- causal coaching conclusions;
- medical/fitness advice;
- public claims of predictive edge before real-data walk-forward validation.

## Inputs

Strictly prior-only rolling summaries of personal tennis metrics, plus optional opponent-level information and days since last session.

## Output

Estimated probability of winning the next match.

## Evaluation

Expanding walk-forward evaluation with ROC-AUC, Brier score and log loss.

## Baseline

Expanding historical win probability.

## Promotion rule

Model predictions remain experimental unless they beat the baseline on out-of-sample Brier score and satisfy minimum sample/discrimination gates.

## Current checked-in result

Synthetic demo only. The model obtains some ranking signal but fails the Brier improvement gate, so predictions are not promoted.

## Key risks

- very small personal sample;
- changing opponent quality;
- non-stationary skill/training;
- measurement error in automated tracking;
- missing contextual variables;
- class imbalance;
- feedback loops if model predictions change behaviour.
