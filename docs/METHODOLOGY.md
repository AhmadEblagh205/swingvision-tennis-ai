# Methodology

## 1. Analytical question

The project separates three tasks that are easy to conflate:

1. **measurement/trend description** — how exported metrics change over time;
2. **prediction** — whether information known before a match helps forecast its outcome;
3. **language generation** — how to explain already-computed evidence.

A language model is not used to estimate a serve percentage, fit a trend or score a probability model.

## 2. Canonical session table

The project currently operates at session level. Required fields are documented in `schema.py`. Count metrics are retained as numerator/denominator pairs wherever possible so percentages can be recomputed rather than imported as rounded display values.

For example:

\[
\text{first serve \%}=\frac{\text{first serves in}}{\text{serve points}}.
\]

Rows are sorted by date before any time-aware feature construction.

## 3. Descriptive progress

For metric \(x_t\), the current report uses:

- an 8-session rolling display;
- an 8-vs-previous-8 block difference;
- a simple least-squares slope against session index, reported per 10 sessions.

These are deliberately transparent rather than pretending a small personal history supports sophisticated causal time-series inference.

Irregular time gaps are retained in the data and supplied to the prediction model as `days_since_last`, but the slope is currently per session rather than per calendar month.

## 4. Prediction estimand

The predictive target is:

> probability of winning match \(t\), using only information available before match \(t\).

For every rolling metric, the key operation is:

```python
x.shift(1).rolling(window).mean()
```

The `shift(1)` is the leakage barrier.

Using same-match winners/errors, point share or serve outcomes to claim pre-match predictability would answer a different, easier question.

## 5. Walk-forward evaluation

For each eligible match \(t\):

1. train on matches strictly before \(t\);
2. predict \(t\);
3. append the prediction;
4. move forward one match.

This mimics how a model would actually accumulate information over a player's career.

The model is logistic regression after median imputation and standardisation.

The baseline is the historical win rate in the training set:

\[
\hat p_{baseline,t}=\frac{1}{t-1}\sum_{i<t} y_i.
\]

## 6. Why Brier score matters

ROC-AUC evaluates ranking/discrimination. It does not tell us whether a probability of 0.7 behaves like a 70% event.

For binary outcome \(y_i\) and predicted probability \(p_i\), Brier score is:

\[
\text{Brier}=\frac1n\sum_i(p_i-y_i)^2.
\]

Lower is better.

The synthetic demo obtains AUC around 0.645 but a worse Brier score than the simple expanding-win-rate baseline. The promotion gate therefore fails.

That is not a contradiction: a model can rank outcomes somewhat usefully while producing probabilities that are too noisy or badly calibrated.

## 7. Promotion gate

The assistant should not surface model probabilities unless:

- at least 40 walk-forward predictions exist;
- model Brier is lower than baseline Brier;
- AUC is at least 0.55.

These are engineering guardrails, not universal statistical laws. They are intentionally modest and explicit.

For serious deployment, thresholds should be pre-registered and accompanied by uncertainty intervals and repeated resampling where the temporal structure permits it.

## 8. Leakage stress tests

Two regression tests protect the time direction:

- changing the current match cannot change its own pre-match rolling features;
- changing future matches cannot alter historical pre-match features.

A separate deliberately leaky diagnostic fits an in-sample classifier using same-match statistics. Its near-perfect AUC exists to show how easy it is to create a misleading success story.

## 9. Local assistant architecture

`build_evidence_packet()` creates a JSON object containing aggregate computed evidence only.

The LLM prompt requires the model to:

- use only the supplied evidence;
- avoid fabricated metrics;
- avoid biomechanical/injury diagnoses;
- separate measured observations from suggestions.

This is closer to retrieval/evidence grounding than model fine-tuning.

## 10. Measurement uncertainty

SwingVision-derived values are measurements produced by a computer-vision system. Their uncertainty may depend on camera setup, frame rate, occlusion, session type and tracking quality.

This repository currently does not possess per-observation confidence intervals from SwingVision, so it does not pretend the exported statistics are exact.

A future real-data extension could include:

- reliability flags per session;
- replicate/manual annotation checks on a subset;
- uncertainty-aware trend modelling;
- hierarchical models separating genuine player change from measurement noise.

## 11. Synthetic benchmark

The checked-in 120-match demo plants gradual improvement in latent skill and several observable metrics. It is used for software validation and known-truth reasoning only.

It must never be presented as my tennis performance.
