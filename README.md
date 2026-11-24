# Personal Tennis Progress & Local AI Coaching Assistant

A privacy-first analytics project for turning exported tennis-session statistics into **longitudinal progress diagnostics, leakage-safe machine-learning experiments, and an optional evidence-grounded local LLM assistant**.

The intended real-data source is SwingVision Pro. SwingVision publicly documents automated shot speed, consistency, first-serve/groundstroke percentages, winners/errors and other match statistics, and its Pro tier advertises raw-data export. This repository does **not** contain my private SwingVision history. Until a real export is supplied and its schema is validated, the checked-in demo uses a clearly labelled synthetic dataset.

## Why this project exists

A session-level sports dashboard is useful, but it answers only "what happened today?". This project asks a harder set of questions:

- Am I improving across months, or reacting to one good session?
- Which metrics are changing consistently?
- Can a model forecast the next match from **information available before that match**?
- Does that model beat a simple historical baseline out of sample?
- Can a local language model explain the evidence without becoming the source of the evidence?

The architecture deliberately keeps those jobs separate:

```text
SwingVision export / canonical session file
        ↓
schema + plausibility validation
        ↓
rate/quality metrics + rolling personal trends
        ↓
strictly prior-only pre-match features
        ↓
expanding walk-forward ML + baseline gate
        ↓
structured evidence packet
        ↓
deterministic brief OR optional local Ollama LLM
```

The LLM is therefore an **explanation layer**, not a numerical oracle.

## What is implemented

### 1. Defensive local data ingestion

`tennis_ai.io` accepts canonical CSV/XLSX session tables, optional source→canonical column mappings and explicit validation for:

- unique session IDs;
- parseable dates;
- match/practice labels;
- non-negative counts;
- numerator ≤ denominator constraints;
- positive serve speeds;
- W/L result normalisation.

Private exports belong in `data/private/`, which is git-ignored.

Because SwingVision's public documentation does not currently define a stable raw-export schema, the project does **not** pretend one has been verified. `python -m tennis_ai.inspect_export YOUR_FILE.xlsx` prints only sheet/column metadata, not row values, so one real export can later be mapped safely.

### 2. Progress metrics

Derived session metrics include:

- first-serve percentage;
- double-fault rate;
- forehand/backhand in percentage;
- rallies lasting 5+ shots;
- break-point conversion/save rates when available;
- winner-minus-unforced-error balance;
- point-win percentage when point counts are available;
- serve speed.

The report compares the latest eight sessions with the preceding eight and calculates simple long-run slopes per ten sessions. These are **descriptive personal trends**, not causal statements about technique, equipment or training interventions.

### 3. Leakage-safe next-match ML

For match forecasting, every rolling feature is generated with:

```python
series.shift(1).rolling(...)
```

so the current match cannot contribute to its own prediction.

The model is a standardised logistic regression evaluated with **expanding walk-forward prediction**. At every evaluation date it is fitted only on earlier matches.

The comparison baseline is the expanding historical win rate. Reported diagnostics include:

- ROC-AUC;
- Brier score;
- log loss.

### 4. A promotion gate instead of leaderboard theatre

The model is allowed into the assistant only if it passes pre-specified checks:

1. enough genuine walk-forward predictions;
2. out-of-sample Brier improvement over the simple expanding-win-rate baseline;
3. minimum discrimination floor.

The checked-in synthetic benchmark is deliberately a **negative result**:

| Diagnostic | Result |
|---|---:|
| Walk-forward predictions | 90 |
| Logistic AUC | 0.645 |
| Logistic Brier | 0.2646 |
| Win-rate baseline Brier | 0.2493 |
| Brier improvement | -0.0153 |
| Promotion gate | **FAIL** |

The model has some ranking signal, but its probabilities are worse than the simple baseline. It is therefore **not promoted as a useful predictor**.

That is intentional. A small personal dataset should not receive a "machine learning works!" claim merely because one metric looks flattering.

### 5. Deliberate leakage demonstration

The demo also fits an explicitly invalid, post-hoc model to statistics from the **same match it is trying to classify**.

Its in-sample AUC is approximately:

**0.9986**

That number is not evidence of a brilliant predictor. It is evidence that using post-match outcome-related statistics such as point share and winner/error balance can make an invalid experiment look spectacular.

The project therefore contains both the correct and the tempting-wrong version.

### 6. Known-truth progress benchmark

`tennis_ai.simulate.make_demo_sessions()` creates 120 labelled synthetic matches with a slowly improving latent skill process. It is not intended to mimic my actual tennis level.

The analysis correctly recovers positive long-run movement in planted metrics, for example:

- average serve speed: about **+0.92 km/h per 10 sessions**;
- backhand-in rate: about **+0.92 percentage points per 10 sessions**;
- first-serve rate: about **+0.68 percentage points per 10 sessions**.

The last 8-vs-previous-8 block happens to show first serve up ~6.50 pp and backhand-in down ~2.18 pp. That apparent short-run contradiction is useful: a noisy local block and a long-run trend answer different questions.

### 7. Evidence-grounded local assistant

The assistant never receives the full raw match table by default. `build_evidence_packet()` sends only aggregate evidence such as:

- data horizon;
- session count;
- recent W/L record;
- recent-vs-previous metric changes.

The deterministic mode works with no LLM at all:

```bash
python -m tennis_ai.cli data/demo/sessions.csv
```

Optional local generation is supported through an Ollama-compatible localhost endpoint:

```bash
python -m tennis_ai.cli data/private/my_sessions.csv \
  --assistant ollama \
  --model llama3.2
```

The prompt explicitly forbids invented metrics, injury/technique diagnoses and unsupported causal explanations.

This is a **local inference integration**, not a fine-tuned LLM. Fine-tuning a language model on a small table of personal tennis sessions would not be a sensible way to learn numerical progress.

## Reproduce the audited demo

```bash
python -m pip install -r requirements.txt
PYTHONPATH=src python -m tennis_ai.audit
PYTHONPATH=src pytest
```

Expected test state:

```text
19 passed
```

The audit writes:

- `reports/audit_summary.json`
- `reports/recent_block_changes.csv`
- `reports/walk_forward_predictions.csv`
- `figures/progress_trends.png`
- `figures/walk_forward_predictions.png`

## Using a real SwingVision export later

1. Export raw/statistical data from SwingVision Pro.
2. Put the file under `data/private/`.
3. For an XLSX workbook, inspect only its schema:

```bash
PYTHONPATH=src python -m tennis_ai.inspect_export data/private/export.xlsx
```

4. Map the available fields into the canonical schema. An example mapping lives in `config/column_map.example.json`.
5. Run the CLI locally.
6. Only after the real export has been checked should any results from this repository be described as my personal tennis performance.

See `docs/SWINGVISION_IMPORT.md`.

## What this project does **not** claim

- The synthetic demo is not my SwingVision record.
- The current ML model has not demonstrated out-of-sample predictive value beyond the simple probability baseline.
- Correlation between a tennis metric and winning does not establish that changing that metric will cause more wins.
- SwingVision tracking measurements have their own measurement error; this project treats the exported values as measurements, not ground truth physics.
- The assistant is not a professional tennis coach, biomechanical model or medical tool.
- No external LLM is required, and private raw exports are not committed.

## Why this is stronger than simply "training an AI on my tennis data"

The statistical layer owns the numbers. The predictive model must beat a baseline under time-aware evaluation. The LLM receives a restricted evidence packet and is allowed only to communicate or interrogate those outputs.

That separation makes failures visible and testable.

## Repository structure

```text
src/tennis_ai/
  io.py              validation + canonical loading
  features.py        session metrics + prior-only features
  model.py           walk-forward ML + promotion gate
  assistant.py       evidence packet + deterministic/Ollama assistant
  simulate.py        controlled known-truth demo
  report.py          reports + figures
  inspect_export.py  privacy-conscious workbook-schema inspection
  audit.py           reproducible full demo

tests/               invariants, leakage tests and assistant guardrails
data/demo/            synthetic demonstration data
data/private/         ignored real exports
reports/              machine-readable audited outputs
figures/              reproducible demo figures
docs/                 methodology/import/model-card documentation
```

## Provenance

Independent personal portfolio project inspired by my own use of SwingVision Pro. The checked-in dataset is synthetic. SwingVision is a third-party product and is not affiliated with this repository.

## External product references

SwingVision product and guide pages consulted during project design:

- https://swing.vision/
- https://swing.tennis/
- https://swing.vision/guides/review-your-shot-stats
- https://swing.vision/guides/track-your-match

