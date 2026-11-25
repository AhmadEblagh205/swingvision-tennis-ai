# Personal Tennis Progress & Local AI Coaching Assistant

I built this to look at my own tennis progress from exported session statistics, using trend metrics, a walk-forward forecasting model and an optional local language model that explains the results. The intended source is a SwingVision Pro export. This repository does not contain my private SwingVision history, so the checked-in demo uses a synthetic dataset of 120 matches that I generated.

## What it does

- Loads and validates canonical session tables (CSV or XLSX), with an optional column mapping for a real export.
- Computes session metrics such as first-serve percentage, double-fault rate, serve speed and winner-minus-error balance, then compares the latest eight sessions with the previous eight and fits slopes per ten sessions.
- Forecasts the next match with a logistic regression using prior-only rolling features and expanding walk-forward evaluation against an expanding win-rate baseline.
- Builds a small evidence packet of aggregate statistics and writes a brief, either deterministically or through an optional local Ollama model.
- Prints only sheet and column names of a real workbook (`inspect_export`), so I can map its schema without exposing row values.

## Results

All numbers below come from the synthetic demo, not from my real sessions.

| Diagnostic | Result |
|---|---:|
| Walk-forward predictions | 90 |
| Logistic AUC | 0.645 |
| Logistic Brier | 0.2646 |
| Win-rate baseline Brier | 0.2493 |
| Brier improvement | -0.0153 |
| Promotion gate | fail |

The model ranks matches a little but its probabilities are worse than the baseline, so I do not promote it. A deliberately invalid model fitted to same-match statistics reaches an in-sample AUC of about 0.9986, which I keep in the demo as a leakage example. The planted long-run trends in the synthetic data are recovered, for example serve speed at about +0.92 km/h per 10 sessions.

## How to run

```bash
python -m pip install -r requirements.txt
PYTHONPATH=src python -m tennis_ai.audit
PYTHONPATH=src pytest
```

The test suite reports 19 passed. The audit writes `reports/audit_summary.json`, `reports/recent_block_changes.csv`, `reports/walk_forward_predictions.csv` and two figures in `figures/`.

Deterministic brief, no language model needed:

```bash
python -m tennis_ai.cli data/demo/sessions.csv
```

Optional local generation through an Ollama-compatible endpoint:

```bash
python -m tennis_ai.cli data/private/my_sessions.csv --assistant ollama --model llama3.2
```

## Using a real export

Put the file under `data/private/` (git-ignored), inspect its schema with `PYTHONPATH=src python -m tennis_ai.inspect_export data/private/export.xlsx`, map the fields using `config/column_map.example.json`, and run the CLI. I will only describe results as my own performance after the export has been checked. Details are in `docs/SWINGVISION_IMPORT.md`.

## Repository layout

```text
src/tennis_ai/   io, features, model, assistant, simulate, report, inspect_export, audit
tests/           invariants, leakage tests and assistant guardrails
data/demo/       synthetic demonstration data
data/private/    ignored real exports
reports/         machine-readable outputs
figures/         demo figures
docs/            methodology, import notes and model card
```

## Notes

SwingVision does not publish a stable raw-export schema, so I have not verified one. Derivations, the leakage test, the assistant design and measurement caveats are in `docs/METHODOLOGY.md`, and `docs/MODEL_CARD.md` describes the model. The assistant is local inference, not a fine-tuned model. This is an independent personal project and SwingVision is a third-party product with no connection to it.
