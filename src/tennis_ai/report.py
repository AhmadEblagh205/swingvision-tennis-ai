from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from .features import derive_session_metrics, compare_recent_blocks, trend_per_10_sessions
from .model import walk_forward_predictions, evaluate_predictions, promotion_gate, leaky_same_match_auc
from .assistant import build_evidence_packet, deterministic_brief


def build_report(df: pd.DataFrame, output_dir: str | Path) -> dict:
    outdir = Path(output_dir); outdir.mkdir(parents=True, exist_ok=True)
    x = derive_session_metrics(df)
    pred = walk_forward_predictions(df)
    perf = evaluate_predictions(pred)
    gate = promotion_gate(perf)
    changes = compare_recent_blocks(df)
    packet = build_evidence_packet(df)
    report = {
        "provenance": "synthetic demonstration unless explicitly replaced by user-exported SwingVision data",
        "n_sessions": int(len(df)),
        "date_start": str(x["date"].min().date()),
        "date_end": str(x["date"].max().date()),
        "recent_block_changes": {r.metric: float(r.change) for r in changes.itertuples(index=False)},
        "trend_per_10_sessions": {m: trend_per_10_sessions(df,m) for m in ["first_serve_pct","backhand_in_pct","avg_serve_speed_kmh","winner_error_balance"]},
        "walk_forward": perf,
        "model_promotion_gate": gate,
        "leaky_same_match_in_sample_auc": leaky_same_match_auc(df),
        "assistant_evidence": packet,
        "deterministic_brief": deterministic_brief(packet),
    }
    (outdir/"audit_summary.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    pred.to_csv(outdir/"walk_forward_predictions.csv", index=False)
    changes.to_csv(outdir/"recent_block_changes.csv", index=False)
    return report


def make_figures(df: pd.DataFrame, output_dir: str | Path) -> None:
    outdir=Path(output_dir); outdir.mkdir(parents=True, exist_ok=True)
    x=derive_session_metrics(df).sort_values("date")
    metrics=[("first_serve_pct","First serve in"),("backhand_in_pct","Backhand in"),("avg_serve_speed_kmh","Average serve speed (km/h)"),("winner_error_balance","Winners - unforced errors")]
    fig, axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
    for ax,(m,label) in zip(axes.ravel(),metrics):
        ax.plot(x["date"],x[m],alpha=.35)
        ax.plot(x["date"],x[m].rolling(8,min_periods=3).mean(),linewidth=2)
        ax.set_title(label); ax.grid(alpha=.2)
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(outdir/"progress_trends.png",dpi=160); plt.close(fig)

    pred=walk_forward_predictions(df)
    if not pred.empty:
        fig, ax=plt.subplots(figsize=(9,4))
        ax.plot(pred["date"], pred["p_model"], label="Model")
        ax.plot(pred["date"], pred["p_baseline"], label="Expanding win-rate baseline")
        ax.scatter(pred["date"], pred["y"], s=12, alpha=.35, label="Outcome")
        ax.set_ylim(-.05,1.05); ax.set_ylabel("Probability / outcome"); ax.legend(); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(outdir/"walk_forward_predictions.png",dpi=160); plt.close(fig)
