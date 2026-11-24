from __future__ import annotations

import argparse
from pathlib import Path
from .io import load_sessions, load_column_map
from .report import build_report, make_figures
from .assistant import build_evidence_packet, deterministic_brief, call_local_ollama


def main() -> None:
    p=argparse.ArgumentParser(description="Local tennis progress analytics from exported session statistics")
    p.add_argument("input", help="Canonical CSV/XLSX session file")
    p.add_argument("--column-map", default=None)
    p.add_argument("--out", default="reports")
    p.add_argument("--figures", default="figures")
    p.add_argument("--assistant", choices=["deterministic","ollama"], default="deterministic")
    p.add_argument("--model", default="llama3.2")
    args=p.parse_args()
    df=load_sessions(args.input, load_column_map(args.column_map))
    report=build_report(df,args.out)
    make_figures(df,args.figures)
    packet=build_evidence_packet(df)
    print(call_local_ollama(packet,args.model) if args.assistant=="ollama" else deterministic_brief(packet))
    print(f"\nWalk-forward Brier: {report['walk_forward']['brier_model']:.4f} (baseline {report['walk_forward']['brier_baseline']:.4f})")

if __name__ == "__main__":
    main()
