from __future__ import annotations

import json
import urllib.request
import pandas as pd

from .features import compare_recent_blocks


def build_evidence_packet(df: pd.DataFrame, block: int = 8) -> dict:
    changes = compare_recent_blocks(df, block=block)
    rows = []
    for r in changes.itertuples(index=False):
        rows.append({
            "metric": r.metric,
            "previous_mean": round(float(r.previous_mean), 6),
            "recent_mean": round(float(r.recent_mean), 6),
            "change": round(float(r.change), 6),
        })
    matches = df[df["session_type"].astype(str).str.lower()=="match"].copy()
    recent = matches.tail(block)
    return {
        "source": "computed session statistics",
        "block_size": block,
        "latest_date": str(pd.to_datetime(df["date"]).max().date()),
        "n_sessions": int(len(df)),
        "n_matches": int(len(matches)),
        "recent_match_record": {
            "wins": int((recent["result"].astype(str).str.upper()=="W").sum()),
            "losses": int((recent["result"].astype(str).str.upper()=="L").sum()),
        },
        "metric_changes": rows,
    }


def deterministic_brief(packet: dict) -> str:
    changes = sorted(packet["metric_changes"], key=lambda r: abs(r["change"]), reverse=True)
    lines = [
        f"Evidence-backed tennis brief through {packet['latest_date']} ({packet['n_sessions']} sessions).",
        f"Recent {packet['block_size']}-match record: {packet['recent_match_record']['wins']}W-{packet['recent_match_record']['losses']}L.",
    ]
    rate_metrics = {"first_serve_pct", "double_fault_rate", "forehand_in_pct", "backhand_in_pct", "rallies_5plus_pct", "break_point_conversion", "break_point_save_pct"}
    for row in changes[:4]:
        delta = row["change"]
        if row["metric"] in rate_metrics:
            formatted = f"{delta*100:+.2f} percentage points"
        elif "speed_kmh" in row["metric"]:
            formatted = f"{delta:+.2f} km/h"
        else:
            formatted = f"{delta:+.2f}"
        lines.append(f"- {row['metric']}: {formatted} versus the previous block.")
    lines.append("Interpret these as descriptive personal trends, not causal coaching conclusions.")
    return "\n".join(lines)


def build_llm_prompt(packet: dict) -> str:
    evidence = json.dumps(packet, indent=2)
    return (
        "You are a private tennis analytics assistant. Use ONLY the evidence JSON below. "
        "Do not invent technique diagnoses, injuries, causal explanations, or missing statistics. "
        "Separate measured changes from suggestions. Give: (1) three evidence-backed observations, "
        "(2) two questions to review in video, and (3) one concise next-session focus.\n\n"
        f"EVIDENCE_JSON:\n{evidence}\n"
    )


def call_local_ollama(packet: dict, model: str = "llama3.2", endpoint: str = "http://localhost:11434/api/generate") -> str:
    body = json.dumps({"model": model, "prompt": build_llm_prompt(packet), "stream": False}).encode("utf-8")
    req = urllib.request.Request(endpoint, data=body, headers={"Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if not isinstance(payload.get("response"), str):
        raise RuntimeError("Unexpected Ollama response")
    return payload["response"]
