from __future__ import annotations

import numpy as np
import pandas as pd

RATE_SPECS = {
    "first_serve_pct": ("first_serves_in", "serve_points"),
    "double_fault_rate": ("double_faults", "serve_points"),
    "forehand_in_pct": ("forehand_in", "forehand_attempted"),
    "backhand_in_pct": ("backhand_in", "backhand_attempted"),
    "rallies_5plus_pct": ("rallies_over_5", "rallies_total"),
    "break_point_conversion": ("break_points_won", "break_points_chances"),
    "break_point_save_pct": ("break_points_saved", "break_points_faced"),
}

CORE_PROGRESS_METRICS = [
    "first_serve_pct", "double_fault_rate", "forehand_in_pct", "backhand_in_pct",
    "rallies_5plus_pct", "avg_serve_speed_kmh", "winner_error_balance",
]


def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    out = num.astype(float) / den.replace(0, np.nan).astype(float)
    return out


def derive_session_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["win"] = (out["result"] == "W").astype(int)
    for name, (num, den) in RATE_SPECS.items():
        if num in out.columns and den in out.columns:
            out[name] = safe_ratio(out[num], out[den])
    out["winner_error_balance"] = out["winners"] - out["unforced_errors"]
    if {"points_for", "points_against"}.issubset(out.columns):
        total = out["points_for"] + out["points_against"]
        out["point_win_pct"] = safe_ratio(out["points_for"], total)
    return out


def prior_only_features(df: pd.DataFrame, window: int = 5, min_periods: int = 3) -> pd.DataFrame:
    """Build pre-session features using only strictly earlier sessions."""
    x = derive_session_metrics(df).sort_values(["date", "session_id"]).reset_index(drop=True)
    metrics = [m for m in CORE_PROGRESS_METRICS if m in x.columns]
    for metric in metrics:
        x[f"recent_{metric}"] = x[metric].shift(1).rolling(window, min_periods=min_periods).mean()
    x["recent_win_rate"] = x["win"].shift(1).rolling(window, min_periods=min_periods).mean()
    x["days_since_last"] = x["date"].diff().dt.days.astype(float)
    x.loc[0, "days_since_last"] = np.nan
    return x


def compare_recent_blocks(df: pd.DataFrame, block: int = 8) -> pd.DataFrame:
    x = derive_session_metrics(df).sort_values(["date", "session_id"]).reset_index(drop=True)
    if len(x) < 2 * block:
        raise ValueError(f"Need at least {2*block} sessions for two {block}-session blocks")
    rows = []
    for metric in [m for m in CORE_PROGRESS_METRICS if m in x.columns]:
        old = x[metric].iloc[-2*block:-block].dropna()
        new = x[metric].iloc[-block:].dropna()
        if len(old) and len(new):
            rows.append({"metric": metric, "previous_mean": old.mean(), "recent_mean": new.mean(), "change": new.mean()-old.mean()})
    return pd.DataFrame(rows)


def trend_per_10_sessions(df: pd.DataFrame, metric: str) -> float:
    x = derive_session_metrics(df).sort_values(["date", "session_id"]).reset_index(drop=True)
    y = x[metric].to_numpy(dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    idx = np.arange(len(x), dtype=float)[mask]
    slope = np.polyfit(idx, y[mask], 1)[0]
    return float(slope * 10.0)
