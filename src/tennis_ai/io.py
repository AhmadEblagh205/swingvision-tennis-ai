from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import numpy as np

from .schema import REQUIRED_COLUMNS, COUNT_COLUMNS


def _normalise_result(value: object) -> str:
    s = str(value).strip().upper()
    mapping = {"W": "W", "WIN": "W", "WON": "W", "1": "W", "TRUE": "W",
               "L": "L", "LOSS": "L", "LOST": "L", "0": "L", "FALSE": "L"}
    if s not in mapping:
        raise ValueError(f"Unsupported result value: {value!r}. Expected W/L or equivalent.")
    return mapping[s]


def load_column_map(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in obj.items()):
        raise ValueError("Column map must be a JSON object mapping source names to canonical names.")
    return obj


def load_sessions(path: str | Path, column_map: dict[str, str] | None = None) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        # Canonical summary workbook path. Raw per-shot SwingVision exports can be
        # converted with a user-specific mapping once a real export is available.
        xls = pd.ExcelFile(path)
        preferred = next((s for s in ["Sessions", "Summary", "Match Stats"] if s in xls.sheet_names), xls.sheet_names[0])
        df = pd.read_excel(path, sheet_name=preferred)
    else:
        raise ValueError("Supported input formats are .csv and .xlsx/.xls")

    if column_map:
        df = df.rename(columns=column_map)
    return validate_sessions(df)


def validate_sessions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df["session_id"].isna().any() or df["session_id"].duplicated().any():
        raise ValueError("session_id must be non-missing and unique")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("date contains unparseable values")
    if df["date"].duplicated().any() and df["session_id"].duplicated().any():
        raise ValueError("duplicate sessions detected")

    df["result"] = df["result"].map(_normalise_result)
    df["session_type"] = df["session_type"].astype(str).str.strip().str.lower()
    if (~df["session_type"].isin(["match", "practice"])).any():
        raise ValueError("session_type must be match or practice")

    for col in [c for c in COUNT_COLUMNS if c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any() or (df[col] < 0).any():
            raise ValueError(f"{col} must contain finite non-negative numbers")

    for col in ["avg_serve_speed_kmh", "avg_forehand_speed_kmh", "avg_backhand_speed_kmh", "opponent_level", "duration_min"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            bad = df[col].notna() & ~np.isfinite(df[col])
            if bad.any():
                raise ValueError(f"{col} contains non-finite values")

    logical = [
        ("first_serves_in", "serve_points"),
        ("double_faults", "serve_points"),
        ("forehand_in", "forehand_attempted"),
        ("backhand_in", "backhand_attempted"),
        ("rallies_over_5", "rallies_total"),
        ("break_points_won", "break_points_chances"),
        ("break_points_saved", "break_points_faced"),
    ]
    for num, den in logical:
        if num in df.columns and den in df.columns:
            if (df[num] > df[den]).any():
                raise ValueError(f"{num} cannot exceed {den}")

    if (df["avg_serve_speed_kmh"] <= 0).any():
        raise ValueError("avg_serve_speed_kmh must be positive")
    return df.sort_values(["date", "session_id"]).reset_index(drop=True)
