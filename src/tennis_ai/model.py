from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

from .features import prior_only_features, derive_session_metrics


def _pipeline() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=1.0, max_iter=2000)),
    ])


def _feature_cols(frame: pd.DataFrame) -> list[str]:
    cols = [c for c in frame.columns if c.startswith("recent_")]
    if "opponent_level" in frame.columns:
        cols.append("opponent_level")
    if "days_since_last" in frame.columns:
        cols.append("days_since_last")
    return cols


def walk_forward_predictions(df: pd.DataFrame, min_train: int = 30, window: int = 5) -> pd.DataFrame:
    frame = prior_only_features(df, window=window)
    frame = frame[frame["session_type"] == "match"].reset_index(drop=True)
    features = _feature_cols(frame)
    predictions = []
    for i in range(min_train, len(frame)):
        train = frame.iloc[:i].dropna(subset=["win"])
        test = frame.iloc[[i]]
        if train["win"].nunique() < 2:
            continue
        model = _pipeline().fit(train[features], train["win"])
        p = float(model.predict_proba(test[features])[:, 1][0])
        baseline = float(train["win"].mean())
        predictions.append({
            "session_id": test["session_id"].iloc[0],
            "date": test["date"].iloc[0],
            "y": int(test["win"].iloc[0]),
            "p_model": p,
            "p_baseline": baseline,
        })
    return pd.DataFrame(predictions)


def evaluate_predictions(pred: pd.DataFrame) -> dict[str, float]:
    if pred.empty:
        raise ValueError("No predictions available")
    y = pred["y"].to_numpy()
    out = {
        "n_predictions": float(len(pred)),
        "brier_model": float(brier_score_loss(y, pred["p_model"])),
        "brier_baseline": float(brier_score_loss(y, pred["p_baseline"])),
        "logloss_model": float(log_loss(y, pred["p_model"], labels=[0,1])),
        "logloss_baseline": float(log_loss(y, pred["p_baseline"], labels=[0,1])),
    }
    if len(np.unique(y)) == 2:
        out["auc_model"] = float(roc_auc_score(y, pred["p_model"]))
        out["auc_baseline"] = float(roc_auc_score(y, pred["p_baseline"]))
    else:
        out["auc_model"] = float("nan")
        out["auc_baseline"] = float("nan")
    return out


def promotion_gate(metrics: dict[str, float], min_predictions: int = 40) -> dict[str, object]:
    """Conservative gate before a predictive model may be surfaced to the assistant."""
    n = int(metrics["n_predictions"])
    brier_gain = float(metrics["brier_baseline"] - metrics["brier_model"])
    auc = float(metrics.get("auc_model", float("nan")))
    reasons = []
    if n < min_predictions:
        reasons.append(f"fewer than {min_predictions} walk-forward predictions")
    if not np.isfinite(brier_gain) or brier_gain <= 0:
        reasons.append("no out-of-sample Brier improvement over expanding win-rate baseline")
    if not np.isfinite(auc) or auc < 0.55:
        reasons.append("AUC below the pre-specified 0.55 discrimination floor")
    return {
        "promote": len(reasons) == 0,
        "brier_gain": brier_gain,
        "reasons": reasons,
    }


def leaky_same_match_auc(df: pd.DataFrame) -> float:
    """Diagnostic only: shows how flattering post-match features can look if misused for prediction."""
    x = derive_session_metrics(df)
    x = x[x["session_type"] == "match"].reset_index(drop=True)
    features = [c for c in [
        "first_serve_pct", "double_fault_rate", "forehand_in_pct", "backhand_in_pct",
        "rallies_5plus_pct", "avg_serve_speed_kmh", "winner_error_balance", "point_win_pct"
    ] if c in x.columns]
    if len(x) < 20 or x["win"].nunique() < 2:
        return float("nan")
    model = _pipeline().fit(x[features], x["win"])
    return float(roc_auc_score(x["win"], model.predict_proba(x[features])[:,1]))
