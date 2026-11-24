from __future__ import annotations

import numpy as np
import pandas as pd


def make_demo_sessions(n: int = 120, seed: int = 28) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-01-05")
    dates = [start + pd.Timedelta(days=int(4.5*i + rng.integers(-1, 2))) for i in range(n)]
    latent = np.linspace(-0.55, 0.85, n) + 0.12*np.sin(np.arange(n)/8)
    opponent = latent + rng.normal(0.0, 0.65, n)
    win_p = 1/(1+np.exp(-(1.20*(latent-opponent) + 0.15*latent)))
    wins = rng.binomial(1, win_p)

    serve_points = rng.integers(42, 82, n)
    first_p = np.clip(0.54 + 0.055*latent + rng.normal(0,0.025,n), 0.42, 0.78)
    first_in = rng.binomial(serve_points, first_p)
    double_fault_p = np.clip(0.055 - 0.012*latent + rng.normal(0,0.006,n), 0.01, 0.12)
    double_faults = rng.binomial(serve_points, double_fault_p)

    fh_att = rng.integers(45, 105, n); bh_att = rng.integers(28, 85, n)
    fh_p = np.clip(0.76 + 0.045*latent + rng.normal(0,0.018,n), .62, .91)
    bh_p = np.clip(0.70 + 0.065*latent + rng.normal(0,0.020,n), .55, .89)
    fh_in = rng.binomial(fh_att, fh_p); bh_in = rng.binomial(bh_att, bh_p)

    total_points = rng.integers(75, 135, n)
    # Post-match point share is deliberately outcome-correlated; this is why it must not be used pre-match.
    point_share = np.clip(0.49 + 0.11*(wins-0.5) + rng.normal(0,0.03,n), 0.30, 0.70)
    points_for = np.rint(total_points*point_share).astype(int)
    points_against = total_points-points_for

    winners = np.clip(np.rint(11 + 3.0*latent + 7*wins + rng.normal(0,4,n)), 1, None).astype(int)
    ues = np.clip(np.rint(28 - 4.5*latent - 5*wins + rng.normal(0,5,n)), 3, None).astype(int)
    rallies_total = rng.integers(35, 80, n)
    rally_p = np.clip(0.30 + 0.055*latent + rng.normal(0,.025,n), .15,.55)
    rallies5 = rng.binomial(rallies_total, rally_p)

    bp_chances = rng.integers(2, 11, n); bp_faced = rng.integers(2, 11, n)
    bp_conv = np.clip(0.36 + .08*latent + rng.normal(0,.05,n), .1,.75)
    bp_save = np.clip(0.48 + .06*latent + rng.normal(0,.05,n), .15,.85)
    bp_won = rng.binomial(bp_chances, bp_conv); bp_saved = rng.binomial(bp_faced, bp_save)

    df = pd.DataFrame({
        "session_id": [f"demo-{i+1:03d}" for i in range(n)],
        "date": dates,
        "session_type": "match",
        "opponent": [f"Opponent {1+(i%18):02d}" for i in range(n)],
        "opponent_level": opponent,
        "result": np.where(wins==1,"W","L"),
        "points_for": points_for,
        "points_against": points_against,
        "serve_points": serve_points,
        "first_serves_in": first_in,
        "double_faults": double_faults,
        "winners": winners,
        "unforced_errors": ues,
        "break_points_won": bp_won,
        "break_points_chances": bp_chances,
        "break_points_saved": bp_saved,
        "break_points_faced": bp_faced,
        "forehand_in": fh_in,
        "forehand_attempted": fh_att,
        "backhand_in": bh_in,
        "backhand_attempted": bh_att,
        "avg_serve_speed_kmh": 126 + 7.5*latent + rng.normal(0,2.5,n),
        "avg_forehand_speed_kmh": 92 + 6.0*latent + rng.normal(0,2.7,n),
        "avg_backhand_speed_kmh": 85 + 5.0*latent + rng.normal(0,2.5,n),
        "rallies_over_5": rallies5,
        "rallies_total": rallies_total,
        "duration_min": np.clip(82 + rng.normal(0,18,n), 42, 150),
    })
    return df
