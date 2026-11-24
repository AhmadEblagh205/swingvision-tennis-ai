from __future__ import annotations

REQUIRED_COLUMNS = [
    "session_id", "date", "session_type", "result", "serve_points",
    "first_serves_in", "double_faults", "winners", "unforced_errors",
    "forehand_in", "forehand_attempted", "backhand_in", "backhand_attempted",
    "avg_serve_speed_kmh", "rallies_over_5", "rallies_total",
]

OPTIONAL_COLUMNS = [
    "opponent", "opponent_level", "points_for", "points_against",
    "break_points_won", "break_points_chances", "break_points_saved",
    "break_points_faced", "avg_forehand_speed_kmh", "avg_backhand_speed_kmh",
    "duration_min",
]

COUNT_COLUMNS = [
    "serve_points", "first_serves_in", "double_faults", "winners",
    "unforced_errors", "forehand_in", "forehand_attempted", "backhand_in",
    "backhand_attempted", "rallies_over_5", "rallies_total", "points_for",
    "points_against", "break_points_won", "break_points_chances",
    "break_points_saved", "break_points_faced",
]
