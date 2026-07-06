"""
elo_backtester.py

Purpose: prove the Elo module is actually predictive BEFORE building anything
else on top of it. This is step 1 of the "start here" build order -- run this
against real historical data and check the metrics before writing any Poisson
or live-layer code.

Expected CSV schema (one row per historical match, chronological order):
    date, home_team, away_team, home_goals, away_goals, neutral, tournament

    - date: any ISO-sortable string (used only for ordering, not parsed)
    - neutral: "true"/"false" (or 1/0) -- whether the match was at a neutral venue
    - tournament: used to look up the K-factor (see TOURNAMENT_K below);
      unrecognized values fall back to a default K.

This is the kind of schema you'd get from a long-running international
results dataset (e.g. the well-known Kaggle datasets covering international
football back to the 1800s). Point `run_backtest()` at your real CSV once
you've confirmed this runs cleanly on the synthetic sample below.
"""

from __future__ import annotations
import csv
import math
import random
from collections import defaultdict
from typing import Dict, List

from prediction_engine import elo_expected_score, update_elo

STARTING_RATING = 1500.0

TOURNAMENT_K = {
    "FIFA World Cup": 60,
    "Continental Championship": 50,
    "Qualifier": 40,
    "Friendly": 20,
}
DEFAULT_K = 30


# ---------------------------------------------------------------------------
# Synthetic sample data -- lets you verify the harness works before pointing
# it at real data. Do NOT treat these numbers as meaningful football insight.
# ---------------------------------------------------------------------------

def generate_synthetic_matches(n_teams: int = 12, n_matches: int = 800, seed: int = 7) -> List[dict]:
    rng = random.Random(seed)
    teams = [f"Team_{i:02d}" for i in range(n_teams)]
    true_strength = {t: rng.uniform(-1.5, 1.5) for t in teams}  # hidden "true" skill
    tournaments = list(TOURNAMENT_K.keys())

    matches = []
    for day in range(n_matches):
        home, away = rng.sample(teams, 2)
        neutral = rng.random() < 0.25
        tournament = rng.choice(tournaments)

        # goals driven by hidden strength + home advantage + noise, then
        # passed through a Poisson-ish process so scorelines look plausible
        base_home = 1.3 + (0 if neutral else 0.35) + true_strength[home] * 0.4
        base_away = 1.1 + true_strength[away] * 0.4 - true_strength[home] * 0.05
        lam_home, lam_away = max(base_home, 0.2), max(base_away, 0.2)
        goals_home = _sample_poisson(rng, lam_home)
        goals_away = _sample_poisson(rng, lam_away)

        matches.append({
            "date": f"day_{day:04d}",
            "home_team": home,
            "away_team": away,
            "home_goals": goals_home,
            "away_goals": goals_away,
            "neutral": neutral,
            "tournament": tournament,
        })
    return matches


def _sample_poisson(rng: random.Random, lam: float) -> int:
    # Knuth's algorithm -- fine for synthetic data generation, avoids a numpy dependency
    l, k, p = math.exp(-lam), 0, 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= l:
            return k - 1


def load_matches_from_csv(path: str) -> List[dict]:
    """Load matches from CSV. Accepts either home_goals/away_goals or
    home_score/away_score column names (the Kaggle international results
    dataset uses the latter). Extra columns (city, country, etc.) are ignored.
    Rows with missing/NA scores are skipped."""
    matches = []
    skipped = 0
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # support both naming conventions; skip rows with missing scores
            raw_home = row.get("home_goals") or row.get("home_score", "")
            raw_away = row.get("away_goals") or row.get("away_score", "")
            try:
                home_goals = int(raw_home)
                away_goals = int(raw_away)
            except (ValueError, TypeError):
                skipped += 1
                continue
            matches.append({
                "date": row["date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_goals": home_goals,
                "away_goals": away_goals,
                "neutral": str(row.get("neutral", "false")).strip().lower() in ("1", "true", "yes"),
                "tournament": row.get("tournament", "Friendly"),
            })
    if skipped:
        print(f"  (skipped {skipped} rows with missing/invalid scores)")
    return matches


# ---------------------------------------------------------------------------
# Backtest: walk matches chronologically, predict BEFORE updating ratings
# ---------------------------------------------------------------------------

def run_backtest(matches: List[dict]) -> dict:
    ratings: Dict[str, float] = defaultdict(lambda: STARTING_RATING)

    log_losses, briers, correct = [], [], []
    baseline_log_losses, baseline_briers, baseline_correct = [], [], []

    for m in matches:
        home, away = m["home_team"], m["away_team"]
        venue = "neutral" if m["neutral"] else "home"
        k_base = TOURNAMENT_K.get(m["tournament"], DEFAULT_K)

        # --- predict BEFORE seeing the result ---
        we_home = elo_expected_score(ratings[home], ratings[away], venue=venue)
        # naive 3-way split derived from the 2-way Elo expectancy: treat the
        # 2-way win expectancy as home-vs-not-home, then split "not home"
        # into away/draw with a fixed draw share. This is a rough approximation,
        # good enough for a first-pass calibration check -- refine once you
        # add the Poisson layer, which gives a proper 3-way split for free.
        draw_share = 0.26
        p_home = we_home * (1 - draw_share)
        p_away = (1 - we_home) * (1 - draw_share)
        p_draw = draw_share

        if m["home_goals"] > m["away_goals"]:
            actual = "home"
        elif m["home_goals"] < m["away_goals"]:
            actual = "away"
        else:
            actual = "draw"

        probs = {"home": p_home, "draw": p_draw, "away": p_away}
        p_actual = max(probs[actual], 1e-9)
        log_losses.append(-math.log(p_actual))
        briers.append(sum((probs[o] - (1.0 if o == actual else 0.0)) ** 2 for o in probs))
        predicted = max(probs, key=probs.get)
        correct.append(1 if predicted == actual else 0)

        # baseline: fixed 44/26/30 (roughly the long-run home/draw/away split
        # in international football) regardless of who's playing
        base_probs = {"home": 0.44, "draw": 0.26, "away": 0.30}
        p_actual_base = base_probs[actual]
        baseline_log_losses.append(-math.log(p_actual_base))
        baseline_briers.append(sum((base_probs[o] - (1.0 if o == actual else 0.0)) ** 2 for o in base_probs))
        baseline_correct.append(1 if max(base_probs, key=base_probs.get) == actual else 0)

        # --- only now update ratings ---
        ratings[home], ratings[away] = update_elo(
            ratings[home], ratings[away], m["home_goals"], m["away_goals"], k_base, venue=venue
        )

    n = len(matches)
    return {
        "n_matches": n,
        "elo_log_loss": round(sum(log_losses) / n, 4),
        "elo_brier": round(sum(briers) / n, 4),
        "elo_accuracy": round(sum(correct) / n, 4),
        "baseline_log_loss": round(sum(baseline_log_losses) / n, 4),
        "baseline_brier": round(sum(baseline_briers) / n, 4),
        "baseline_accuracy": round(sum(baseline_correct) / n, 4),
        "final_ratings_sample": dict(list(sorted(ratings.items(), key=lambda kv: -kv[1]))[:5]),
    }


if __name__ == "__main__":
    import sys
    import os

    DEFAULT_CSV = os.path.join(os.path.dirname(__file__), "results.csv")

    if len(sys.argv) > 1 and sys.argv[1] == "--synthetic":
        print("Running on synthetic sample data (for harness verification only).")
        matches = generate_synthetic_matches()
    elif len(sys.argv) > 1:
        print(f"Loading matches from {sys.argv[1]} ...")
        matches = load_matches_from_csv(sys.argv[1])
    elif os.path.exists(DEFAULT_CSV):
        print(f"Loading matches from {DEFAULT_CSV} ...")
        matches = load_matches_from_csv(DEFAULT_CSV)
    else:
        print("No CSV found -- running on synthetic sample data (for harness verification only).")
        matches = generate_synthetic_matches()

    results = run_backtest(matches)
    print()
    for k, v in results.items():
        print(f"{k}: {v}")

    print()
    if results["elo_log_loss"] < results["baseline_log_loss"]:
        print("Elo beats the naive baseline on log loss -- good sign, proceed to the Poisson layer.")
    else:
        print("Elo did NOT beat the naive baseline -- something's off before you add anything else.")
