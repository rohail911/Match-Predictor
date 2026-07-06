#!/usr/bin/env python3
"""
match_predictor.py

Interactive football match predictor. Builds team Elo ratings and
attack/defense strengths directly from results.csv, then lets you:

  predict  -- pre-match prediction (prompts for teams, competition, venue)
  live     -- live in-play update (enter current score + time + possession)
  scores   -- fetch today's live scores from football-data.org (free API key required)
  quit     -- exit

Run: python match_predictor.py
"""

from __future__ import annotations
import csv
import json
import os
import sys
import urllib.request
from collections import defaultdict
from typing import Dict, Optional

# Force UTF-8 output on Windows so box-drawing / special chars don't crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

import prediction_engine as pe
from llm_pipeline import narrate_result

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

CSV_PATH         = os.path.join(os.path.dirname(__file__), "results.csv")
LEAGUE_AVG_GOALS = 2.6    # international goals-per-team average
HOME_ADV_FACTOR  = 1.15   # home advantage multiplier (calibrate from backtest)
RHO              = -0.08  # Dixon-Coles correction (fit from backtest)
RECENT_N         = 40     # how many recent matches to use for attack/defense

TOURNAMENT_K = {
    "FIFA World Cup":        60,
    "UEFA Euro":             50,
    "Copa America":          50,
    "AFC Asian Cup":         50,
    "Africa Cup of Nations": 50,
    "Qualifier":             40,
    "Friendly":              20,
}
DEFAULT_K = 30

FOOTBALL_DATA_KEY = os.environ.get("FOOTBALL_DATA_API_KEY", "")


# ─────────────────────────────────────────────────────────────────────────────
# Stats builder  (Elo + attack/defense from results.csv)
# ─────────────────────────────────────────────────────────────────────────────

class _TeamRecord:
    __slots__ = ("elo", "scored", "conceded")

    def __init__(self):
        self.elo:      float      = 1500.0
        self.scored:   list[int]  = []
        self.conceded: list[int]  = []

    def attack_strength(self) -> float:
        recent = self.scored[-RECENT_N:] if self.scored else [1]
        avg = sum(recent) / len(recent)
        return avg / (LEAGUE_AVG_GOALS / 2)

    def defense_weakness(self) -> float:
        recent = self.conceded[-RECENT_N:] if self.conceded else [1]
        avg = sum(recent) / len(recent)
        return avg / (LEAGUE_AVG_GOALS / 2)


def build_stats(csv_path: str) -> Dict[str, _TeamRecord]:
    records: Dict[str, _TeamRecord] = defaultdict(_TeamRecord)

    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw_h = row.get("home_goals") or row.get("home_score", "")
            raw_a = row.get("away_goals") or row.get("away_score", "")
            try:
                hg, ag = int(raw_h), int(raw_a)
            except (ValueError, TypeError):
                continue

            home, away = row["home_team"], row["away_team"]
            neutral = str(row.get("neutral", "false")).strip().lower() in ("1", "true", "yes")
            venue = "neutral" if neutral else "home"
            k = TOURNAMENT_K.get(row.get("tournament", ""), DEFAULT_K)

            # update Elo
            rh, ra = records[home], records[away]
            rh.elo, ra.elo = pe.update_elo(rh.elo, ra.elo, hg, ag, k, venue=venue)

            # track goals
            rh.scored.append(hg);   rh.conceded.append(ag)
            ra.scored.append(ag);   ra.conceded.append(hg)

    return dict(records)


# ─────────────────────────────────────────────────────────────────────────────
# Live score fetching  (football-data.org free tier)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_live_scores() -> list[dict]:
    if not FOOTBALL_DATA_KEY:
        return []
    try:
        req = urllib.request.Request(
            "https://api.football-data.org/v4/matches?status=LIVE",
            headers={"X-Auth-Token": FOOTBALL_DATA_KEY},
        )
        with urllib.request.urlopen(req, timeout=6) as resp:
            return json.loads(resp.read()).get("matches", [])
    except Exception as e:
        print(f"  [live API error: {e}]")
        return []


def fetch_todays_scores() -> list[dict]:
    """Also fetch PAUSED and IN_PLAY matches (covers half-time etc.)."""
    if not FOOTBALL_DATA_KEY:
        return []
    try:
        req = urllib.request.Request(
            "https://api.football-data.org/v4/matches?status=LIVE,IN_PLAY,PAUSED",
            headers={"X-Auth-Token": FOOTBALL_DATA_KEY},
        )
        with urllib.request.urlopen(req, timeout=6) as resp:
            return json.loads(resp.read()).get("matches", [])
    except Exception as e:
        print(f"  [live API error: {e}]")
        return []


def print_live_scores(matches: list[dict]):
    if not matches:
        if not FOOTBALL_DATA_KEY:
            print(f"  [!] No FOOTBALL_DATA_API_KEY in .env")
            print("     Get a free key at https://www.football-data.org/client/register")
        else:
            print("\n  No live matches right now.")
        return
    print(f"\n  {'HOME':<28} {'SCORE':^9} {'AWAY':<28} {'MIN':>5}")
    print("  " + "─" * 72)
    for m in matches:
        home  = m["homeTeam"]["shortName"]
        away  = m["awayTeam"]["shortName"]
        sc    = m.get("score", {}).get("fullTime", {})
        hs    = sc.get("home") if sc.get("home") is not None else "-"
        aw    = sc.get("away") if sc.get("away") is not None else "-"
        mins  = m.get("minute") or m.get("status", "?")
        comp  = m.get("competition", {}).get("name", "")
        print(f"  {home:<28} {hs!s:^4}-{aw!s:^4} {away:<28} {mins!s:>5}'  [{comp}]")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _prompt(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"  {label}{suffix}: ").strip()
    return val if val else default


def _find_team(name: str, stats: Dict[str, _TeamRecord]) -> Optional[str]:
    """Case-insensitive lookup with prefix fallback."""
    lo = name.lower()
    for t in stats:
        if t.lower() == lo:
            return t
    hits = [t for t in stats if lo in t.lower()]
    if len(hits) == 1:
        return hits[0]
    if hits:
        print(f"  Did you mean: {', '.join(hits[:6])}?")
    return None


def _print_result(title: str, result: dict, narration: str):
    print(f"\n  {'-'*58}")
    print(f"  {title}")
    print(f"  {'-'*58}")
    if "predicted_winner" in result:
        w = result["predicted_winner"].upper()
        print(f"  Predicted winner : {w}")
        print(f"  Most likely score: {result['most_likely_score']}  "
              f"({result['most_likely_score_prob']:.1%} chance)")
    else:
        print(f"  Likely final score: {result['most_likely_final_score']}  "
              f"({result['most_likely_final_score_prob']:.1%} chance)")
    print(f"  Probabilities    : "
          f"Home {result['win_prob_home']:.1%}  "
          f"Draw {result['draw_prob']:.1%}  "
          f"Away {result['win_prob_away']:.1%}")
    if "elo_home" in result:
        print(f"  Elo ratings      : Home {result['elo_home']}  |  Away {result['elo_away']}")
    print(f"\n  >> {narration}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Prediction flow
# ─────────────────────────────────────────────────────────────────────────────

def do_predict(stats: Dict[str, _TeamRecord]):
    print()
    home_input = _prompt("Home team")
    if not home_input:
        return
    away_input = _prompt("Away team")
    if not away_input:
        return

    home_team = _find_team(home_input, stats)
    away_team = _find_team(away_input, stats)
    if not home_team:
        print(f"  [!] '{home_input}' not found in dataset. Check spelling.")
        return
    if not away_team:
        print(f"  [!] '{away_input}' not found in dataset. Check spelling.")
        return

    competition = _prompt("Competition", "Friendly")
    venue_raw   = _prompt("Venue (home / neutral)", "home").lower()
    venue       = "home" if venue_raw.startswith("h") else "neutral"

    hs, as_ = stats[home_team], stats[away_team]
    factor  = HOME_ADV_FACTOR if venue == "home" else 1.0

    print(f"  Computing prediction for {home_team} vs {away_team}...")
    result = pe.predict_match(
        home_attack=hs.attack_strength(),
        home_defense=hs.defense_weakness(),
        away_attack=as_.attack_strength(),
        away_defense=as_.defense_weakness(),
        league_avg_goals=LEAGUE_AVG_GOALS,
        venue=venue,
        home_advantage_factor=factor,
        rho=RHO,
    )
    result["elo_home"] = round(hs.elo)
    result["elo_away"] = round(as_.elo)

    narration = narrate_result(result)
    _print_result(f"PRE-MATCH  {home_team} vs {away_team}  [{competition}]", result, narration)

    # ── optional live update ──────────────────────────────────────────────────
    do_live = _prompt("Enter live match state? (y / n)", "n").lower()
    if not do_live.startswith("y"):
        return

    try:
        mins  = float(_prompt("Minutes elapsed", "45"))
        hg    = int(_prompt(f"  {home_team} goals scored so far", "0"))
        ag    = int(_prompt(f"  {away_team} goals scored so far", "0"))
        poss  = float(_prompt(f"  {home_team} possession advantage (e.g. 0.10 for 60/40)", "0.0"))
        shots = float(_prompt(f"  {home_team} shots-on-target lead", "0"))
    except ValueError:
        print("  [!] Invalid input -- skipping live update.")
        return

    live = pe.update_prediction_live(
        pre_match_lambda_home=result["lambda_home"],
        pre_match_lambda_away=result["lambda_away"],
        minutes_elapsed=mins,
        goals_home_so_far=hg,
        goals_away_so_far=ag,
        possession_diff_home=poss,
        shots_diff_home=shots,
        rho=RHO,
    )
    live_narr = narrate_result(live)
    _print_result(
        f"LIVE ({int(mins)}')  {home_team} {hg}–{ag} {away_team}",
        live, live_narr,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n  Loading team stats from results.csv...")
    stats = build_stats(CSV_PATH)
    print(f"  >> {len(stats)} teams ready.\n")

    print("=" * 62)
    print("  FOOTBALL MATCH PREDICTOR")
    print("=" * 62)
    live_status = "[connected]" if FOOTBALL_DATA_KEY else "[not configured -- add FOOTBALL_DATA_API_KEY to .env]"
    print(f"  Live scores API : {live_status}")
    print()
    print("  Commands:")
    print("    predict  (or p)  -- get a match prediction")
    print("    scores   (or s)  -- show live scores right now")
    print("    quit     (or q)  -- exit")
    print()

    while True:
        try:
            cmd = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye.")
            break

        if cmd in ("q", "quit", "exit"):
            print("  Goodbye.")
            break
        elif cmd in ("p", "predict", ""):
            do_predict(stats)
        elif cmd in ("s", "scores", "live", "live scores"):
            print("\n  Fetching live scores...")
            print_live_scores(fetch_todays_scores())
        else:
            print("  Unknown command. Try: predict / scores / quit")


if __name__ == "__main__":
    main()
