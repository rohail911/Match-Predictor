"""
llm_pipeline.py

The entire "agentic" surface of this system: exactly two LLM calls.
    1. parse_query()    -- structured-output extraction of team names / intent
                            from a natural-language question.
    2. narrate_result()  -- turns the deterministic prediction JSON into
                            sports-commentator prose.

Everything between those two calls (run_pipeline) is plain, deterministic
Python -- no LLM decides what to fetch or in what order. That's the whole
point: the control flow is fixed, so nothing here needs to "reason" about it.

Requires: pip install google-genai python-dotenv
Requires: GEMINI_API_KEY set in a .env file (or the environment).
"""

from __future__ import annotations
import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

import prediction_engine as pe

# Load GEMINI_API_KEY from .env in the same directory as this script
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

MODEL = "gemini-2.5-flash"   # free-tier; swap to "gemini-2.5-pro" for more power


# ---------------------------------------------------------------------------
# LLM call #1 -- parse the question into a structured request
# ---------------------------------------------------------------------------

PARSE_SYSTEM = (
    "You are a structured-data extractor for a football prediction system. "
    "Given a natural-language question about a football match, return ONLY a "
    "valid JSON object with these fields -- no prose, no markdown fences:\n"
    '  "team_a": string (first team, treated as home unless stated otherwise)\n'
    '  "team_b": string (second team)\n'
    '  "competition": string (e.g. "FIFA World Cup", "Friendly"; use "unknown" if not stated)\n'
    '  "venue": "home" or "neutral" ("home" only if team_a is clearly at home)\n'
    '  "query_type": "pre_match" or "live_update" ("live_update" only if match is in progress)'
)


def parse_query(user_text: str) -> dict:
    response = client.models.generate_content(
        model=MODEL,
        contents=user_text,
        config=types.GenerateContentConfig(
            system_instruction=PARSE_SYSTEM,
            response_mime_type="application/json",
            max_output_tokens=300,
        ),
    )
    return json.loads(response.text)


# ---------------------------------------------------------------------------
# LLM call #2 -- narrate the deterministic result
# ---------------------------------------------------------------------------

NARRATION_SYSTEM_PROMPT = (
    "You are a football analyst narrating a match prediction to a fan. "
    "You will be given a JSON object with win probabilities and a predicted "
    "scoreline, already computed by a statistical model. Describe ONLY the "
    "numbers in that JSON -- never invent injuries, form, storylines, or "
    "stats that aren't present in the data. Two to three sentences, "
    "confident sports-commentator tone."
)


def narrate_result(prediction_json: dict) -> str:
    response = client.models.generate_content(
        model=MODEL,
        contents=json.dumps(prediction_json),
        config=types.GenerateContentConfig(
            system_instruction=NARRATION_SYSTEM_PROMPT,
            max_output_tokens=300,
        ),
    )
    return response.text.strip()


# ---------------------------------------------------------------------------
# Deterministic dispatcher -- plain Python, no LLM involved past this point
# ---------------------------------------------------------------------------

def fetch_historical_stats(team: str, stats_lookup: dict) -> dict:
    """
    stats_lookup: a dict you build offline from your Elo backtester + a
    goals-scored/conceded pass over the same historical data, e.g.
        {"Argentina": {"attack": 1.25, "defense": 0.85, "elo": 1978}, ...}
    Replace this stub with a real DB read once you have one.
    """
    if team not in stats_lookup:
        raise KeyError(
            f"No historical stats for '{team}' -- check spelling/aliases, "
            f"or this team isn't in your backtested dataset yet."
        )
    return stats_lookup[team]


def fetch_live_data(team_a: str, team_b: str) -> dict:
    """
    Stub. Replace with a real live-match API call. Expected shape:
        {"minutes_elapsed": int, "goals_a": int, "goals_b": int,
         "possession_diff_a": float, "shots_diff_a": float}
    possession_diff_a / shots_diff_a are from team_a's perspective.
    """
    raise NotImplementedError("Wire this up to your chosen live-data provider.")


def run_pipeline(user_text: str, stats_lookup: dict, league_avg_goals: float = 2.6) -> dict:
    query = parse_query(user_text)
    team_a, team_b = query["team_a"], query["team_b"]
    venue = query["venue"]

    stats_a = fetch_historical_stats(team_a, stats_lookup)
    stats_b = fetch_historical_stats(team_b, stats_lookup)

    pre_match = pe.predict_match(
        home_attack=stats_a["attack"],
        home_defense=stats_a["defense"],
        away_attack=stats_b["attack"],
        away_defense=stats_b["defense"],
        league_avg_goals=league_avg_goals,
        venue=venue,
        home_advantage_factor=1.15,  # placeholder -- calibrate from your own backtest
        rho=-0.08,  # placeholder -- fit via MLE against your own data
    )

    result = pre_match
    if query["query_type"] == "live_update":
        live = fetch_live_data(team_a, team_b)
        result = pe.update_prediction_live(
            pre_match_lambda_home=pre_match["lambda_home"],
            pre_match_lambda_away=pre_match["lambda_away"],
            minutes_elapsed=live["minutes_elapsed"],
            goals_home_so_far=live["goals_a"],
            goals_away_so_far=live["goals_b"],
            possession_diff_home=live["possession_diff_a"],
            shots_diff_home=live["shots_diff_a"],
            rho=-0.08,
        )

    narration = narrate_result(result)
    return {"query": query, "result": result, "narration": narration}


if __name__ == "__main__":
    # Demo stats -- replace with real output from elo_backtester.py + a
    # goals-scored/conceded pass over the same data.
    demo_lookup = {
        "Argentina": {"attack": 1.28, "defense": 0.82, "elo": 2010},
        "Brazil":    {"attack": 1.22, "defense": 0.88, "elo": 1995},
    }
    output = run_pipeline("Who wins between Argentina and Brazil?", demo_lookup)
    print(json.dumps(output, indent=2))
