"""
prediction_engine.py

Pure, deterministic match-prediction math. No LLM calls in this file --
everything here is a plain function you can unit test and backtest on its own.

Three layers, matching the refined pipeline:
    1. Elo            -- baseline team strength, with home advantage and a
                          goal-difference-weighted K (matches the eloratings.net
                          methodology for national teams).
    2. Poisson + DC    -- turns attack/defense strength into a full scoreline
                          probability matrix, with the Dixon-Coles correction
                          for the low-score dependency that plain independent
                          Poisson gets wrong (0-0, 1-0, 0-1, 1-1).
    3. Momentum        -- adjusts the REMAINING-time expected goals during a
                          live match. Fixes the bug where a naive implementation
                          re-applies a multiplier to the full 90-minute lambda
                          after the match has already started.

The momentum function here is a simple, clearly-labeled placeholder -- not a
calibrated fuzzy controller. Treat it as a hypothesis to validate, not a
finished component. See the README for how to replace it responsibly.
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# 1. Elo ratings (football-specific: home advantage + goal-difference K)
# ---------------------------------------------------------------------------

HOME_ADVANTAGE_POINTS = 100  # added to the home team's rating before win-expectancy


def elo_expected_score(rating_home: float, rating_away: float, venue: str = "home") -> float:
    """
    Win expectancy for the HOME-listed team (the team passed as `rating_home`).
    venue: "home" applies the +100 home-advantage adjustment; "neutral" does not.
    Use venue="neutral" for tournament matches at a neutral site (e.g. most
    World Cup games involving neither host nation).
    """
    if venue == "home":
        dr = (rating_home + HOME_ADVANTAGE_POINTS) - rating_away
    elif venue == "neutral":
        dr = rating_home - rating_away
    else:
        raise ValueError("venue must be 'home' or 'neutral'")
    return 1.0 / (10 ** (-dr / 400) + 1)


def goal_difference_multiplier(goal_diff: int) -> float:
    """
    K-factor scaling by margin of victory, following the eloratings.net
    convention: unchanged for a 0/1-goal result, x1.5 for two goals,
    x1.75 for three, and x1.75 + (N-3)/8 for four or more.
    """
    n = abs(goal_diff)
    if n <= 1:
        return 1.0
    if n == 2:
        return 1.5
    if n == 3:
        return 1.75
    return 1.75 + (n - 3) / 8


def update_elo(
    rating_home: float,
    rating_away: float,
    goals_home: int,
    goals_away: int,
    k_base: float,
    venue: str = "home",
) -> Tuple[float, float]:
    """
    Returns (new_rating_home, new_rating_away) after one match.
    k_base: tournament weight (eloratings.net uses 60 for World Cup finals,
    50 for continental championships, 40 for qualifiers, 30 for other
    tournaments, 20 for friendlies).
    """
    we_home = elo_expected_score(rating_home, rating_away, venue=venue)
    if goals_home > goals_away:
        s_home = 1.0
    elif goals_home < goals_away:
        s_home = 0.0
    else:
        s_home = 0.5

    k = k_base * goal_difference_multiplier(goals_home - goals_away)
    delta = k * (s_home - we_home)
    return rating_home + delta, rating_away - delta


# ---------------------------------------------------------------------------
# 2. Poisson scoring + Dixon-Coles low-score correction
# ---------------------------------------------------------------------------

def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def dixon_coles_tau(x: int, y: int, lam_home: float, lam_away: float, rho: float) -> float:
    """
    Dixon & Coles (1997) correction factor. Only touches the four low-score
    cells; every other scoreline is untouched (tau = 1). `rho` is fit from
    your own historical data via maximum likelihood -- don't hardcode a
    borrowed value, it's specific to your dataset and era.
    """
    if x == 0 and y == 0:
        return 1 - (lam_home * lam_away * rho)
    if x == 0 and y == 1:
        return 1 + (lam_home * rho)
    if x == 1 and y == 0:
        return 1 + (lam_away * rho)
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def score_matrix(
    lam_home: float, lam_away: float, rho: float = 0.0, max_goals: int = 8
) -> List[List[float]]:
    """
    Full P(home=i, away=j) matrix for i,j in [0, max_goals], DC-corrected and
    renormalized to sum to 1 (the correction alone can shift the total
    slightly away from 1, so we always renormalize after applying it).
    """
    matrix = [
        [poisson_pmf(i, lam_home) * poisson_pmf(j, lam_away) for j in range(max_goals + 1)]
        for i in range(max_goals + 1)
    ]
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            matrix[i][j] *= dixon_coles_tau(i, j, lam_home, lam_away, rho)

    total = sum(sum(row) for row in matrix)
    if total > 0:
        matrix = [[cell / total for cell in row] for row in matrix]
    return matrix


def outcome_probabilities(matrix: List[List[float]]) -> Dict[str, float]:
    home_win = sum(matrix[i][j] for i in range(len(matrix)) for j in range(len(matrix[0])) if i > j)
    draw = sum(matrix[i][i] for i in range(len(matrix)))
    away_win = sum(matrix[i][j] for i in range(len(matrix)) for j in range(len(matrix[0])) if i < j)
    return {"home_win": home_win, "draw": draw, "away_win": away_win}


def most_likely_score(matrix: List[List[float]]) -> Tuple[int, int, float]:
    best_i, best_j, best_p = 0, 0, matrix[0][0]
    for i, row in enumerate(matrix):
        for j, p in enumerate(row):
            if p > best_p:
                best_i, best_j, best_p = i, j, p
    return best_i, best_j, best_p


def expected_goals(
    attack_strength: float,
    opponent_defense_weakness: float,
    league_avg_goals: float,
    home_advantage_factor: float = 1.0,
) -> float:
    """
    lambda = attack strength x opponent's defense weakness x league average
    goals, optionally scaled by a home-advantage factor.

    attack_strength / defense_weakness are each computed relative to the
    league (or competition) average -- e.g. attack_strength = team's average
    goals scored / league average goals. This engine deliberately does NOT
    split attack/defense into separate home and away splits per team, because
    international teams don't play enough matches for that split to be
    reliable (club leagues with 19+ home and 19+ away matches per team are a
    different story -- split it there if you go that route). A single global
    home_advantage_factor is used instead; calibrate it from your own data
    rather than trusting the ~1.1-1.35 range commonly cited.
    """
    lam = attack_strength * opponent_defense_weakness * league_avg_goals
    return lam * home_advantage_factor


# ---------------------------------------------------------------------------
# 3. Live momentum -- corrected for remaining time, pluggable multiplier
# ---------------------------------------------------------------------------

def in_play_expected_goals(
    pre_match_lambda: float,
    minutes_elapsed: float,
    momentum_multiplier: float = 1.0,
    match_length: float = 90.0,
) -> float:
    """
    THE FIX: scales lambda down to the time remaining BEFORE applying the
    momentum multiplier. Returns expected goals for the REST of the match
    only -- goals already scored are certain and must be added separately
    by the caller, not folded back into this Poisson estimate.
    """
    minutes_remaining = max(match_length - minutes_elapsed, 0.0)
    remaining_lambda = pre_match_lambda * (minutes_remaining / match_length)
    return remaining_lambda * momentum_multiplier


def momentum_multiplier(
    possession_diff: float,
    shots_on_target_diff: float,
    possession_weight: float = 0.5,
    shots_weight: float = 0.5,
    scale: float = 0.3,
) -> float:
    """
    PLACEHOLDER. This is a simple logistic squashing of a weighted sum of
    live-state differentials -- it is NOT a calibrated fuzzy controller or a
    trained model, and it has not been validated against real outcomes.

    Inputs are expected as differentials from the perspective of the team
    you're computing lambda for: possession_diff in [-1, 1] (e.g. 0.15 for a
    57.5%/42.5% split), shots_on_target_diff as a raw count difference.

    Replace this with either:
      (a) a scikit-fuzzy control system whose membership functions and rules
          you've tuned and backtested against logged live-match snapshots, or
      (b) a small trained model (logistic regression / gradient boosting) on
          the same features, validated with log loss against actual results.
    Do not ship this default to anything you're relying on.
    """
    weighted = possession_weight * possession_diff + shots_weight * (shots_on_target_diff / 10)
    # logistic squashing centered at 1.0, bounded roughly in (1-scale, 1+scale)
    return 1.0 + scale * (2 / (1 + math.exp(-3 * weighted)) - 1)


# ---------------------------------------------------------------------------
# End-to-end helpers
# ---------------------------------------------------------------------------

def predict_match(
    home_attack: float,
    home_defense: float,
    away_attack: float,
    away_defense: float,
    league_avg_goals: float,
    venue: str = "neutral",
    home_advantage_factor: float = 1.0,
    rho: float = 0.0,
    max_goals: int = 8,
) -> dict:
    """Pre-match prediction. Set home_advantage_factor > 1.0 only when venue == 'home'."""
    factor = home_advantage_factor if venue == "home" else 1.0
    lam_home = expected_goals(home_attack, away_defense, league_avg_goals, factor)
    lam_away = expected_goals(away_attack, home_defense, league_avg_goals, 1.0)

    matrix = score_matrix(lam_home, lam_away, rho=rho, max_goals=max_goals)
    outcomes = outcome_probabilities(matrix)
    best_h, best_a, best_p = most_likely_score(matrix)

    if outcomes["home_win"] > outcomes["away_win"] and outcomes["home_win"] > outcomes["draw"]:
        predicted_winner = "home"
    elif outcomes["away_win"] > outcomes["home_win"] and outcomes["away_win"] > outcomes["draw"]:
        predicted_winner = "away"
    else:
        predicted_winner = "draw"

    return {
        "predicted_winner": predicted_winner,
        "most_likely_score": f"{best_h}-{best_a}",
        "most_likely_score_prob": round(best_p, 4),
        "win_prob_home": round(outcomes["home_win"], 4),
        "draw_prob": round(outcomes["draw"], 4),
        "win_prob_away": round(outcomes["away_win"], 4),
        "lambda_home": round(lam_home, 3),
        "lambda_away": round(lam_away, 3),
    }


def update_prediction_live(
    pre_match_lambda_home: float,
    pre_match_lambda_away: float,
    minutes_elapsed: float,
    goals_home_so_far: int,
    goals_away_so_far: int,
    possession_diff_home: float = 0.0,
    shots_diff_home: float = 0.0,
    rho: float = 0.0,
    max_goals: int = 8,
) -> dict:
    """
    In-play update. Momentum differentials are from the HOME team's perspective;
    the away team gets the negated values automatically.
    """
    mult_home = momentum_multiplier(possession_diff_home, shots_diff_home)
    mult_away = momentum_multiplier(-possession_diff_home, -shots_diff_home)

    remaining_home = in_play_expected_goals(pre_match_lambda_home, minutes_elapsed, mult_home)
    remaining_away = in_play_expected_goals(pre_match_lambda_away, minutes_elapsed, mult_away)

    # Convolve: remaining goals are still Poisson-distributed going forward;
    # already-scored goals are certain, so we shift the matrix rather than
    # re-running Poisson over the full match.
    remaining_matrix = score_matrix(remaining_home, remaining_away, rho=rho, max_goals=max_goals)
    final_matrix_size = max_goals + 1
    final_matrix = [[0.0] * final_matrix_size for _ in range(final_matrix_size)]
    for i in range(len(remaining_matrix)):
        for j in range(len(remaining_matrix[0])):
            fi, fj = i + goals_home_so_far, j + goals_away_so_far
            if fi < final_matrix_size and fj < final_matrix_size:
                final_matrix[fi][fj] += remaining_matrix[i][j]

    outcomes = outcome_probabilities(final_matrix)
    best_h, best_a, best_p = most_likely_score(final_matrix)
    return {
        "most_likely_final_score": f"{best_h}-{best_a}",
        "most_likely_final_score_prob": round(best_p, 4),
        "win_prob_home": round(outcomes["home_win"], 4),
        "draw_prob": round(outcomes["draw"], 4),
        "win_prob_away": round(outcomes["away_win"], 4),
        "remaining_lambda_home": round(remaining_home, 3),
        "remaining_lambda_away": round(remaining_away, 3),
    }
