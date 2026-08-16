"""
predictor.py — Stateless prediction function.
Takes a ModelState and match parameters, returns a structured dict.
"""
import pandas as pd
from app.model import (
    compute_stats, calculate_match_probs_dc,
    get_dynamic_rho, poisson_prob, HOME_ADVANTAGE,
)


def predict_match(
    home_team: str,
    away_team: str,
    state,
    competition: str = "Friendly",
    is_neutral: int = 0,
) -> dict:
    """
    Predict a match outcome and return a structured result dict.
    Raises ValueError if either team is not found in historical data.
    """
    home_team = home_team.strip().lower()
    away_team = away_team.strip().lower()

    if home_team not in state.team_memory:
        raise ValueError(f"Team not found: '{home_team}'")
    if away_team not in state.team_memory:
        raise ValueError(f"Team not found: '{away_team}'")

    home_hist = state.team_memory[home_team]
    away_hist = state.team_memory[away_team]

    home_wins, home_draws, home_losses, home_goals, home_conceded = compute_stats(home_hist)
    away_wins, away_draws, away_losses, away_goals, away_conceded = compute_stats(away_hist)

    home_elo = state.elo[home_team]
    away_elo = state.elo[away_team]

    effective_home_elo = home_elo + (0 if is_neutral else HOME_ADVANTAGE)

    match_features = {
        "home_wins": home_wins, "home_draws": home_draws, "home_losses": home_losses,
        "home_goals": home_goals, "home_conceded": home_conceded,
        "home_goal_diff": home_goals - home_conceded,
        "home_elo": home_elo,
        "elo_diff": effective_home_elo - away_elo,
        "is_neutral": is_neutral,
        "away_wins": away_wins, "away_draws": away_draws, "away_losses": away_losses,
        "away_goals": away_goals, "away_conceded": away_conceded,
        "away_goal_diff": away_goals - away_conceded,
        "away_elo": away_elo,
    }

    X_new = pd.DataFrame([match_features])

    if is_neutral:
        # On neutral ground, average predictions in both orientations to ensure 100% mathematical symmetry
        inv_match_features = {
            "home_wins": away_wins, "home_draws": away_draws, "home_losses": away_losses,
            "home_goals": away_goals, "home_conceded": away_conceded,
            "home_goal_diff": away_goals - away_conceded,
            "home_elo": away_elo,
            "elo_diff": away_elo - home_elo,
            "is_neutral": is_neutral,
            "away_wins": home_wins, "away_draws": home_draws, "away_losses": home_losses,
            "away_goals": home_goals, "away_conceded": home_conceded,
            "away_goal_diff": home_goals - home_conceded,
            "away_elo": home_elo,
        }
        X_inv = pd.DataFrame([inv_match_features])

        xg_home_direct = state.model_home.predict(X_new)[0]
        xg_away_direct = state.model_away.predict(X_new)[0]

        xg_away_as_home = state.model_home.predict(X_inv)[0]
        xg_home_as_away = state.model_away.predict(X_inv)[0]

        home_xg = max(0.01, (xg_home_direct + xg_home_as_away) / 2.0)
        away_xg = max(0.01, (xg_away_direct + xg_away_as_home) / 2.0)
    else:
        home_xg = max(0.01, state.model_home.predict(X_new)[0])
        away_xg = max(0.01, state.model_away.predict(X_new)[0])

    p_draw, p_home, p_away = calculate_match_probs_dc(home_xg, away_xg, competition)

    # Top-2 most likely scorelines
    rho = get_dynamic_rho(competition)
    scores = []
    for h in range(6):
        for a in range(6):
            prob = poisson_prob(home_xg, h) * poisson_prob(away_xg, a)
            if h == 0 and a == 0: prob *= max(0, 1 - home_xg * away_xg * rho)
            elif h == 0 and a == 1: prob *= max(0, 1 + home_xg * rho)
            elif h == 1 and a == 0: prob *= max(0, 1 + away_xg * rho)
            elif h == 1 and a == 1: prob *= max(0, 1 - rho)
            scores.append((prob, h, a))

    scores.sort(reverse=True)
    best  = scores[0]
    second = scores[1]

    return {
        "home_team": home_team,
        "away_team": away_team,
        "competition": competition,
        "home_xg": round(home_xg, 2),
        "away_xg": round(away_xg, 2),
        "most_likely_score": f"{best[1]}-{best[2]}",
        "second_likely_score": f"{second[1]}-{second[2]}",
        "p_home_win": round(p_home, 4),
        "p_draw": round(p_draw, 4),
        "p_away_win": round(p_away, 4),
        "odds_home_win": round(1 / p_home, 2),
        "odds_draw": round(1 / p_draw, 2),
        "odds_away_win": round(1 / p_away, 2),
    }
