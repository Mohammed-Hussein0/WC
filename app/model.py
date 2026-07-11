"""
model.py — All training logic: Elo ratings, team memory, feature engineering,
and xG regressor training. Exposes a single train() function that returns a
ModelState object used at prediction time.
"""
import math
import os
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, classification_report


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CUTOFF_DATE = pd.to_datetime("1994-01-01")
HOME_ADVANTAGE = 100
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "results.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_dynamic_rho(competition: str) -> float:
    """Returns the Dixon-Coles rho correlation parameter for a competition."""
    if competition in ["UEFA Euro", "Copa America", "African Cup of Nations", "AFC Asian Cup"]:
        return -0.13
    elif competition == "FIFA World Cup":
        return -0.15
    else:
        return -0.12


def poisson_prob(lam: float, k: int) -> float:
    """Probability of exactly k goals given expected goals lam."""
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def compute_stats(history):
    """
    Computes rolling stats from a team's match history using:
    - Exponential time-decay weighting (gamma=0.95, newest = highest weight)
    - Opponent Elo-adjusted goals (goals vs strong teams count more)
    """
    if not history:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    wins, draws, losses = 0.0, 0.0, 0.0
    goals, conceded = 0.0, 0.0

    gamma = 0.95
    weights = [gamma ** (len(history) - 1 - i) for i in range(len(history))]
    total_weight = sum(weights)

    for i, match in enumerate(history):
        w = weights[i]
        opp_elo = match.get("opponent_elo", 1500.0)
        elo_multiplier = opp_elo / 1500.0

        goals    += match["scored"]   * elo_multiplier * w
        conceded += match["conceded"] / max(0.1, elo_multiplier) * w

        if match["scored"] > match["conceded"]:
            wins += w
        elif match["scored"] < match["conceded"]:
            losses += w
        else:
            draws += w

    scale = len(history) / total_weight
    return wins * scale, draws * scale, losses * scale, goals * scale, conceded * scale


def calculate_match_probs_dc(home_xg: float, away_xg: float, competition: str,
                              rho: float = None, max_goals: int = 10):
    """
    1X2 probabilities via Dixon-Coles adjustment over independent Poisson scorelines.
    Returns (p_draw, p_home, p_away).
    """
    if rho is None:
        rho = get_dynamic_rho(competition)

    prob_home = prob_draw = prob_away = 0.0

    for h in range(max_goals):
        for a in range(max_goals):
            p = poisson_prob(home_xg, h) * poisson_prob(away_xg, a)

            if h == 0 and a == 0:
                p *= max(0, 1 - home_xg * away_xg * rho)
            elif h == 0 and a == 1:
                p *= max(0, 1 + home_xg * rho)
            elif h == 1 and a == 0:
                p *= max(0, 1 + away_xg * rho)
            elif h == 1 and a == 1:
                p *= max(0, 1 - rho)

            if h > a:
                prob_home += p
            elif h == a:
                prob_draw += p
            else:
                prob_away += p

    total = prob_home + prob_draw + prob_away
    return prob_draw / total, prob_home / total, prob_away / total


# ---------------------------------------------------------------------------
# ModelState — the object held in app memory after training
# ---------------------------------------------------------------------------
class ModelState:
    def __init__(self, model_home, model_away, team_memory, elo):
        self.model_home  = model_home
        self.model_away  = model_away
        self.team_memory = team_memory
        self.elo         = elo

    def known_teams(self) -> list[str]:
        return sorted(self.team_memory.keys())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(data_path: str = DATA_PATH) -> ModelState:
    """
    Reads the CSV, builds Elo ratings and team memory chronologically
    (all non-friendly matches from the beginning of history), then trains
    two HistGradientBoostingRegressor models for home/away xG.

    Only matches on or after CUTOFF_DATE are added to the ML feature set.
    """
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=["home_score", "away_score"])

    df["result"] = df.apply(
        lambda x: 0 if x["home_score"] == x["away_score"]
        else (1 if x["home_score"] > x["away_score"] else 2),
        axis=1,
    )

    team_memory = defaultdict(lambda: deque(maxlen=20))
    elo         = defaultdict(lambda: 1500.0)

    features, home_label, away_label, result_label, tournaments = [], [], [], [], []

    for _, row in df.iterrows():
        if row["tournament"] == "Friendly":
            continue

        home = row["home_team"].strip().lower()
        away = row["away_team"].strip().lower()

        home_elo = elo[home]
        away_elo = elo[away]

        home_hist = team_memory[home]
        away_hist = team_memory[away]

        home_wins, home_draws, home_losses, home_goals, home_conceded = compute_stats(home_hist)
        away_wins, away_draws, away_losses, away_goals, away_conceded = compute_stats(away_hist)

        is_neutral = int(row["neutral"])
        effective_home_elo = home_elo + (0 if is_neutral else HOME_ADVANTAGE)

        if row["date"] >= CUTOFF_DATE:
            features.append({
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
            })
            home_label.append(row["home_score"])
            away_label.append(row["away_score"])
            result_label.append(row["result"])
            tournaments.append(row["tournament"])

        # Elo update
        expected_home = 1 / (1 + 10 ** ((away_elo - effective_home_elo) / 400))
        expected_away = 1 - expected_home

        if row["home_score"] > row["away_score"]:
            actual_home, actual_away = 1.0, 0.0
        elif row["home_score"] < row["away_score"]:
            actual_home, actual_away = 0.0, 1.0
        else:
            actual_home = actual_away = 0.5

        if row["tournament"] in ["UEFA Euro", "Copa America"]:
            K = 25
        elif row["tournament"] == "FIFA World Cup":
            K = 30
        else:
            K = 15

        margin      = abs(row["home_score"] - row["away_score"])
        effective_K = K * (math.log(margin + 1) + 1)

        elo[home] = home_elo + effective_K * (actual_home - expected_home)
        elo[away] = away_elo + effective_K * (actual_away - expected_away)

        team_memory[home].append({
            "scored": row["home_score"], "conceded": row["away_score"],
            "opponent": away, "opponent_elo": away_elo,
        })
        team_memory[away].append({
            "scored": row["away_score"], "conceded": row["home_score"],
            "opponent": home, "opponent_elo": home_elo,
        })

    X        = pd.DataFrame(features)
    y_home   = pd.Series(home_label)
    y_away   = pd.Series(away_label)

    X_train, _, y_home_train, _, y_away_train, _ = train_test_split(
        X, y_home, y_away, test_size=0.2, shuffle=False
    )

    model_home = HistGradientBoostingRegressor(loss="poisson", random_state=42)
    model_home.fit(X_train, y_home_train)

    model_away = HistGradientBoostingRegressor(loss="poisson", random_state=42)
    model_away.fit(X_train, y_away_train)

    return ModelState(model_home, model_away, team_memory, elo)
