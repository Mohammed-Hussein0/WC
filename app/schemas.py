from pydantic import BaseModel
from typing import Optional


class PredictRequest(BaseModel):
    home_team: str
    away_team: str
    competition: str = "Friendly"
    is_neutral: int = 0


class PredictResponse(BaseModel):
    home_team: str
    away_team: str
    competition: str
    home_xg: float
    away_xg: float
    most_likely_score: str
    second_likely_score: str
    p_home_win: float
    p_draw: float
    p_away_win: float
    odds_home_win: float
    odds_draw: float
    odds_away_win: float


class EloResponse(BaseModel):
    team_name: str
    elo_rating: float
    form_score: float

class EloRequest(BaseModel):
    team_name: str
