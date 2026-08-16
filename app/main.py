"""
main.py — FastAPI application entry point.
Models are trained once at server startup via the lifespan context manager.
"""
from app.schemas import EloResponse
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
import os

from fastapi.middleware.cors import CORSMiddleware
from app.model import train
from app.predictor import predict_match
from app.schemas import PredictRequest, PredictResponse


# ---------------------------------------------------------------------------
# Lifespan: train models once on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Training models — please wait...")
    app.state.model_state = train()
    n = len(app.state.model_state.known_teams())
    print(f"Models ready. {n} teams loaded.")
    yield
    print("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Football Match Predictor",
    description="xG-based match outcome predictor using Elo ratings and Dixon-Coles probabilities.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow requests from any origin (file://, localhost frontends, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the static HTML UI
STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", tags=["UI"])
def root():
    """Serve the interactive HTML UI directly on the main root URL."""
    return RedirectResponse(url="/static/index.html")


@app.get("/health", tags=["Info"])
def health():
    """Health check and service info JSON."""
    state = app.state.model_state
    return {
        "service": "Football Match Predictor",
        "status": "ready",
        "known_teams": len(state.known_teams()),
    }


@app.get("/ui", tags=["UI"])
def ui():
    """Redirect to the static HTML test page."""
    return RedirectResponse(url="/static/index.html")


@app.get("/teams", tags=["Teams"])
def list_teams():
    """Return a sorted list of all known team names."""
    return {"teams": app.state.model_state.known_teams()}


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(req: PredictRequest):
    """
    Predict the outcome of a match between two teams.

    - **home_team**: Team name (case-insensitive, e.g. `brazil`)
    - **away_team**: Team name (case-insensitive, e.g. `argentina`)
    - **competition**: `FIFA World Cup`, `UEFA Euro`, `Copa America`, `Friendly`, etc.
    - **is_neutral**: `0` = home advantage applies, `1` = neutral venue
    """
    try:
        result = predict_match(
            home_team=req.home_team,
            away_team=req.away_team,
            state=app.state.model_state,
            competition=req.competition,
            is_neutral=req.is_neutral,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/elo/{team_name}",response_model=EloResponse, tags=["ELO"])
def get_elo(team_name: str):
    """
    Return the current Elo rating and form score for a given team.
    """
    team_name = team_name.lower().strip()
    if team_name not in app.state.model_state.known_teams():
        raise HTTPException(status_code=404, detail=f"Team '{team_name}' not found")
    return {
        "team_name": team_name,
        "elo_rating": app.state.model_state.elo[team_name],
        "form_score": app.state.model_state.team_memory[team_name],
    }
