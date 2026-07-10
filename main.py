import math

import pandas as pd
from collections import defaultdict, deque
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("results.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# remove future/unplayed matches
df = df.dropna(subset=["home_score", "away_score"])

# -----------------------------
# TARGET (LABEL)
# -----------------------------
df["result"] = df.apply(
    lambda x: 0 if x["home_score"] == x["away_score"]
    else (1 if x["home_score"] > x["away_score"] else 2),
    axis=1
)

# -----------------------------
# TEAM MEMORY (ROLLING WINDOW)
# -----------------------------
team_memory = defaultdict(lambda: deque(maxlen=25))
elo = defaultdict(lambda: 1500)

# -----------------------------
# FEATURE STORAGE
# -----------------------------
features = []
labels = []

# -----------------------------
# HELPER: compute stats
# -----------------------------
def compute_stats(history):
    wins,draws,losses = 0,0,0,
    goals = 0
    conceded = 0

    for match in history:
        goals += match["scored"]
        conceded += match["conceded"]

        if match["scored"] > match["conceded"]:
            wins += 1
        elif match["scored"] < match["conceded"]:
            losses += 1
        else:
            draws += 1


    return wins, draws, losses, goals ,conceded

# -----------------------------
# MAIN LOOP (TIME ORDER)
# -----------------------------
for j, row in df.iterrows():

    if row["tournament"] == "Friendly":
         continue

    home = row["home_team"]
    away = row["away_team"]
    

    home_hist = team_memory[home]
    away_hist = team_memory[away]
    
    home_elo = elo[home]
    away_elo = elo[away]
        

   # ... (existing setup code) ...
    home_wins, home_draws, home_losses, home_goals, home_conceded = compute_stats(home_hist)
    away_wins, away_draws, away_losses, away_goals, away_conceded = compute_stats(away_hist)
    
    # 1. Check if the match is neutral
    is_neutral = int(row["neutral"])
    
    HOME_ADVANTAGE = 100
    
    # 2. Calculate Effective Home Elo for this specific match
    # If neutral is True (1), add 0. If False (0), add HOME_ADVANTAGE
    effective_home_elo = home_elo + (0 if is_neutral else HOME_ADVANTAGE)

    # 3. Build feature row (Update elo_diff and add is_neutral)
    features.append({
        "home_wins": home_wins,
        "home_draws": home_draws,
        "home_losses": home_losses,
        "home_goals": home_goals,
        "home_conceded": home_conceded,
        "home_goal_diff": home_goals - home_conceded,
        "home_elo": home_elo, # Keep base Elo for the model to see raw strength

        # Use the effective Elo for the difference so the model understands the true gap
        "elo_diff": effective_home_elo - away_elo, 
        
        # Tell the tree if it's a neutral venue
        "is_neutral": is_neutral, 
      
        "away_wins": away_wins,
        "away_draws": away_draws,
        "away_losses": away_losses,
        "away_goals": away_goals,
        "away_conceded": away_conceded,
        "away_goal_diff": away_goals - away_conceded,
        "away_elo": away_elo,
    })

    labels.append({row["result"]})

    # 4. Calculate expected result using the EFFECTIVE Home Elo
    expected_home = 1 / (1 + 10 ** ((away_elo - effective_home_elo) / 400))
    expected_away = 1 - expected_home
    
    # ... (the rest of your actual_home/away and Elo update code remains exactly the same) ...
    if row["home_score"] > row["away_score"]:
        actual_home = 1
        actual_away = 0

    elif row["home_score"] < row["away_score"]:
        actual_home = 0
        actual_away = 1

    else:
        actual_home = 0.5
        actual_away = 0.5
    
    # 1 k factor based on competition type 
    if row["tournament"] == "Friendly":
     K = 10
     
    elif row["tournament"] == "UEFA Euro" or row["tournament"] == "Copa America":
        K = 25

    elif row["tournament"] == "FIFA World Cup":
        K = 30
        
    else:
        K = 15
        
    #calculating margin based on goal difference
    goal_margin = abs(row["home_score"] - row["away_score"])
    
    margin_multiplier = math.log(goal_margin + 1) + 1
    
    effective_K = K * margin_multiplier
    
    elo[home] = home_elo + effective_K * (actual_home - expected_home)
    elo[away] = away_elo + effective_K * (actual_away - expected_away)
    # -----------------------------
    # UPDATE MEMORY (AFTER MATCH)
    # -----------------------------
    team_memory[home].append({
        "scored": row["home_score"],
        "conceded": row["away_score"],
        "opponent": away,
    })

    team_memory[away].append({
        "scored": row["away_score"],
        "conceded": row["home_score"],
         "opponent": home,

    })

# -----------------------------
# ML DATASET
# -----------------------------
X = pd.DataFrame(features)
y = pd.Series(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=False  # IMPORTANT for time data
)

# -----------------------------
# MODEL
# -----------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
)

model.fit(X_train, y_train)

# -----------------------------
# EVALUATION
# -----------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print()

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})


print(importance.sort_values("Importance", ascending=False))


print(confusion_matrix(y_test, y_pred))
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))