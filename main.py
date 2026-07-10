import math

import numpy as np
import pandas as pd
from collections import defaultdict, deque
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


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

def poisson_prob(lam, k):
    """Calculates the probability of scoring exactly 'k' goals given expected goals 'lam'"""
    return ((lam ** k) * math.exp(-lam)) / math.factorial(k)

def calculate_match_probs(home_xg, away_xg, max_goals=10):
    """Calculates 1X2 probabilities by checking every possible scoreline"""
    prob_home = 0.0
    prob_draw = 0.0
    prob_away = 0.0
    
    for h in range(max_goals):
        for a in range(max_goals):
            p_scoreline = poisson_prob(home_xg, h) * poisson_prob(away_xg, a)
            
            if h > a:
                prob_home += p_scoreline
            elif h == a:
                prob_draw += p_scoreline
            else:
                prob_away += p_scoreline
                
    total = prob_home + prob_draw + prob_away
    return prob_draw / total, prob_home / total, prob_away / total
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
home_label = []
away_label = []
result_label = []

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

    home_label.append(row["home_score"])
    away_label.append(row["away_score"])
    result_label.append(row["result"])

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
y_home = pd.Series(home_label)
y_away = pd.Series(away_label)
y_result = pd.Series(result_label)

X_train, X_test, y_home_train, y_home_test, y_away_train, y_away_test, y_res_train, y_res_test = train_test_split(
    X, y_home, y_away, y_result,
    test_size=0.2,
    shuffle=False
)

# -----------------------------
# MODEL
# -----------------------------
print("Training Home xG Model...")
model_home = HistGradientBoostingRegressor(loss="poisson", random_state=42)
model_home.fit(X_train, y_home_train)

print("Training Away xG Model...")
model_away = HistGradientBoostingRegressor(loss="poisson", random_state=42)
model_away.fit(X_train, y_away_train)

# -----------------------------
# EVALUATION
# -----------------------------
print("Predicting expected goals...")
pred_home_xg = model_home.predict(X_test)
pred_away_xg = model_away.predict(X_test)

final_predictions = []

for i in range(len(pred_home_xg)):
    h_xg = pred_home_xg[i]
    a_xg = pred_away_xg[i]
    
    p_draw, p_home, p_away = calculate_match_probs(h_xg, a_xg)
    
    probs = [p_draw, p_home, p_away]
    
    # Baseline logic: predict whatever has the highest strict mathematical probability
    predicted_class = np.argmax(probs)
    final_predictions.append(predicted_class)

print("\n--- Poisson Regression 1X2 Evaluation ---")

print("Accuracy:", accuracy_score(y_res_test, final_predictions))

print("\nClassification Report:")
print(classification_report(y_res_test, final_predictions, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_res_test, final_predictions))

print("\n--- Match Sample Analysis ---")
for i in range(3500, 3505):
    if i < len(pred_home_xg):
        print(f"Match {i}:")
        print(f"  Expected Goals: Home {pred_home_xg[i]:.2f} - {pred_away_xg[i]:.2f} Away")
        p_d, p_h, p_a = calculate_match_probs(pred_home_xg[i], pred_away_xg[i])
        print(f"  Probs: Home={p_h:.2f}, Draw={p_d:.2f}, Away={p_a:.2f}")
        print(f"  Predicted Class: {final_predictions[i]} | Actual: {y_res_test.iloc[i]}\n")