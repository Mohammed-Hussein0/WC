import math

import numpy as np
import pandas as pd
from collections import defaultdict, deque
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def get_dynamic_rho(competition):
    """
    Returns the appropriate rho value based on the competition type.
    This allows for dynamic adjustment of the Dixon-Coles correlation parameter.
    """
    if competition in ["UEFA Euro", "Copa America", "African Cup of Nations", "AFC Asian Cup"]:
        return -0.13
    elif competition == "FIFA World Cup":
        return -0.15
    else:
        return -0.12  # Default for friendlies and qualifiers


def predict_match(home_team, away_team, model_home, model_away, team_memory, elo, competition="Friendly", is_neutral=0):
    """
    Predicts the outcome of a hypothetical match using the most up-to-date 
    team memory and Elo ratings from the end of the dataset.
    """
    # 1. Check if teams exist in memory
    if home_team not in team_memory or away_team not in team_memory:
        return "Error: One or both teams not found in historical data."

    # 2. Fetch the most recent rolling stats and Elo
    home_hist = team_memory[home_team]
    away_hist = team_memory[away_team]
    
    home_wins, home_draws, home_losses, home_goals, home_conceded = compute_stats(home_hist)
    away_wins, away_draws, away_losses, away_goals, away_conceded = compute_stats(away_hist)
    
    home_elo = elo[home_team]
    away_elo = elo[away_team]
    
    # Apply Home Advantage to the Elo difference if not neutral
    HOME_ADVANTAGE = 100
    effective_home_elo = home_elo + (0 if is_neutral else HOME_ADVANTAGE)
    
    # 3. Build the exact feature structure your model expects
    match_features = {
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
    }
    
    # Convert to dataframe for the model
    X_new = pd.DataFrame([match_features])
    
    # 4. Predict Expected Goals (xG)
    home_xg = model_home.predict(X_new)[0]
    away_xg = model_away.predict(X_new)[0]
    
    # Failsafe: Regressors can technically predict negative goals if the team is terrible. Clamp to 0.01.
    home_xg = max(0.01, home_xg)
    away_xg = max(0.01, away_xg)
    
    # 5. Get Match Probabilities using Dixon-Coles
    p_draw, p_home, p_away = calculate_match_probs_dc(home_xg, away_xg, competition)
    
    # 6. Find the Single Most Likely Exact Scoreline
    best_score_prob = 0
    best_score = (0, 0)
    rho = get_dynamic_rho(competition)
    
    for h in range(6):
        for a in range(6):
            prob = poisson_prob(home_xg, h) * poisson_prob(away_xg, a)
            # Apply Dixon-Coles modifiers to the exact scoreline calculation
            if h == 0 and a == 0: prob *= max(0, 1 - (home_xg * away_xg * rho))
            elif h == 0 and a == 1: prob *= max(0, 1 + (home_xg * rho))
            elif h == 1 and a == 0: prob *= max(0, 1 + (away_xg * rho))
            elif h == 1 and a == 1: prob *= max(0, 1 - rho)
            
            if prob > best_score_prob:
                best_score_prob = prob
                best_score = (h, a)
                
    # 7. Print the Professional Betting Output
    print(f"\n{'='*45}")
    print(f" {home_team} vs {away_team} | {competition}")
    print(f"{'='*45}")
    print(f"Expected Goals : {home_team} {home_xg:.2f} - {away_xg:.2f} {away_team}")
    print(f"Most Likely Result : {best_score[0]} - {best_score[1]}")
    print("-" * 45)
    print("WIN PROBABILITIES")
    print(f"{home_team} Win : {p_home*100:.1f}%")
    print(f"Draw       : {p_draw*100:.1f}%")
    print(f"{away_team} Win : {p_away*100:.1f}%")
    print("-" * 45)
    print("FAIR DECIMAL ODDS (Bookmaker Pricing)")
    print(f"{home_team} Win : {1/p_home:.2f}")
    print(f"Draw       : {1/p_draw:.2f}")
    print(f"{away_team} Win : {1/p_away:.2f}")
    print(f"{'='*45}\n")

def run_interactive_predictor(model_home, model_away, team_memory, elo):
    print("\n" + "*"*45)
    print("⚽ WELCOME TO THE ML MATCH PREDICTOR ⚽")
    print("*"*45)
    
    while True:
        print("\n" + "-"*45)
        home_team = input("Enter Home Team (or type 'exit' to quit): ").strip().lower()
        if home_team.lower() in ['exit', 'quit', 'q']:
            print("Exiting predictor. Goodbye!")
            break
            
        away_team = input("Enter Away Team: ").strip().lower()
        
        print("\nSelect Competition Type:")
        print("1. FIFA World Cup (Final Tournament)")
        print("2. Continental Cup (UEFA Euro, Copa America, AFCON, Asian Cup)")
        print("3. Qualifiers (World Cup / Continental)")
        print("4. Nations League")
        print("5. Friendly")
        
        choice = input("Enter choice (1-5): ").strip()
        
        # Map the number to a string that your get_dynamic_rho() function understands
        if choice == "1":
            competition = "FIFA World Cup"
        elif choice == "2":
            competition = "UEFA Euro" # Triggers the -0.13 rho logic
        elif choice == "3":
            competition = "FIFA World Cup qualification" # Triggers the -0.10 rho logic
        elif choice == "4":
            competition = "UEFA Nations League"
        else:
            competition = "Friendly" # Triggers the baseline -0.05 rho logic
            
        neutral_choice = input("\nIs this match on neutral ground? (y/n): ").strip().lower()
        is_neutral = 1 if neutral_choice in ['y', 'yes'] else 0
        
        # Run the prediction
        predict_match(home_team, away_team, model_home, model_away, team_memory, elo, competition, is_neutral)
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

def calculate_match_probs_dc(home_xg, away_xg, com, rho=None, max_goals=10):
    """
    Calculates 1X2 probabilities using the Dixon-Coles adjustment.
    rho (correlation): Negative values artificially increase the probability of 0-0 and 1-1 draws,
    reflecting real-world team psychology (settling for a point).
    """
    if rho is None:
        rho = get_dynamic_rho(com)
    prob_home = 0.0
    prob_draw = 0.0
    prob_away = 0.0
        
        
    for h in range(max_goals):
        for a in range(max_goals):
            # Base independent Poisson probability
            p_scoreline = poisson_prob(home_xg, h) * poisson_prob(away_xg, a)
            
            # ---------------------------------------------------------
            # DIXON-COLES ADJUSTMENT
            # Surgically alters the probabilities of 0-0, 1-0, 0-1, and 1-1
            # ---------------------------------------------------------
            if h == 0 and a == 0:
                tau = max(0, 1 - (home_xg * away_xg * rho))
                p_scoreline *= tau
            elif h == 0 and a == 1:
                tau = max(0, 1 + (home_xg * rho))
                p_scoreline *= tau
            elif h == 1 and a == 0:
                tau = max(0, 1 + (away_xg * rho))
                p_scoreline *= tau
            elif h == 1 and a == 1:
                tau = max(0, 1 - rho)
                p_scoreline *= tau
                
            # Aggregate into 1X2
            if h > a:
                prob_home += p_scoreline
            elif h == a:
                prob_draw += p_scoreline
            else:
                prob_away += p_scoreline
                
    # Normalize probabilities to ensure they sum perfectly to 1.0
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
team_memory = defaultdict(lambda: deque(maxlen=20))
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

    home = row["home_team"].strip().lower()
    away = row["away_team"].strip().lower()
    

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
     
    if row["tournament"] == "UEFA Euro" or row["tournament"] == "Copa America":
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
    
    com = df.iloc[i]["tournament"]
    p_draw, p_home, p_away = calculate_match_probs_dc(h_xg, a_xg, com)
    
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

    
run_interactive_predictor(model_home, model_away, team_memory, elo)