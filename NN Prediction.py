import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random

# --- TEAM INPUT ---
print("\n=== MLB GAME SIMULATOR ===")
print("Enter the 3-letter acronyms for the two teams (e.g., 'LAD ARI'):")
away_acro, home_acro = input().strip().upper().split()

# --- LOAD DATA ---
try:
    print(f"\nLoading data for {away_acro} (Away) vs {home_acro} (Home)...")
    
    # Away Team (Team 1)
    team1_hitters = pd.read_csv(f"{away_acro}h.csv")
    team1_pitchers = pd.read_csv(f"{away_acro}p.csv")
    
    # Home Team (Team 2)
    team2_hitters = pd.read_csv(f"{home_acro}h.csv")
    team2_pitchers = pd.read_csv(f"{home_acro}p.csv")

except FileNotFoundError as e:
    print(f"\nERROR: Missing data file - {e.filename}")
    print("Please ensure:")
    print(f"1. Files exist: '{away_acro}h.csv', '{away_acro}p.csv', '{home_acro}h.csv', '{home_acro}p.csv'")
    print("2. Team acronyms are correct (e.g., 'LAD' for Dodgers, 'NYM' for Mets)")
    sys.exit()

# --- REST OF YOUR SIMULATION CODE ---
# (Keep your existing data processing, model training, and game simulation logic)
# Check for required columns
for df, name in zip([team1_hitters, team1_pitchers, team2_hitters, team2_pitchers], ['LADh', 'LADp', 'ARIh', 'ARIp']):
    if 'Player' not in df.columns:
        print(f"Error: 'Player' column missing in {name}.csv")
        sys.exit()

# Handle missing values
for df in [team1_hitters, team1_pitchers, team2_hitters, team2_pitchers]:
    df.fillna(df.mean(numeric_only=True), inplace=True)

# Define features
hitter_features = ['AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'BB', 'SO', 'BA', 'OBP', 'SLG', 'OPS']
pitcher_features = [ 'ERA', 'H', 'R', 'ER', 'HR', 'BB', 'WP', 'ERA+', 'WHIP', 'SO/BB']

# Prepare lineups - assuming the CSV files are already ordered by lineup position
team1_lineup = team1_hitters['Player'].tolist()
team2_lineup = team2_hitters['Player'].tolist()

# Select starting pitchers
team1_pitcher = team1_pitchers.iloc[0]['Player']
team2_pitcher = team2_pitchers.iloc[0]['Player']

# Create dictionaries for quick stats lookup
team1_hitter_stats = {row['Player']: row[hitter_features] for i, row in team1_hitters.iterrows()}
team2_hitter_stats = {row['Player']: row[hitter_features] for i, row in team2_hitters.iterrows()}
team1_pitcher_stats = {row['Player']: row[pitcher_features] for i, row in team1_pitchers.iterrows()}
team2_pitcher_stats = {row['Player']: row[pitcher_features] for i, row in team2_pitchers.iterrows()}

# Prepare data for training
num_samples = 100
labels = np.random.choice(
    [0, 1, 2, 3, 4, 5],  # Corresponding to the outcomes
    num_samples,
    p=[0.73, 0.07, 0.09, 0.05, 0.02, 0.04]  # Adjusted probabilities
)

categorical_labels = to_categorical(labels, num_classes=6)
all_features = []

for i in range(num_samples):
    hitter_team = random.choice([team1_hitters, team2_hitters])
    pitcher_team = random.choice([team1_pitchers, team2_pitchers])
    
    hitter_sample_index = random.randint(0, len(hitter_team) - 1)
    pitcher_sample_index = random.randint(0, len(pitcher_team) - 1)
    
    combined_features = np.concatenate([
        hitter_team[hitter_features].iloc[hitter_sample_index].values,
        pitcher_team[pitcher_features].iloc[pitcher_sample_index].values
    ])
    all_features.append(combined_features)

X = np.array(all_features)
X_train, X_test, y_train, y_test = train_test_split(X, categorical_labels, train_size=0.8, random_state=3)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Define and Train the Model ---
model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(6, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#model.fit(X_train, y_train, epochs=200, batch_size=10, validation_data=(X_test, y_test))
    
 
# Calculate average accuracy and loss for each epoch
history = model.fit(X_train, y_train, epochs=200, batch_size=10, validation_data=(X_test, y_test))

# Extract accuracy and loss values
train_accuracies = history.history['accuracy']
val_accuracies = history.history['val_accuracy']
train_losses = history.history['loss']
val_losses = history.history['val_loss']

# Calculate averages
avg_train_accuracy = np.mean(train_accuracies)
avg_val_accuracy = np.mean(val_accuracies)
avg_train_loss = np.mean(train_losses)
avg_val_loss = np.mean(val_losses)

print("\n=== AVERAGE METRICS ===")
print(f"Average Training Accuracy: {avg_train_accuracy:.4f}")
print(f"Average Validation Accuracy: {avg_val_accuracy:.4f}")
print(f"Average Training Loss: {avg_train_loss:.4f}")
print(f"Average Validation Loss: {avg_val_loss:.4f}")
 
def simulate_game(away_team, home_team, away_pitcher, home_pitcher, 
                 away_stats, home_stats, away_pitcher_stats, home_pitcher_stats):
    outcomes = ['out','walk', 'single', 'double', 'triple', 'homerun']
    inning = 1
    outs = 0
    away_score = 0
    home_score = 0
    away_batter_index = 0
    home_batter_index = 0
    
    # Game state
    bases = [None, None, None]  # 1st, 2nd, 3rd
    is_top_inning = True
    
    print("\n=== GAME SIMULATION ===")
    print(f"Away Team vs Home Team")
    print(f"Starting Pitchers: {away_pitcher} vs {home_pitcher}\n")
    
    while inning <= 9 or (inning > 9 and away_score == home_score):  # Play 9 innings or extra innings if tied
        if is_top_inning:
            team = "Away"
            pitcher_stats = home_pitcher_stats[home_pitcher]
            batter = away_team[away_batter_index]
            batter_stats = away_stats[batter]
        else:
            team = "Home"
            pitcher_stats = away_pitcher_stats[away_pitcher]
            batter = home_team[home_batter_index]
            batter_stats = home_stats[batter]
        
        # Prepare input features
        features = np.concatenate([batter_stats.values, pitcher_stats.values]).reshape(1, -1)
        features_scaled = scaler.transform(features)
        
        # Predict outcome
        outcome_probs = model.predict(features_scaled, verbose=0)[0]
        outcome_idx = np.random.choice(len(outcomes), p=outcome_probs)
        outcome = outcomes[outcome_idx]
        
        # Update batter index
        if is_top_inning:
            away_batter_index = (away_batter_index + 1) % len(away_team)
        else:
            home_batter_index = (home_batter_index + 1) % len(home_team)
        
        # Process outcome
        runs_scored = 0
        if outcome == 'out':
            outs += 1
            print(f"{inning}{'T' if is_top_inning else 'B'}: {team} - {batter} vs {home_pitcher if is_top_inning else away_pitcher}: OUT ({outs} out{'s' if outs != 1 else ''})")
            
            if outs % 3 == 0:  # Switch innings after 3 outs
                if not is_top_inning:
                    inning += 1  # Only increment after bottom of inning
                is_top_inning = not is_top_inning
                bases = [None, None, None]
                outs = 0  # Reset outs for new inning
        else:
            # Handle different hit types (same as before)
            if outcome == 'walk':
                # Move runners only if forced
                if bases[0] is not None:
                    if bases[1] is not None:
                        if bases[2] is not None:
                            runs_scored += 1
                        bases[2] = bases[1]
                    bases[1] = bases[0]
                bases[0] = batter
            elif outcome == 'single':
                runs_scored += sum(1 for base in bases[2:] if base is not None)
                bases[2] = bases[1] if bases[1] is not None else None
                bases[1] = bases[0] if bases[0] is not None else None
                bases[0] = batter
            elif outcome == 'double':
                runs_scored += sum(1 for base in bases[1:] if base is not None)
                bases[2] = bases[0] if bases[0] is not None else None
                bases[1] = batter
                bases[0] = None
            elif outcome == 'triple':
                runs_scored += sum(1 for base in bases if base is not None)
                bases[2] = batter
                bases[1] = bases[0] = None
            elif outcome == 'homerun':
                runs_scored += 1 + sum(1 for base in bases if base is not None)
                bases = [None, None, None]
            
            if is_top_inning:
                away_score += runs_scored
            else:
                home_score += runs_scored
            
            print(f"{inning}{'T' if is_top_inning else 'B'}: {team} - {batter} vs {home_pitcher if is_top_inning else away_pitcher}: {outcome.upper()}{' with ' + str(runs_scored) + ' run(s)' if runs_scored > 0 else ''}")
    
    print("\n=== FINAL SCORE ===")
    print(f"Away Team: {away_score}")
    print(f"Home Team: {home_score}")
    print("===================")
# Simulate the game
simulate_game(
    away_team=team1_lineup,
    home_team=team2_lineup,
    away_pitcher=team1_pitcher,
    home_pitcher=team2_pitcher,
    away_stats=team1_hitter_stats,
    home_stats=team2_hitter_stats,
    away_pitcher_stats=team1_pitcher_stats,
    home_pitcher_stats=team2_pitcher_stats
)

