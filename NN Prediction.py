import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Load the datasets (ensure 'Player' column exists)
try:
    hitters_df = pd.read_csv('hitters.csv')
    pitchers_df = pd.read_csv('pitchers.csv')
except FileNotFoundError:
    print("Error: 'hitters.csv' or 'pitchers.csv' not found.")
    sys.exit()
if 'Player' not in hitters_df.columns or 'Player' not in pitchers_df.columns:
    print("Error: 'Player' column missing in one or both CSV files.")
    sys.exit()

# Handle missing values
hitters_df = hitters_df.fillna(hitters_df.mean(numeric_only=True))
pitchers_df = pitchers_df.fillna(pitchers_df.mean(numeric_only=True))

# Define features
hitter_features = ['Age', 'WAR', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'BA', 'OBP', 'SLG', 'OPS', 'OPS+', 'rOBA', 'Rbat+', 'TB', 'GIDP', 'HBP', 'SH', 'SF']
pitcher_features = ['Age', 'WAR', 'W', 'L', 'ERA', 'G', 'GS', 'GF', 'CG', 'SHO', 'SV', 'IP', 'H', 'R', 'ER', 'HR', 'BB', 'IBB', 'SO', 'HBP', 'BK', 'WP', 'BF', 'ERA+', 'FIP', 'WHIP', 'H9', 'HR9', 'BB9', 'SO9', 'SO/BB']


top_hitter = hitters_df.sort_values(by='Rk', ascending=True).iloc[0]
top_hitter_name = top_hitter['Player']
top_hitter_stats = top_hitter[hitter_features].values


top_pitcher = pitchers_df.sort_values(by='Rk', ascending=True).iloc[0]
top_pitcher_name = top_pitcher['Player']
top_pitcher_stats = top_pitcher[pitcher_features].values

# Combine stats for the top matchup
input_stats_top = np.concatenate([top_hitter_stats, top_pitcher_stats]).reshape(1, -1)

# Prepare data for training
num_samples = 100
labels = np.random.randint(0, 6, num_samples)
categorical_labels = to_categorical(labels, num_classes=6)
all_features = []
for i in range(num_samples):
    hitter_sample_index = np.random.randint(0, len(hitters_df))
    pitcher_sample_index = np.random.randint(0, len(pitchers_df))
    combined_features = np.concatenate([
        hitters_df[hitter_features].iloc[hitter_sample_index].values,
        pitchers_df[pitcher_features].iloc[pitcher_sample_index].values
    ])
    all_features.append(combined_features)
X = np.array(all_features)
X_train, X_test, y_train, y_test = train_test_split(X, categorical_labels, test_size=0.8, random_state=3)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
input_stats_top_scaled = scaler.transform(input_stats_top) # Scale the top matchup data

# --- Define and Train the Model (Adjusted Class Weights) ---
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(6, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define class weights to favor 'out'
# Assuming the order of outcomes is ['walk', 'out', 'single', 'double', 'triple', 'homerun']
class_weights = {
    0: 0.8,  # walk
    1: 7.33,  # out (increased weight)
    2: 0.7,  # single
    3: 0.6,  # double
    4: 0.1,  # triple
    5: 0.4   # homerun
}

model.fit(X_train, y_train, epochs=75, batch_size=5, class_weight=class_weights, validation_data=(X_test, y_test))

# --- Predict the outcome for the top matchup ---
outcome_probabilities_top = model.predict(input_stats_top_scaled)
outcome_index_top = np.argmax(outcome_probabilities_top)
outcomes = ['walk', 'out', 'single', 'double', 'triple', 'homerun']
predicted_outcome_top = outcomes[outcome_index_top]

print("\n--- Prediction for Top Hitter vs. Top Pitcher ---")
print(f"Top Hitter: {top_hitter_name}")
print(f"Top Pitcher: {top_pitcher_name}")
print(f"Predicted Outcome: {predicted_outcome_top}")

# --- After training the model, add this block to save it ---
model.save('baseball_outcome_model.h5')
print("Trained model saved to baseball_outcome_model.h5")

import joblib
joblib.dump(scaler, 'scaler.pkl')
print("Scaler object saved to scaler.pkl")

# --- If you want to save the top matchup prediction as well ---
import pandas as pd
top_matchup_prediction = pd.DataFrame({
    'Top Hitter': [top_hitter_name],
    'Top Pitcher': [top_pitcher_name],
    'Predicted Outcome': [predicted_outcome_top]
})
top_matchup_prediction.to_csv('top_matchup_prediction.csv', index=False)
