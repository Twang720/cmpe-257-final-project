import os
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score,accuracy_score,precision_score,recall_score,f1_score

DATA_PATH = "src/data/processed/cleaned_tmdb_5000_movies.csv"
MODEL_PATH = "models/logistic_regression_optimized.joblib"
RESULTS_PATH = "results/logistic_regression_results.txt"
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Open the processed data and put it into a dataframe
df = pd.read_csv(DATA_PATH)
df["target"] = (df["revenue"] >= 1.5 * df["budget"]).astype(int)

# Extract the most important genres
df['genre_list'] = [
    [genre['name'] for genre in json.loads(movie_genres)] if isinstance(movie_genres, str) else []
    for movie_genres in df['genres']
]
top_genres = pd.Series(sum(df['genre_list'], [])).value_counts().head(10).index
for genre in top_genres:
    df[f'genre_{genre}'] = df['genre_list'].apply(lambda x: 1 if genre in x else 0)

# Bin the english language movies due to how many there are
df['is_english'] = (df['original_language'] == 'en').astype(int)
for col in ['budget', 'popularity', 'vote_count']:
    df[f'log_{col}'] = np.log1p(df[col])
feature_cols = ['log_budget', 'log_popularity', 'log_vote_count', 'runtime', 'vote_average', 'is_english'] + [f'genre_{g}' for g in top_genres]
movie_features = df[feature_cols].dropna()
profitability_targets = df.loc[movie_features.index, "target"]
(features_train, features_test, targets_train, targets_test) = train_test_split(movie_features, profitability_targets, test_size=0.2, random_state=42, stratify=profitability_targets)

# Use RobustScaler instead of StandardScaler because of the skewed dataset
scaler = RobustScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.transform(features_test)

# Create and train the model
model = LogisticRegression(solver="liblinear",penalty="l1",class_weight="balanced", random_state=42,C=0.1)
model.fit(features_train_scaled, targets_train)
predicted_labels = model.predict(features_test_scaled)
predicted_probabilities = model.predict_proba(features_test_scaled)[:, 1]

# Print out the metrics and store them in results
acc = accuracy_score(targets_test, predicted_labels)
prec = precision_score(targets_test, predicted_labels, zero_division=0)
rec = recall_score(targets_test, predicted_labels, zero_division=0)
f1 = f1_score(targets_test, predicted_labels, zero_division=0)
auc = roc_auc_score(targets_test, predicted_probabilities)
cm = confusion_matrix(targets_test, predicted_labels)
report = classification_report(targets_test, predicted_labels, zero_division=0)

output = []
output.append("Logistic Regression Optimized Results")
output.append("=" * 40)
output.append(f"Rows used: {len(df)}")
output.append(f"Feature count: {movie_features.shape[1]}")
output.append("")
output.append(f"Accuracy : {acc:.4f}")
output.append(f"Precision: {prec:.4f}")
output.append(f"Recall   : {rec:.4f}")
output.append(f"F1-score : {f1:.4f}")
output.append(f"ROC-AUC  : {auc:.4f}")
output.append("")
output.append("Confusion Matrix:")
output.append(str(cm))
output.append("")
output.append("Classification Report:")
output.append(report)
result_text = "\n".join(output)
print(result_text)
with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    f.write(result_text)

# Save the model and store it models
joblib.dump({"model": model, "scaler": scaler, "features": feature_cols}, MODEL_PATH)

print(f"\nSaved model and scaler to: {MODEL_PATH}")
print(f"Saved metrics to: {RESULTS_PATH}")
