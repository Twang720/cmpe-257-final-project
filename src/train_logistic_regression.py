import os
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# 1) Paths
DATA_PATH = "data/processed/cleaned_tmdb_5000_movies.csv"
MODEL_PATH = "models/logistic_regression_optimized.joblib"
RESULTS_PATH = "results/logistic_regression_results.txt"

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# 2) Load and filter data
df = pd.read_csv(DATA_PATH)
df = df[(df["budget"] > 0) & (df["revenue"] > 0)].copy()

# 3) Build target
df["target"] = (df["revenue"] >= 1.5 * df["budget"]).astype(int)

# 4) Feature Engineering
# Extract Genres
def extract_genres(json_str):
    try:
        genres = json.loads(json_str)
        return [g['name'] for g in genres]
    except:
        return []

df['genre_list'] = df['genres'].apply(extract_genres)
top_genres = pd.Series(sum(df['genre_list'], [])).value_counts().head(10).index
for genre in top_genres:
    df[f'genre_{genre}'] = df['genre_list'].apply(lambda x: 1 if genre in x else 0)

# Language Binning
df['is_english'] = (df['original_language'] == 'en').astype(int)

# Log Transformations
for col in ['budget', 'popularity', 'vote_count']:
    df[f'log_{col}'] = np.log1p(df[col])

# 5) Define Feature Set
feature_cols = [
    'log_budget', 'log_popularity', 'log_vote_count', 
    'runtime', 'vote_average', 'is_english'
] + [f'genre_{g}' for g in top_genres]

X = df[feature_cols].dropna()
y = df.loc[X.index, "target"]

# 6) Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7) Scaling
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8) Train Balanced Logistic Regression
model = LogisticRegression(
    solver="liblinear",
    penalty="l1",
    class_weight="balanced", 
    random_state=42,
    C=0.1
)
model.fit(X_train_scaled, y_train)

# 9) Predict
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# 10) Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)

# 11) Format Output
output = []
output.append("Logistic Regression Optimized Results")
output.append("=" * 40)
output.append(f"Rows used: {len(df)}")
output.append(f"Feature count: {X.shape[1]}")
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

# 12) Save to disk
with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    f.write(result_text)

joblib.dump({"model": model, "scaler": scaler, "features": feature_cols}, MODEL_PATH)

print(f"\nSaved model and scaler to: {MODEL_PATH}")
print(f"Saved metrics to: {RESULTS_PATH}")
