import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

DATA_PATH = "src/data/processed/cleaned_tmdb_5000_movies.csv"
MODEL_PATH = "models/xgboost_tuned.joblib"
RESULTS_PATH = "results/xgboost_tuned_metrics.txt"

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


df = pd.read_csv(DATA_PATH)
df = df[(df["budget"] > 0) & (df["revenue"] > 0)].copy()
df["target"] = (df["revenue"] >= 1.5 * df["budget"]).astype(int)
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
df["release_year"] = df["release_date"].dt.year
df["release_month"] = df["release_date"].dt.month


feature_cols = [
    "budget",
    "popularity",
    "runtime",
    "vote_average",
    "vote_count",
    "release_year",
    "release_month",
    "original_language",
    "status",
]

df = df[feature_cols + ["target"]].dropna().copy()

X = pd.get_dummies(
    df[feature_cols],
    columns=["original_language", "status"],
    drop_first=False
)
y = df["target"]

# Train / validation / test split
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.20, random_state=42, stratify=y_train_full
)

# Randomized search
base_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=4
)

param_dist = {
    "n_estimators": [200, 400, 600, 800],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "min_child_weight": [1, 3, 5, 7],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "gamma": [0, 0.1, 0.3, 1.0],
    "reg_alpha": [0, 0.1, 1.0],
    "reg_lambda": [1, 3, 10],
}

search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=20,
    scoring="f1_macro",
    cv=5,
    random_state=42,
    n_jobs=1,
    verbose=1
)

search.fit(X_train, y_train)
best_model = search.best_estimator_

val_prob = best_model.predict_proba(X_val)[:, 1]

best_threshold = 0.50
best_score = -1

for t in np.arange(0.10, 0.91, 0.01):
    val_pred = (val_prob >= t).astype(int)
    score = f1_score(y_val, val_pred, average="macro", zero_division=0)
    if score > best_score:
        best_score = score
        best_threshold = float(t)

# 10) Final test evaluation
test_prob = best_model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= best_threshold).astype(int)

acc = accuracy_score(y_test, test_pred)
prec = precision_score(y_test, test_pred, average="macro", zero_division=0)
rec = recall_score(y_test, test_pred, average="macro", zero_division=0)
f1 = f1_score(y_test, test_pred, average="macro", zero_division=0)
auc = roc_auc_score(y_test, test_prob)
cm = confusion_matrix(y_test, test_pred)
report = classification_report(y_test, test_pred, zero_division=0)

output = []
output.append("XGBoost Tuned Results")
output.append("=" * 40)
output.append(f"Rows used: {len(df)}")
output.append(f"Feature count after encoding: {X.shape[1]}")
output.append(f"Best params: {search.best_params_}")
output.append(f"Best threshold: {best_threshold:.2f}")
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

joblib.dump(best_model, MODEL_PATH)

print(f"\nSaved model to: {MODEL_PATH}")
print(f"Saved metrics to: {RESULTS_PATH}")
