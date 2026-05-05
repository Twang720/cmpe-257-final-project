import os
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
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
MODEL_PATH = "models/svm_tuned.joblib"
SCALER_PATH = "models/svm_scaler_tuned.joblib"
RESULTS_PATH = "results/svm_tuned_metrics.txt"

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# 1) Load data
df = pd.read_csv(DATA_PATH)

# 2) Keep only rows where target can be built reliably
df = df[(df["budget"] > 0) & (df["revenue"] > 0)].copy()

# 3) Build target
df["target"] = (df["revenue"] >= 1.5 * df["budget"]).astype(int)

# 4) Create simple date features
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
df["release_year"] = df["release_date"].dt.year
df["release_month"] = df["release_date"].dt.month

# 5) Same feature set as baseline
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

# 6) One-hot encode
X = pd.get_dummies(
    df[feature_cols],
    columns=["original_language", "status"],
    drop_first=False
)
y = df["target"]

# 7) Train / validation / test split
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.20, random_state=42, stratify=y_train_full
)

# 8) Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 9) Randomized search
base_model = SVC(
    probability=True,
    random_state=42,
)

param_dist = {
    "C": [0.1, 1, 10, 50],
    "gamma": ["scale", "auto", 0.01, 0.1],
    "kernel": ["rbf", "linear", "poly"],
    "degree": [2, 3] # only used by poly
}

search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=20,
    scoring="f1_macro",
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

search.fit(X_train_scaled, y_train)
best_model = search.best_estimator_

# 10) Threshold tuning on validation set
val_prob = best_model.predict_proba(X_val_scaled)[:, 1]

best_threshold = 0.50
best_score = -1

for t in np.arange(0.10, 0.91, 0.01):
    val_pred = (val_prob >= t).astype(int)
    score = f1_score(y_val, val_pred, average="macro", zero_division=0)
    if score > best_score:
        best_score = score
        best_threshold = float(t)

# 11) Final test evaluation
test_prob = best_model.predict_proba(X_test_scaled)[:, 1]
test_pred = (test_prob >= best_threshold).astype(int)

acc = accuracy_score(y_test, test_pred)
prec = precision_score(y_test, test_pred, average="macro", zero_division=0)
rec = recall_score(y_test, test_pred, average="macro", zero_division=0)
f1 = f1_score(y_test, test_pred, average="macro", zero_division=0)
auc = roc_auc_score(y_test, test_prob)
cm = confusion_matrix(y_test, test_pred)
report = classification_report(y_test, test_pred, zero_division=0)

output = []
output.append("SVM Tuned Results")
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
joblib.dump(scaler, SCALER_PATH)

print(f"\nSaved model to: {MODEL_PATH}")
print(f"Saved scaler to: {SCALER_PATH}")
print(f"Saved metrics to: {RESULTS_PATH}")
