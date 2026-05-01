import os
import joblib
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
MODEL_PATH = "models/svm_baseline.joblib"
SCALER_PATH = "models/svm_scaler_baseline.joblib"
RESULTS_PATH = "results/svm_baseline_metrics.txt"

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

# 5) First baseline feature set
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

# 6) One-hot encode the simple categoricals
X = pd.get_dummies(
    df[feature_cols],
    columns=["original_language", "status"],
    drop_first=False
)
y = df["target"]

# 7) Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8) Scale features (Crucial for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 9) Baseline SVM
model = SVC(
    probability=True,
    random_state=42,
)

# 10) Train
model.fit(X_train_scaled, y_train)

# 11) Predict
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# 12) Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)

output = []
output.append("SVM Baseline Results")
output.append("=" * 40)
output.append(f"Rows used: {len(df)}")
output.append(f"Feature count after encoding: {X.shape[1]}")
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

joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print(f"\nSaved model to: {MODEL_PATH}")
print(f"Saved scaler to: {SCALER_PATH}")
print(f"Saved metrics to: {RESULTS_PATH}")
