import os
import joblib
import pandas as pd
from xgboost import XGBClassifier
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
MODEL_PATH = "models/xgboost_baseline.joblib"
RESULTS_PATH = "results/xgboost_baseline_metrics.txt"

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)

output = []
output.append("XGBoost Baseline Results")
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

# Create graphs for ROC, feature importance and confusion matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from xgboost import plot_importance

os.makedirs("figures", exist_ok=True)

# Confusion Matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["Flop", "Hit"],
    cmap="Blues"
)
plt.title("XGBoost Baseline Confusion Matrix")
plt.tight_layout()
plt.savefig("figures/confusion_matrix_xgb.png", dpi=300)
plt.close()

# ROC Curve
RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title("XGBoost Baseline ROC Curve")
plt.tight_layout()
plt.savefig("figures/roc_curve_xgb.png", dpi=300)
plt.close()

# Feature Importance
plot_importance(model, max_num_features=10)
plt.title("Top 10 XGBoost Feature Importances")
plt.tight_layout()
plt.savefig("figures/feature_importance_xgb.png", dpi=300)
plt.close()



print(f"\nSaved model to: {MODEL_PATH}")
print(f"Saved metrics to: {RESULTS_PATH}")
print("Saved graphs to: figures/")
print(f"\nSaved model to: {MODEL_PATH}")
print(f"Saved metrics to: {RESULTS_PATH}")