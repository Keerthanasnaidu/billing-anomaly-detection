
import pandas as pd
import joblib

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score
)

# 1. Load the saved Random Forest model

model = joblib.load("models/random_forest.pkl")

# 2. Load the untouched test dataset

X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").squeeze("columns")

print("Test dataset loaded successfully!")
print("Test features:", X_test.shape)
print("Actual anomalies:", int(y_test.sum()))

# 3. Generate predictions

y_test_pred = model.predict(X_test)
y_test_prob = model.predict_proba(X_test)[:, 1]

# 4. Evaluate model performance

print("\nFinal Test Classification Report:")
print(
    classification_report(
        y_test,
        y_test_pred,
        target_names=["Normal", "Anomaly"],
        zero_division=0
    )
)

print("\nTest Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

# 5. Calculate additional metrics

pr_auc = average_precision_score(y_test, y_test_prob)
roc_auc = roc_auc_score(y_test, y_test_prob)

print("\nTest PR-AUC:", round(pr_auc, 4))
print("Test ROC-AUC:", round(roc_auc, 4))

# 6. Save predictions for further analysis

results = X_test.copy()
results["actual_anomaly"] = y_test.values
results["predicted_anomaly"] = y_test_pred
results["anomaly_probability"] = y_test_prob

results.to_csv("data/test_predictions.csv", index=False)

print("\nTest predictions saved to data/test_predictions.csv")

