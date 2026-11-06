
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score
)

# 1. Load prepared datasets

X_train = pd.read_csv("data/X_train.csv")
X_validation = pd.read_csv("data/X_validation.csv")
X_test = pd.read_csv("data/X_test.csv")

y_train = pd.read_csv("data/y_train.csv").squeeze("columns")
y_validation = pd.read_csv("data/y_validation.csv").squeeze("columns")
y_test = pd.read_csv("data/y_test.csv").squeeze("columns")

print("Datasets loaded successfully!")
print("Training data:", X_train.shape)
print("Validation data:", X_validation.shape)
print("Testing data:", X_test.shape)

# 2. Create a pipeline with feature scaling and Logistic Regression

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42
    ))
])

# 3. Train the model

model.fit(X_train, y_train)

print("\nModel training completed!")

# 4. Evaluate on validation data

y_validation_pred = model.predict(X_validation)
y_validation_prob = model.predict_proba(X_validation)[:, 1]

print("\nValidation Results:")
print(
    classification_report(
        y_validation,
        y_validation_pred,
        zero_division=0
    )
)

print("\nValidation Confusion Matrix:")
print(confusion_matrix(y_validation, y_validation_pred))

print(
    "\nValidation PR-AUC (Average Precision):",
    round(
        average_precision_score(y_validation, y_validation_prob),
        4
    )
)

import joblib
from pathlib import Path

# Save the trained model

Path("models").mkdir(exist_ok=True)

joblib.dump(model, "models/logistic_regression.pkl")

print("\nModel saved successfully!")
print("Location: models/logistic_regression.pkl")



