
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

# 1. Load the prepared datasets

X_train = pd.read_csv("data/X_train.csv")
X_validation = pd.read_csv("data/X_validation.csv")
X_test = pd.read_csv("data/X_test.csv")

y_train = pd.read_csv("data/y_train.csv").squeeze("columns")
y_validation = pd.read_csv("data/y_validation.csv").squeeze("columns")
y_test = pd.read_csv("data/y_test.csv").squeeze("columns")

# 2. Load the existing Logistic Regression model

logistic_model = joblib.load("models/logistic_regression.pkl")

# 3. Create the Random Forest model

random_forest = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# 4. Train Random Forest

print("Training Random Forest...")
random_forest.fit(X_train, y_train)

print("Random Forest training completed!")

# 5. Evaluate both models on validation data

models = {
    "Logistic Regression": logistic_model,
    "Random Forest": random_forest
}

results = []

for name, model in models.items():

    predictions = model.predict(X_validation)
    probabilities = model.predict_proba(X_validation)[:, 1]

    results.append({
        "Model": name,
        "Precision": precision_score(
            y_validation, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_validation, predictions, zero_division=0
        ),
        "F1-score": f1_score(
            y_validation, predictions, zero_division=0
        ),
        "PR-AUC": average_precision_score(
            y_validation, probabilities
        )
    })

# 6. Display the comparison

results_df = pd.DataFrame(results)

print("\nModel Comparison Results:")
print(results_df.round(4).to_string(index=False))

# 7. Save the Random Forest model

joblib.dump(random_forest, "models/random_forest.pkl")

print("\nRandom Forest model saved successfully!")

