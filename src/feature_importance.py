
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

# 1. Load the trained Random Forest model

model = joblib.load("models/random_forest.pkl")

# 2. Load the feature names

X_train = pd.read_csv("data/X_train.csv")

# 3. Extract feature importance

importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

# 4. Sort by importance

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 15 Most Important Features:")
print(importance_df.head(15).to_string(index=False))

# 5. Plot the top 15 features

top_features = importance_df.head(15).sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 15 Features for Billing Anomaly Detection")
plt.tight_layout()

# 6. Save the plot

Path("data").mkdir(exist_ok=True)

plt.savefig(
    "data/feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nFeature importance plot saved to data/feature_importance.png")

