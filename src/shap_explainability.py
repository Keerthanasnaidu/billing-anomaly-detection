
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
from pathlib import Path

# 1. Load the trained Random Forest model
model = joblib.load("models/random_forest.pkl")

# 2. Load test data
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").squeeze("columns")

# Use a small sample first to keep SHAP calculations manageable
X_sample = X_test.sample(n=min(200, len(X_test)), random_state=42)

print("Model and test data loaded successfully!")
print("Sample shape:", X_sample.shape)

# 3. Create the SHAP explainer
explainer = shap.TreeExplainer(model)

# 4. Calculate SHAP values
shap_values = explainer.shap_values(X_sample)

print("SHAP values calculated successfully!")

# 5. Handle SHAP output format
# For binary classification, some SHAP versions return a list
# containing one array per class.
if isinstance(shap_values, list):
    anomaly_shap_values = shap_values[1]
elif len(shap_values.shape) == 3:
    anomaly_shap_values = shap_values[:, :, 1]
else:
    anomaly_shap_values = shap_values

# 6. Create the output folder
Path("data").mkdir(exist_ok=True)

# 7. Generate the global feature importance plot
plt.figure()
shap.summary_plot(
    anomaly_shap_values,
    X_sample,
    show=False,
    max_display=15
)

plt.tight_layout()
plt.savefig(
    "data/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Global SHAP plot saved to data/shap_summary.png")

# 8. Explain one individual prediction
sample_index = 0
individual_bill = X_sample.iloc[[sample_index]]

prediction = model.predict(individual_bill)[0]
probability = model.predict_proba(individual_bill)[0, 1]

print("\nIndividual Bill Explanation")
print("Predicted class:", prediction)
print("Anomaly probability:", round(probability, 4))
print("Actual class:", y_test.loc[individual_bill.index[0]])

# 9. Generate a waterfall plot for the individual bill
base_values = explainer.expected_value

if isinstance(base_values, (list, tuple)):
    base_value = base_values[1]
else:
    base_value = base_values

if hasattr(base_value, "__len__"):
    base_value = base_value[1] if len(base_value) > 1 else base_value[0]

if len(anomaly_shap_values.shape) == 3:
    individual_shap_values = anomaly_shap_values[sample_index, :, 1]
else:
    individual_shap_values = anomaly_shap_values[sample_index]

explanation = shap.Explanation(
    values=individual_shap_values,
    base_values=base_value,
    data=individual_bill.iloc[0].values,
    feature_names=individual_bill.columns.tolist()
)

shap.plots.waterfall(explanation, max_display=15, show=False)

plt.savefig(
    "data/shap_individual.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Individual SHAP plot saved to data/shap_individual.png")

