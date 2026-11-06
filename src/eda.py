import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load the generated billing dataset
bills = pd.read_csv("data/bills.csv")
print("Loaded file columns:")
print(bills.columns.tolist())

# Count normal and anomalous bills
anomaly_counts = bills["anomaly_type"].apply(
    lambda x: "Anomaly" if x != "normal" else "Normal"
).value_counts()

# Create a bar chart
plt.figure(figsize=(8, 5))

anomaly_counts.plot(kind="bar")

plt.title("Normal Bills vs Anomalous Bills")
plt.xlabel("Bill Category")
plt.ylabel("Number of Bills")
plt.xticks(rotation=0)

plt.tight_layout()

# Save the chart
Path("data").mkdir(exist_ok=True)
plt.savefig("data/anomaly_distribution.png", dpi=300)

# Display the chart
plt.show()
# Visualise individual anomaly categories

anomaly_type_counts = bills[
    bills["anomaly_type"] != "normal"
]["anomaly_type"].value_counts()

plt.figure(figsize=(10, 6))

anomaly_type_counts.plot(kind="bar")

plt.title("Distribution of Billing Anomaly Types")
plt.xlabel("Anomaly Type")
plt.ylabel("Number of Records")
plt.xticks(rotation=30, ha="right")

plt.tight_layout()

# Save the chart
plt.savefig("data/anomaly_types.png", dpi=300)

plt.show()
# Compare billing amounts for normal and anomalous bills

bills["bill_category"] = bills["anomaly_type"].apply(
    lambda x: "Anomaly" if x != "normal" else "Normal"
)

plt.figure(figsize=(8, 6))

bills.boxplot(
    column="amount_billed",
    by="bill_category",
    showfliers=False
)

plt.title("Billing Amount Comparison")
plt.suptitle("")
plt.xlabel("Bill Category")
plt.ylabel("Billing Amount (£)")

plt.tight_layout()

# Save the chart
plt.savefig("data/billing_amount_comparison.png", dpi=300)

plt.show()
# Compare billing amounts across individual anomaly types

anomaly_bills = bills[
    bills["anomaly_type"] != "normal"
]

plt.figure(figsize=(12, 6))

anomaly_bills.boxplot(
    column="amount_billed",
    by="anomaly_type",
    showfliers=False
)

plt.title("Billing Amount by Anomaly Type")
plt.suptitle("")
plt.xlabel("Anomaly Type")
plt.ylabel("Billing Amount (£)")
plt.xticks(rotation=25, ha="right")

plt.tight_layout()

# Save the chart
plt.savefig("data/billing_amount_by_anomaly.png", dpi=300)

plt.show()

print("\nAnomaly type distribution:")
print(anomaly_type_counts)

print("\nBill category counts:")
print(anomaly_counts)

print("\nChart saved to data/anomaly_distribution.png")