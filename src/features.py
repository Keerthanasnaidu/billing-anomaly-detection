import pandas as pd

# Load the billing dataset
bills = pd.read_csv("data/bills.csv", low_memory=False)

# Calculate consumption ratio
bills["consumption_ratio"] = (
    bills["consumption_kwh"] / bills["baseline_kwh"]
)

# Display the first five records
print("\nConsumption Ratio:")
print(
    bills[
        ["customer_id", "consumption_kwh",
         "baseline_kwh", "consumption_ratio"]
    ].head()
)

# Display basic statistics
print("\nConsumption Ratio Statistics:")
print(bills["consumption_ratio"].describe())

# Sort bills by customer and billing date
bills = bills.sort_values(
    by=["customer_id", "bill_date"]
)

# Calculate previous bill amount for each customer
bills["previous_bill_amount"] = (
    bills.groupby("customer_id")["amount_billed"].shift(1)
)

# Display previous bill amounts
print("\nPrevious Bill Amount:")

print(
    bills[
        ["customer_id", "bill_date",
         "amount_billed", "previous_bill_amount"]
    ].head(10)
)

# Calculate change in bill amount
bills["bill_amount_change"] = (
    bills["amount_billed"] - bills["previous_bill_amount"]
)

print("\nBill Amount Change:")

print(
    bills[
        ["customer_id", "bill_date",
         "amount_billed", "previous_bill_amount",
         "bill_amount_change"]
    ].head(10)
)
# Calculate percentage change in bill amount
bills["bill_amount_percentage_change"] = (
    bills["bill_amount_change"]
    / bills["previous_bill_amount"]
) * 100

print("\nBill Amount Percentage Change:")

print(
    bills[
        ["customer_id", "bill_date",
         "previous_bill_amount",
         "bill_amount_change",
         "bill_amount_percentage_change"]
    ].head(10)
)

# Calculate average consumption over the previous 3 bills
bills["avg_consumption_previous_3"] = (
    bills.groupby("customer_id")["consumption_kwh"]
    .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
)

print("\nAverage Consumption Over Previous 3 Bills:")

print(
    bills[
        ["customer_id", "bill_date",
         "consumption_kwh",
         "avg_consumption_previous_3"]
    ].head(10)
)

# Calculate difference from previous 3-bill average
bills["consumption_vs_recent_average"] = (
    bills["consumption_kwh"] - bills["avg_consumption_previous_3"]
)

print("\nConsumption Compared with Recent Average:")

print(
    bills[
        ["customer_id", "bill_date",
         "consumption_kwh",
         "avg_consumption_previous_3",
         "consumption_vs_recent_average"]
    ].head(10)
)

# Create an indicator for estimated meter readings
bills["is_estimated_read"] = (
    bills["meter_read_type"] == "estimated"
).astype(int)

print("\nEstimated Meter Reading Indicator:")

print(
    bills[
        ["customer_id", "bill_date",
         "meter_read_type", "is_estimated_read"]
    ].head(10)
)

print("\nMeter Reading Indicator Counts:")
print(bills["is_estimated_read"].value_counts())

# Calculate difference from a standard 30-day billing period
bills["billing_period_difference"] = (
    bills["days_in_period"] - 30
)

print("\nBilling Period Difference:")

print(
    bills[
        ["customer_id", "bill_date",
         "days_in_period",
         "billing_period_difference"]
    ].head(10)
)
# Display available tariff categories
print("\nTariff Categories:")
print(bills["tariff"].value_counts())

# One-hot encode tariff categories
bills = pd.get_dummies(
    bills,
    columns=["tariff"],
    prefix="tariff",
    dtype=int
)

print("\nTariff Encoding:")
print(bills.filter(like="tariff_").head())

print("\nNew Dataset Shape:")
print(bills.shape)
# Save the engineered dataset
bills.to_csv("data/bills_with_features.csv", index=False)

print("\nFeature-engineered dataset saved successfully!")