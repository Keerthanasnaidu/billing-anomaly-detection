import numpy as np
import pandas as pd
from pathlib import Path

# Project settings
SEED = 42
N_CUSTOMERS = 5000
N_MONTHS = 24

START = "2023-01-01"

BASE_RATE = {
    "STD": 0.28,
    "FIXED": 0.25,
    "EV": 0.20,
    "COM_A": 0.24,
    "COM_B": 0.21
}

STANDING = {
    "STD": 0.60,
    "FIXED": 0.55,
    "EV": 0.65,
    "COM_A": 1.20,
    "COM_B": 1.00
}
PRICE_STEPS = pd.Series({
    pd.Timestamp("2023-01-01"): 1.00,
    pd.Timestamp("2023-04-01"): 1.05,
    pd.Timestamp("2023-10-01"): 1.10,
    pd.Timestamp("2024-04-01"): 1.04,
    pd.Timestamp("2024-10-01"): 1.08
})

# Create a reproducible random number generator
rng = np.random.default_rng(SEED)


def make_customers():

    n = N_CUSTOMERS

    customers = pd.DataFrame({
        "customer_id": np.arange(1, n + 1),

        "customer_type": rng.choice(
            ["residential", "commercial"],
            n,
            p=[0.85, 0.15]
        ),

        "region": rng.choice(
            ["North", "Midlands", "South", "London", "Scotland"],
            n,
            p=[0.20, 0.20, 0.25, 0.20, 0.15]
        )
    })

    residential_tariff = rng.choice(
        ["STD", "FIXED", "EV"],
        n,
        p=[0.6, 0.3, 0.1]
    )

    commercial_tariff = rng.choice(
        ["COM_A", "COM_B"],
        n,
        p=[0.6, 0.4]
    )

    customers["tariff"] = np.where(
        customers["customer_type"] == "residential",
        residential_tariff,
        commercial_tariff
    )

    customers["baseline_kwh"] = np.where(
        customers["customer_type"] == "residential",

        rng.lognormal(np.log(300), 0.4, n),

        rng.lognormal(np.log(2500), 0.7, n)
    )

    customers["seasonality_amp"] = rng.uniform(0.15, 0.45, n)

    customers["estimated_read_prob"] = rng.beta(1, 12, n)

    return customers

def make_bills(customers):

    # Generate 24 monthly dates
    months = pd.date_range(
        START,
        periods=N_MONTHS,
        freq="MS"
    )

    monthly_data = pd.DataFrame({
        "bill_date": months,
        "month_idx": np.arange(N_MONTHS)
    })

    # Combine every customer with every month
    df = customers.merge(
        monthly_data,
        how="cross"
    )

    # Generate unique bill IDs
    df["bill_id"] = np.arange(len(df))

    # Calculate billing period
    df["days_in_period"] = df.bill_date.dt.days_in_month

    # Calculate seasonal consumption
    seasonal = 1 + df.seasonality_amp * np.cos(
        2 * np.pi * (df.bill_date.dt.month - 1) / 12
    )

    smooth = (
        df.baseline_kwh
        * seasonal
        * (df.days_in_period / 30.4)
    )

    # Add realistic consumption variations
    noisy = smooth * rng.lognormal(
        0,
        0.12,
        len(df)
    )

    # Generate meter reading types
    estimated = rng.random(len(df)) < df.estimated_read_prob

    df["meter_read_type"] = np.where(
        estimated,
        "estimated",
        "actual"
    )

    df["consumption_kwh"] = np.where(
        estimated,
        smooth,
        noisy
    )

    # Calculate electricity rates
    price_multiplier = PRICE_STEPS.reindex(
        months,
        method="ffill"
    )

    df["price_mult"] = df.bill_date.map(price_multiplier)

    base = df.tariff.map(BASE_RATE)

    df["unit_rate"] = np.where(
        df.tariff == "FIXED",
        base,
        base * df.price_mult
    ).round(4)

    df["standing_charge"] = df.tariff.map(STANDING)

    # Generate small billing adjustments
    has_adjustment = rng.random(len(df)) < 0.03

    df["adjustments"] = np.where(
        has_adjustment,
        rng.normal(0, 5, len(df)).round(2),
        0.0
    )

    # Generate payment status
    df["payment_status"] = rng.choice(
        ["paid", "pending", "overdue"],
        len(df),
        p=[0.85, 0.10, 0.05]
    )

    return df.drop(columns=["price_mult"])

def inject_anomalies(bills):
    # Make a copy so the original data remains unchanged
    bills = bills.copy()

    # Initially, every bill is normal
    bills["is_anomaly"] = 0
    bills["anomaly_type"] = "normal"

    # Select approximately 1.5% of bills
    n_anomalies = int(len(bills) * 0.015)

    anomaly_indices = rng.choice(
        bills.index,
        size=n_anomalies,
        replace=False
    )

    # Introduce incorrect meter readings
    bills.loc[anomaly_indices, "consumption_kwh"] *= rng.uniform(
        3.0, 5.0, size=n_anomalies
    )

    bills.loc[anomaly_indices, "is_anomaly"] = 1
    bills.loc[anomaly_indices, "anomaly_type"] = "meter_misread"

    # Recalculate the bill amount
    bills["amount_billed"] = (
        bills["consumption_kwh"] * bills["unit_rate"]
        + bills["standing_charge"] * bills["days_in_period"]
        + bills["adjustments"]
    )
    # Introduce wrong tariff anomalies
    n_wrong_tariff = int(len(bills) * 0.005)
    
    wrong_tariff_indices = rng.choice(
            bills.index[bills["is_anomaly"] == 0],
            size=n_wrong_tariff,
            replace=False
        )
    
    bills.loc[wrong_tariff_indices, "unit_rate"] *= 1.5
    
    bills.loc[wrong_tariff_indices, "is_anomaly"] = 1
    bills.loc[wrong_tariff_indices, "anomaly_type"] = "wrong_tariff"
    
        # Recalculate the bill amount
    bills["amount_billed"] = (
            bills["consumption_kwh"] * bills["unit_rate"]
            + bills["standing_charge"] * bills["days_in_period"]
            + bills["adjustments"]
        )
    
    # Introduce duplicate bill anomalies
    n_duplicates = int(len(bills) * 0.005)

    duplicate_indices = rng.choice(
        bills.index[bills["is_anomaly"] == 0],
        size=n_duplicates,
        replace=False
    )

    # Copy selected billing records
    duplicate_bills = bills.loc[duplicate_indices].copy()

    # Give duplicate records unique bill IDs
    duplicate_bills["bill_id"] = (
        duplicate_bills["bill_id"].astype(str) + "_DUP"
    )

    # Mark duplicate records as anomalies
    duplicate_bills["is_anomaly"] = 1
    duplicate_bills["anomaly_type"] = "duplicate_bill"

    # Add duplicate records to the dataset
    bills = pd.concat(
        [bills, duplicate_bills],
        ignore_index=True
    )

    # Introduce estimated-read catch-up anomalies

    # Work only with original monthly records
    monthly_bills = bills[
        ~bills["bill_id"].astype(str).str.endswith("_DUP")
    ].copy()

    # Find customers with estimated readings
    estimated_customers = monthly_bills.loc[
        monthly_bills["meter_read_type"] == "estimated",
        "customer_id"
    ].unique()

    # Select a small group of customers
    n_catchup = min(300, len(estimated_customers))

    catchup_customers = rng.choice(
        estimated_customers,
        size=n_catchup,
        replace=False
    )

    for customer_id in catchup_customers:
        customer_bills = monthly_bills[
            (monthly_bills["customer_id"] == customer_id)
            & (monthly_bills["meter_read_type"] == "actual")
            & (monthly_bills["is_anomaly"] == 0)
        ]

        if not customer_bills.empty:
            idx = customer_bills.index[-1]

            bills.loc[idx, "consumption_kwh"] *= 2.5
            bills.loc[idx, "is_anomaly"] = 1
            bills.loc[idx, "anomaly_type"] = "estimated_read_catchup"

    # Recalculate billing amounts
    bills["amount_billed"] = (
        bills["consumption_kwh"] * bills["unit_rate"]
        + bills["standing_charge"] * bills["days_in_period"]
        + bills["adjustments"]
    )
        # Introduce manual adjustment errors

    n_adjustment_errors = int(len(bills) * 0.003)

    adjustment_indices = rng.choice(
        bills.index[bills["is_anomaly"] == 0],
        size=n_adjustment_errors,
        replace=False
    )

    # Introduce unusually large manual adjustments
    bills.loc[adjustment_indices, "adjustments"] += rng.uniform(
        200, 500, size=n_adjustment_errors
    )

    # Mark these bills as anomalies
    bills.loc[adjustment_indices, "is_anomaly"] = 1
    bills.loc[adjustment_indices, "anomaly_type"] = "manual_adjustment_error"

    # Recalculate billing amounts
    bills["amount_billed"] = (
        bills["consumption_kwh"] * bills["unit_rate"]
        + bills["standing_charge"] * bills["days_in_period"]
        + bills["adjustments"]
    )





    return bills

    
def main():

    # Generate customer records
    customers = make_customers()

    # Generate billing records
    bills = make_bills(customers)

    # Introduce anomalies
    bills = inject_anomalies(bills)

    # Create data folder
    Path("data").mkdir(exist_ok=True)

    # Save datasets
    customers.to_csv("data/customers.csv", index=False)
    bills.to_csv("data/bills.csv", index=False)

    # Data validation
    print("\nMissing values:")
    print(bills.isnull().sum().sum())

    print("\nDuplicate bill IDs:")
    print(bills["bill_id"].duplicated().sum())

    print("\nInconsistent anomaly labels:")

    inconsistent_labels = (
        ((bills["anomaly_type"] == "normal") & (bills["is_anomaly"] != 0))
        |
        ((bills["anomaly_type"] != "normal") & (bills["is_anomaly"] != 1))
    ).sum()

    print(inconsistent_labels)

    # Dataset information
    print("\nCustomer dataset shape:", customers.shape)
    print("Billing dataset shape:", bills.shape)

    print("\nAnomaly distribution:")
    print(bills["anomaly_type"].value_counts())

    print("\nAnomaly percentage:")
    print(bills["is_anomaly"].mean() * 100)

    # Confirm saved file location and columns
    print("\nSaved billing file to:", Path("data/bills.csv").resolve())
    print("Saved billing columns:", bills.columns.tolist())


if __name__ == "__main__":
    main()