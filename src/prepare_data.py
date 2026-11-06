import pandas as pd
from sklearn.impute import SimpleImputer

# Load the feature-engineered dataset
bills = pd.read_csv(
    "data/bills_with_features.csv",
    low_memory=False
)

# Display basic information
print("Dataset loaded successfully!")
print("Dataset shape:", bills.shape)

print("\nAvailable columns:")
print(bills.columns.tolist())

print("\nTarget variable distribution:")
print(bills["is_anomaly"].value_counts())
# Define the target variable
y = bills["is_anomaly"]

# Define columns that should not be used as input features
columns_to_drop = [
    "customer_id",
    "bill_id",
    "bill_date",
    "is_anomaly",
    "anomaly_type",
    "anomaly_type"
]

# Create the input features
X = bills.drop(columns=columns_to_drop)
# Identify categorical columns
categorical_columns = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()

print("\nCategorical Columns:")
print(categorical_columns)
# One-hot encode categorical features
X = pd.get_dummies(
    X,
    columns=categorical_columns,
    dtype=int
)

print("\nAfter One-Hot Encoding:")
print("Feature matrix shape:", X.shape)
print("\nRemaining text columns:")
print(X.select_dtypes(include=["object", "str"]).columns.tolist())

print("\nFeature and Target Preparation:")
print("Feature matrix shape:", X.shape)
print("Target vector shape:", y.shape)

print("\nInput features:")
print(X.columns.tolist())
# Check categorical columns
categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nCategorical Columns:")
print(categorical_columns)

# Check missing values
print("\nMissing Values in Features:")
print(X.isnull().sum()[X.isnull().sum() > 0])

# Check data types
print("\nFeature Data Types:")

print(X.dtypes)
# Chronological train, validation, and test split
# Chronological train, validation, and test split
# Month indices range from 0 to 23.

train_mask = bills["month_idx"] <= 15

validation_mask = (
    (bills["month_idx"] >= 16) &
    (bills["month_idx"] <= 19)
)

test_mask = bills["month_idx"] >= 20

# Split the features
X_train = X.loc[train_mask].copy()
X_validation = X.loc[validation_mask].copy()
X_test = X.loc[test_mask].copy()

# Split the target
y_train = y.loc[train_mask].copy()
y_validation = y.loc[validation_mask].copy()
y_test = y.loc[test_mask].copy()

# Display split sizes
print("\nChronological Data Split:")
print("Training features:", X_train.shape)
print("Validation features:", X_validation.shape)
print("Testing features:", X_test.shape)

print("\nAnomalies in Each Split:")
print("Training:", y_train.sum())
print("Validation:", y_validation.sum())
print("Testing:", y_test.sum())

# Verify that all records are included
print("\nTotal records across splits:",
      len(X_train) + len(X_validation) + len(X_test))
# Handle missing values using median imputation

imputer = SimpleImputer(strategy="median")

# Fit only on training data
X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

# Apply the same imputer to validation and test data
X_validation = pd.DataFrame(
    imputer.transform(X_validation),
    columns=X_validation.columns,
    index=X_validation.index
)

X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

# Verify missing values have been handled
print("\nMissing Values After Imputation:")
print("Training:", X_train.isnull().sum().sum())
print("Validation:", X_validation.isnull().sum().sum())
print("Testing:", X_test.isnull().sum().sum())
# Save prepared datasets

X_train.to_csv("data/X_train.csv", index=False)
X_validation.to_csv("data/X_validation.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)

y_train.to_csv("data/y_train.csv", index=False)
y_validation.to_csv("data/y_validation.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

print("\nPrepared datasets saved successfully!")

print("Training:", X_train.shape)
print("Validation:", X_validation.shape)
print("Testing:", X_test.shape)