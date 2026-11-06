**Billing Anomaly Detection System**

An end-to-end Machine Learning project that identifies unusual patterns and potential anomalies in utility billing data using Random Forest classification.

The system analyses billing information, detects potentially incorrect bills, and explains model predictions using SHAP (SHapley Additive exPlanations).

**Project Highlights**

* Generated a synthetic utility billing dataset containing 120,600 records.
* Engineered features to identify unusual consumption and billing patterns.
* Trained and evaluated Logistic Regression and Random Forest models.
* Achieved 99.43% precision, recall, and F1-score on the synthetic test dataset.
* Integrated SHAP for model explainability.
* Developed a REST API using FastAPI.
* Built an interactive dashboard using Streamlit.

Note: This project uses synthetically generated data. The reported performance reflects evaluation on this dataset and does not represent validated performance on real-world utility billing data.

## Technology Stack

| Technology          | Purpose                                           |
| ------------------- | ------------------------------------------------- |
| Python              | Core programming language                         |
| Pandas              | Data manipulation and preprocessing               |
| NumPy               | Numerical operations                              |
| Scikit-learn        | Machine learning model development and evaluation |
| Random Forest       | Main anomaly classification model                 |
| Logistic Regression | Baseline classification model                     |
| SHAP                | Model explainability                              |
| Matplotlib          | Data visualisation                                |
| Joblib              | Model persistence                                 |
| FastAPI             | Backend REST API                                  |
| Streamlit           | Interactive web dashboard                         |

## Project Workflow

1. **Data Generation:** Created synthetic utility customer and billing records.
2. **Exploratory Data Analysis:** Analysed billing patterns and anomaly distributions.
3. **Feature Engineering:** Created features based on consumption changes, previous bills, and tariff information.
4. **Data Preprocessing:** Handled missing values and split data chronologically into training, validation, and test sets.
5. **Model Training:** Trained Logistic Regression and Random Forest classifiers.
6. **Model Evaluation:** Evaluated model performance using precision, recall, F1-score, PR-AUC, and ROC-AUC.
7. **Explainability:** Used SHAP to understand the features influencing individual predictions.
8. **API Development:** Built a prediction API using FastAPI.
9. **Dashboard Development:** Created an interactive Streamlit dashboard to display predictions and SHAP explanations.


## Dataset Description

This project uses a synthetically generated utility billing dataset containing customer information and billing records.

### Dataset Summary

| Dataset         | Records | Description                           |
| --------------- | ------: | ------------------------------------- |
| Customers       |   5,000 | Synthetic customer information        |
| Bills           | 120,600 | Utility billing transactions          |
| Normal Bills    | 116,939 | Records classified as normal          |
| Anomalous Bills |   3,661 | Records containing injected anomalies |

### Anomaly Types

The dataset includes five types of artificially introduced billing anomalies:

| Anomaly Type            | Records | Description                                                  |
| ----------------------- | ------: | ------------------------------------------------------------ |
| Meter Misread           |   1,800 | Unusual consumption caused by simulated meter-reading errors |
| Wrong Tariff            |     600 | Billing calculated using an incorrect tariff                 |
| Duplicate Bill          |     600 | Duplicate billing transactions                               |
| Manual Adjustment Error |     361 | Incorrect manual billing adjustments                         |
| Estimated Read Catch-up |     300 | Unusual billing changes following estimated readings         |

### Feature Engineering

Additional features were created to help the model identify unusual billing patterns, including:

* Consumption ratio
* Previous bill amount
* Bill amount change
* Bill amount percentage change
* Average consumption over the previous three bills
* Consumption compared with recent averages
* Estimated meter reading indicator
* Billing period difference
* One-hot encoded tariff categories

**Important:** All customer information, billing records, and injected anomalies are synthetic and were created for educational and demonstration purposes.


## Model Development

Two machine learning classification algorithms were trained and evaluated:

### 1. Logistic Regression

Used as a baseline model to establish initial anomaly detection performance.

### 2. Random Forest Classifier

Selected as the primary model because of its strong classification performance and ability to capture complex relationships in billing data.

The dataset was split chronologically into training, validation, and testing sets to simulate predictions on future billing records.

### Model Performance

The Random Forest model achieved the following results on the final test dataset:

| Evaluation Metric |  Score |
| ----------------- | -----: |
| Precision         | 99.43% |
| Recall            | 99.43% |
| F1-Score          | 99.43% |
| PR-AUC            | 99.76% |
| ROC-AUC           | 99.84% |

### Test Set Confusion Matrix

|                | Predicted Normal | Predicted Anomaly |
| -------------- | ---------------: | ----------------: |
| Actual Normal  |           19,218 |                 5 |
| Actual Anomaly |                5 |               868 |

The model correctly identified 868 anomalous records and 19,218 normal records in the test dataset.

**Note:** These results are based on synthetic data and should not be interpreted as expected performance on real-world billing systems.

## Model Explainability with SHAP

To improve model transparency and interpretability, SHAP (SHapley Additive exPlanations) was integrated into the project.

SHAP helps identify which features contribute most to a model's prediction.

### Explainability Features

* **Feature Importance:** Identifies the features that have the greatest overall influence on model predictions.
* **Individual Prediction Explanation:** Explains how individual billing features influence a specific prediction.
* **Positive SHAP Values:** Indicate features pushing the prediction towards the anomaly class.
* **Negative SHAP Values:** Indicate features pushing the prediction towards the normal class.

### Key Influential Features

The model identified several important billing features, including:

* Bill amount percentage change
* Bill amount change
* Unit rate
* Consumption ratio
* Consumption compared with recent averages
* Amount billed
* Consumption in kWh
* Billing adjustments

SHAP visualisations are generated to help users understand the model's behaviour at both overall and individual-record levels.

## Application Architecture

The project includes a FastAPI backend and an interactive Streamlit frontend.

### FastAPI Backend

A REST API was developed using FastAPI to serve predictions from the trained Random Forest model.

Available endpoints:

| Endpoint   | Method | Description                                    |
| ---------- | ------ | ---------------------------------------------- |
| `/`        | GET    | Returns API information                        |
| `/health`  | GET    | Checks API health and model status             |
| `/predict` | POST   | Predicts whether a billing record is anomalous |

The prediction endpoint accepts engineered billing features and returns:

* Prediction label (Normal or Anomaly)
* Prediction class
* Anomaly probability

Interactive API documentation is available through Swagger UI at `/docs`.

### Streamlit Dashboard

An interactive dashboard was developed using Streamlit.

Dashboard features include:

* Model performance metrics
* Billing anomaly distribution visualisations
* Anomaly category analysis
* Test record selection
* Real-time prediction through the FastAPI backend
* Anomaly probability display
* Individual prediction explanations using SHAP
* Feature contribution charts and tables

The dashboard connects to the FastAPI backend to generate predictions and display explainability results.

## Installation and Setup

Follow these steps to run the project locally.

### Prerequisites

* Python 3.10 or later
* Git
* Visual Studio Code (recommended)

### 1. Clone the Repository

```bash
git clone <your-github-repository-url>
cd billing-anomaly
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment.

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn joblib shap fastapi uvicorn streamlit requests
```

### 4. Start the FastAPI Backend

Open a terminal in the project root directory and run:

```bash
uvicorn api.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Swagger API documentation:

```text
http://127.0.0.1:8000/docs
```

### 5. Start the Streamlit Dashboard

Open a second terminal, activate the virtual environment, navigate to the project root, and run:

bash
streamlit run app/app.py


The dashboard will open in your browser.

**Note:** Start the FastAPI backend before launching the Streamlit dashboard. The project also requires its prepared datasets and trained model files to be present in their respective directories.


## Project Structure

```text
billing-anomaly/
│
├── api/
│   └── main.py
│
├── app/
│   └── app.py
│
├── data/
│   ├── customers.csv
│   ├── bills.csv
│   ├── bills_with_features.csv
│   ├── X_train.csv
│   ├── X_validation.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   ├── y_validation.csv
│   ├── y_test.csv
│   ├── test_predictions.csv
│   └── Visualisation and SHAP output files
│
├── models/
│   ├── random_forest.pkl
│   └── logistic_regression.pkl
│
├── notebooks/
│
├── src/
│   ├── generate_data.py
│   ├── features.py
│   ├── prepare_data.py
│   └── shap_explainability.py
│
├── tests/
│
├── .venv/
├── README.md
└── requirements.txt
```
