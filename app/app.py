from pathlib import Path

import joblib
import shap
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_DIR / "data" / "X_test.csv"
MODEL_PATH = PROJECT_DIR / "models" / "random_forest.pkl"

API_URL = "http://127.0.0.1:8000/predict"


# --------------------------------------------------
# 2. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Billing Anomaly Detection",
    page_icon="🔍",
    layout="wide"
)


# --------------------------------------------------
# 3. Load model and SHAP explainer
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_explainer():
    return shap.TreeExplainer(load_model())


model = load_model()


# --------------------------------------------------
# 4. Session state
# --------------------------------------------------

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "analysis_features" not in st.session_state:
    st.session_state.analysis_features = None

if "analysis_explanation" not in st.session_state:
    st.session_state.analysis_explanation = None

if "analysis_row" not in st.session_state:
    st.session_state.analysis_row = None

if "analysis_error" not in st.session_state:
    st.session_state.analysis_error = None


# --------------------------------------------------
# 5. Sidebar
# --------------------------------------------------

with st.sidebar:

    st.title("🔍 Billing AI")

    st.write("Anomaly Detection System")

    st.divider()

    st.subheader("About the Project")

    st.write(
        "This project uses Machine Learning to identify "
        "unusual patterns in synthetic utility billing data."
    )

    st.write("**Model:** Random Forest")
    st.write("**Task:** Binary Classification")

    st.divider()

    st.subheader("Technology Stack")

    st.write("""
    - Python
    - Pandas
    - Scikit-learn
    - SHAP
    - FastAPI
    - Streamlit
    """)

    st.divider()

    st.caption("MSc Artificial Intelligence Portfolio Project")


# --------------------------------------------------
# 6. Main heading
# --------------------------------------------------

st.title("🔍 Billing Anomaly Detection System")

st.write(
    "An intelligent billing analysis platform that identifies "
    "potential anomalies using Machine Learning."
)

st.info(
    "Demo environment: Predictions use sample records from a "
    "synthetic utility billing dataset."
)


# --------------------------------------------------
# 7. Model performance
# --------------------------------------------------

st.subheader("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Precision", "99.43%")
col2.metric("Recall", "99.43%")
col3.metric("F1-Score", "99.43%")
col4.metric("PR-AUC", "99.76%")

st.caption(
    "Metrics are from the held-out synthetic test dataset. "
    "They do not represent verified real-world performance."
)

st.divider()


# --------------------------------------------------
# 8. Charts
# --------------------------------------------------

chart_col1, chart_col2 = st.columns(2)

with chart_col1:

    st.subheader("Anomaly Distribution")

    labels = ["Normal", "Anomaly"]
    values = [116939, 3661]

    fig, ax = plt.subplots(figsize=(3, 2))

    ax.bar(labels, values)

    ax.set_ylabel("Number of Bills")
    ax.set_title("Normal vs Anomalous Bills")

    plt.tight_layout()

    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


with chart_col2:

    st.subheader("Anomaly Types")

    anomaly_types = [
        "Meter Misread",
        "Wrong Tariff",
        "Duplicate Bill",
        "Manual Adjustment Error",
        "Estimated Read Catchup"
    ]

    anomaly_counts = [1800, 600, 600, 361, 300]

    fig2, ax2 = plt.subplots(figsize=(4, 2))

    ax2.barh(anomaly_types, anomaly_counts)

    ax2.set_xlabel("Number of Bills")
    ax2.set_title("Anomaly Type Breakdown")

    plt.tight_layout()

    st.pyplot(fig2, use_container_width=False)
    plt.close(fig2)


st.divider()


# --------------------------------------------------
# 9. Load test data
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


# --------------------------------------------------
# 10. Generate SHAP explanation
# --------------------------------------------------

def generate_shap_explanation(features):

    explainer = load_explainer()

    input_data = pd.DataFrame(
        [features],
        columns=model.feature_names_in_
    )

    shap_values = explainer.shap_values(
        input_data,
        check_additivity=False
    )

    # Identify the anomaly class (class 1)
    anomaly_class_index = list(model.classes_).index(1)

    if isinstance(shap_values, list):

        anomaly_shap = shap_values[anomaly_class_index][0]

    elif len(shap_values.shape) == 3:

        anomaly_shap = shap_values[0, :, anomaly_class_index]

    else:

        anomaly_shap = shap_values[0]

    explanation_df = pd.DataFrame({
        "Feature": input_data.columns,
        "SHAP Impact": anomaly_shap
    })

    explanation_df["Absolute Impact"] = (
        explanation_df["SHAP Impact"].abs()
    )

    explanation_df = explanation_df.sort_values(
        "Absolute Impact",
        ascending=False
    ).head(10)

    return explanation_df


# --------------------------------------------------
# 11. Billing record selection
# --------------------------------------------------

try:

    df = load_data()

    st.subheader("🔎 Analyse a Billing Record")

    st.write(
        "Select a sample record from the test dataset "
        "to check whether it contains unusual billing patterns."
    )

    row_number = st.number_input(
        "Select test record number",
        min_value=1,
        max_value=len(df),
        value=1,
        step=1
    )

    selected_record = df.iloc[int(row_number) - 1]

    with st.expander("View Model Input Features"):

        st.dataframe(selected_record.to_frame("Value"))


    # --------------------------------------------------
    # 12. Analyse button
    # --------------------------------------------------

    if st.button(
        "🚀 Analyse Billing Record",
        use_container_width=True
    ):

        features = selected_record.astype(float).to_dict()

        # Clear previous error
        st.session_state.analysis_error = None

        try:

            with st.spinner("Analysing billing record..."):

                response = requests.post(
                    API_URL,
                    json={"features": features},
                    timeout=30
                )

            if response.status_code == 200:

                result = response.json()

                # Save prediction in session state
                st.session_state.analysis_result = result

                st.session_state.analysis_features = features

                st.session_state.analysis_row = int(row_number)

                # Generate and save SHAP explanation
                with st.spinner("Generating SHAP explanation..."):

                    try:

                        explanation_df = generate_shap_explanation(
                            features
                        )

                        st.session_state.analysis_explanation = (
                            explanation_df
                        )

                    except Exception as e:

                        st.session_state.analysis_explanation = None

                        st.session_state.analysis_error = (
                            f"SHAP explanation error: {e}"
                        )

            else:

                st.session_state.analysis_error = (
                    f"API Error: {response.text}"
                )

        except requests.exceptions.ConnectionError:

            st.session_state.analysis_error = (
                "Cannot connect to the API. "
                "Please make sure FastAPI is running."
            )

        except requests.exceptions.RequestException as e:

            st.session_state.analysis_error = (
                f"Request failed: {e}"
            )


    # --------------------------------------------------
    # 13. Display saved prediction
    # --------------------------------------------------

    if st.session_state.analysis_error:

        st.error(st.session_state.analysis_error)


    if st.session_state.analysis_result is not None:

        result = st.session_state.analysis_result

        probability = result["anomaly_probability"]

        st.divider()

        st.subheader("📋 Prediction Results")

        st.caption(
            f"Last analysed test record: "
            f"#{st.session_state.analysis_row}"
        )

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            if result["prediction"] == 1:

                st.error("⚠️ Anomaly Detected")

            else:

                st.success("✅ Normal Bill")

        with result_col2:

            st.metric(
                "Anomaly Probability",
                f"{probability * 100:.2f}%"
            )

        st.write("### Anomaly Probability Indicator")

        st.progress(probability)

        st.caption(
            "Higher probability indicates a greater estimated "
            "likelihood of an anomaly."
        )


        # --------------------------------------------------
        # 14. Display saved SHAP explanation
        # --------------------------------------------------

        st.divider()

        st.subheader(
            "🧠 Why Did the Model Make This Prediction?"
        )

        explanation_df = st.session_state.analysis_explanation

        if explanation_df is not None:

            fig3, ax3 = plt.subplots(figsize=(9, 5))

            ax3.barh(
                explanation_df["Feature"],
                explanation_df["SHAP Impact"]
            )

            ax3.set_xlabel("SHAP Value")

            ax3.set_title(
                "Top 10 Features Influencing the Prediction"
            )

            ax3.axvline(0, linewidth=0.8)

            ax3.invert_yaxis()

            plt.tight_layout()

            st.pyplot(fig3)

            plt.close(fig3)

            st.caption(
                "Positive SHAP values push the model towards "
                "predicting an anomaly. Negative SHAP values "
                "push it towards predicting a normal bill."
            )

            st.write("### Feature Contribution Details")

            st.dataframe(
                explanation_df[
                    ["Feature", "SHAP Impact"]
                ].reset_index(drop=True),
                use_container_width=True
            )


except Exception as e:

    st.error(f"Something went wrong: {e}")


# --------------------------------------------------
# 15. Footer
# --------------------------------------------------

st.divider()

st.markdown(
    """
    <div style="text-align: center; color: grey;">
        <p>Billing Anomaly Detection System</p>
        <p>Built with Python | Machine Learning | FastAPI | Streamlit | SHAP</p>
        <p>MSc Artificial Intelligence Project</p>
    </div>
    """,
    unsafe_allow_html=True
)