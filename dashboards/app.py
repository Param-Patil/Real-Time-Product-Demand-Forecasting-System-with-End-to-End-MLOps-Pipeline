#app
import streamlit as st
import pandas as pd
import requests
import os
import plotly.express as px
from dotenv import load_dotenv

# === Page Config ===
st.set_page_config(page_title=" Product Demand Forecasting", layout="wide")

# === Load Environment Variables ===
load_dotenv()
API_BASE = os.getenv("API_URL", "https://api1-g6d4.onrender.com/predict")  # use http://localhost:8000 for local if running render then use backend app render url

# === App Title ===
st.title(" Product Demand Forecasting App")

# === Sidebar Model Selector ===
model_type = st.sidebar.selectbox(" Select Model", ["xgboost", "lightgbm"])

# === Tab Navigation ===
tab_metrics, tab1, tab2 = st.tabs([" Model Evaluation", "🔍 Single Prediction", "📤 Batch Prediction"])
# === tab_metrics ===
import json

with tab_metrics:
    st.subheader(f" {model_type.upper()} Model Evaluation Metrics")

    def load_metrics(model_type):
        metrics_path = os.path.join("..", "models", f"{model_type}_metrics.json")
        try:
            with open(metrics_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {"error": f"Metrics file not found for {model_type}."}
        except Exception as e:
            return {"error": str(e)}

    metrics = load_metrics(model_type)

    if "error" in metrics:
        st.error(metrics["error"])
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric(" RMSE", metrics["rmse"])
        col2.metric(" MAE", metrics["mae"])
        col3.metric("R² Score", metrics["r2"])
        st.success(" Metrics loaded successfully.")

        st.markdown("""
        ---
        📌 **Interpretation**
        - **RMSE** (Root Mean Squared Error) indicates average prediction error.
        - **MAE** (Mean Absolute Error) is the average magnitude of errors.
        - **R²** shows how well the model explains sales variability (1 = perfect).
        """)


# ----------------------------
#  SINGLE RECORD PREDICTION
# ----------------------------
with tab1:
    st.subheader(" Predict Sales for a Single Record")

    with st.form("single_prediction_form"):
        Store = st.number_input("Store ID", value=1)
        DayOfWeek = st.slider("Day of Week (1=Mon ... 7=Sun)", 1, 7, 1)
        Promo = st.selectbox("Promo", [0, 1])
        SchoolHoliday = st.selectbox("School Holiday", [0, 1])
        StoreType = st.selectbox("Store Type", ["a", "b", "c", "d"])
        Assortment = st.selectbox("Assortment", ["a", "b", "c"])
        CompetitionDistance = st.number_input("Competition Distance", value=100.0)
        Promo2 = st.selectbox("Promo2", [0, 1])
        Date = st.date_input("Forecast Date")

        submitted = st.form_submit_button("Predict")

    if submitted:
        payload = {
            "Store": Store,
            "DayOfWeek": DayOfWeek,
            "Promo": Promo,
            "SchoolHoliday": SchoolHoliday,
            "StoreType": StoreType,
            "Assortment": Assortment,
            "CompetitionDistance": CompetitionDistance,
            "Promo2": Promo2,
            "Date": str(Date)
        }

        try:
            response = requests.post(f"{API_BASE}/predict?model_type={model_type}", json=payload)
            if response.status_code == 200:
                result = response.json()
                prediction = result.get("predicted_sales")
                if prediction is not None:
                    st.success(f" Predicted Sales: **{prediction}**")
                else:
                    st.warning(" No prediction returned.")
            else:
                st.error(f" API Error [{response.status_code}]: {response.text}")
        except Exception as e:
            st.error(f" Failed to connect to API: {e}")

# ----------------------------
#  BATCH PREDICTION
# ----------------------------
with tab2:
    st.subheader(" Predict Sales from CSV File")
    uploaded_file = st.file_uploader("Upload CSV (must contain required columns)", type=["csv"])

    def preprocess_and_predict_batch(df: pd.DataFrame):
        required_cols = [
            "Store", "Promo", "SchoolHoliday", "StoreType",
            "Assortment", "CompetitionDistance", "Promo2", "Date"
        ]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            st.error(f" Missing required columns: {missing}")
            return None

        # Convert dates and calculate day features
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df.dropna(subset=["Date"], inplace=True)
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day
        df["DayOfWeek"] = df["Date"].dt.dayofweek

        predictions = []
        for _, row in df.iterrows():
            record = {
                "Store": int(row["Store"]),
                "DayOfWeek": int(row["DayOfWeek"]),
                "Promo": int(row["Promo"]),
                "SchoolHoliday": int(row["SchoolHoliday"]),
                "StoreType": str(row["StoreType"]),
                "Assortment": str(row["Assortment"]),
                "CompetitionDistance": float(row["CompetitionDistance"]),
                "Promo2": int(row["Promo2"]),
                "Date": row["Date"].strftime("%Y-%m-%d")
            }

            try:
                res = requests.post(f"{API_BASE}/predict?model_type={model_type}", json=record)
                if res.status_code == 200:
                    predictions.append(res.json().get("predicted_sales"))
                else:
                    predictions.append(None)
            except Exception as e:
                predictions.append(None)
                st.warning(f" API error for row: {e}")

        df["Predicted Sales"] = predictions
        return df

    if uploaded_file:
        try:
            df_input = pd.read_csv(uploaded_file)
            with st.spinner(" Predicting sales..."):
                result_df = preprocess_and_predict_batch(df_input.copy())

            if result_df is not None:
                st.success(" Batch prediction completed.")
                st.dataframe(result_df.head(10))

                #  Line chart
                fig = px.line(
                    result_df.sort_values("Date"),
                    x="Date",
                    y="Predicted Sales",
                    title=f" Forecasted Sales with {model_type.upper()}",
                    markers=True
                )
                fig.update_layout(xaxis_title="Date", yaxis_title="Predicted Sales")
                st.plotly_chart(fig, use_container_width=True)

                #  Download
                st.download_button(
                    label="⬇ Download Results as CSV",
                    data=result_df.to_csv(index=False),
                    file_name="predicted_sales.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f" Error processing uploaded CSV: {e}")
