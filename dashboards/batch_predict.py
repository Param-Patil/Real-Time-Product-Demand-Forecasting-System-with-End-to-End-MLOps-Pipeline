#batch_predict
import streamlit as st
import pandas as pd
import joblib
import re
import os
import plotly.express as px

st.set_page_config(page_title=" Batch Prediction - Product Forecast", layout="wide")
st.title(" Batch Product Demand Prediction")

# === Model Selector ===
model_type = st.selectbox("Choose Model", ["xgboost", "lightgbm"])

uploaded_file = st.file_uploader(" Upload CSV File", type=["csv"])

def preprocess_batch(df: pd.DataFrame, train_columns: list) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek

    df["Open"] = 1
    df["Customers"] = 0
    df["StateHoliday"] = "0"
    df["PromoInterval"] = "None"

    drop_cols = ["Date", "Customers", "Open"]
    df.drop(columns=drop_cols, errors="ignore", inplace=True)

    df = pd.get_dummies(df, columns=["StateHoliday", "StoreType", "Assortment", "PromoInterval"])
    df.columns = [re.sub(r"[^\w]", "_", str(col)) for col in df.columns]

    missing_cols = set(train_columns) - set(df.columns)
    for col in missing_cols:
        df[col] = 0
    df = df.reindex(columns=train_columns, fill_value=0)

    return df

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(" Uploaded Data Preview", df.head())

    try:
        model_path = f"models/{model_type}_model.pkl"
        columns_path = f"models/{model_type}_columns.pkl"

        if not os.path.exists(model_path) or not os.path.exists(columns_path):
            st.error(f" Required model or columns not found for `{model_type}`.")
        else:
            model = joblib.load(model_path)
            train_columns = joblib.load(columns_path)

            processed_df = preprocess_batch(df, train_columns)
            predictions = model.predict(processed_df)
            df["Predicted Sales"] = predictions

            st.success(" Prediction Completed")
            st.dataframe(df.head(10))

            fig = px.line(df.sort_values("Date"), x="Date", y="Predicted Sales",
                          title=f"Forecasted Sales - {model_type.upper()}", markers=True)
            st.plotly_chart(fig, use_container_width=True)

            st.download_button(
                "⬇ Download Predictions CSV",
                data=df.to_csv(index=False),
                file_name=f"predictions_{model_type}.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error during prediction: {e}")
