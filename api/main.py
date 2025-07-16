from fastapi import FastAPI, Query
from pydantic import BaseModel
import pandas as pd
import re
import joblib
import os
import logging
from prometheus_fastapi_instrumentator import Instrumentator
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
import sys

# === FastAPI App ===
app = FastAPI(title=" Product Demand Forecasting API")

# === CORS for Streamlit Frontend ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Logging Setup (UTF-8 safe with emoji support) ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(BASE_DIR, "models")      # for render MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "api.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# === Input Schema ===
class SaleRecord(BaseModel):
    Store: int
    DayOfWeek: int
    Promo: int
    SchoolHoliday: int
    StoreType: str
    Assortment: str
    CompetitionDistance: float
    Promo2: int
    Date: str  # Format: YYYY-MM-DD


# === Preprocessing Function ===
def preprocess(record: SaleRecord, train_columns):
    df = pd.DataFrame([record.dict()])

    # Parse date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isnull().any():
        raise ValueError("Invalid date format. Use YYYY-MM-DD.")

    # Extract date features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek

    # Add dummy cols
    df["Open"] = 1
    df["Customers"] = 0
    df["StateHoliday"] = "0"
    df["PromoInterval"] = "None"

    # Drop unused
    df.drop(columns=["Date", "Customers", "Open"], errors="ignore", inplace=True)

    # One-hot encode
    df = pd.get_dummies(df, columns=["StateHoliday", "StoreType", "Assortment", "PromoInterval"])

    # Clean column names
    df.columns = [re.sub(r"[^\w]", "_", str(col)) for col in df.columns]

    # Reindex
    missing_cols = set(train_columns) - set(df.columns)
    if missing_cols:
        logger.warning(f" Missing columns filled with 0: {missing_cols}")
    df = df.reindex(columns=train_columns, fill_value=0)

    if df.isnull().values.any():
        logger.error(" Null values detected after preprocessing.")

    return df


# === Root Health Check ===
@app.get("/")
def read_root():
    return {"message": " Product Demand Forecasting API is live."}


# === Inference Endpoint ===
@app.post("/predict")
def predict_sale(
    record: SaleRecord,
    model_type: str = Query("xgboost", enum=["xgboost", "lightgbm"])
):
    try:
        model_path = os.path.join(MODELS_DIR, f"{model_type}_model.pkl")
        columns_path = os.path.join(MODELS_DIR, f"{model_type}_columns.pkl")

        if not os.path.exists(model_path) or not os.path.exists(columns_path):
            logger.error(f" Model or columns file not found for '{model_type}'.")
            return {"error": f" Model or columns file missing for {model_type}."}

        # Load model and column list
        model = joblib.load(model_path)
        train_columns = joblib.load(columns_path)

        # Preprocess input
        data = preprocess(record, train_columns)

        logger.info(f"Input Record: {record.dict()}")
        logger.info(f" Processed Columns: {list(data.columns)}")

        # Predict
        prediction = model.predict(data)[0]
        logger.info(f" {model_type.upper()} Prediction: {prediction:.2f}")

        return {"predicted_sales": round(float(prediction), 2)}

    except Exception as e:
        logger.exception(" Exception during prediction.")
        return {"error": f" Prediction failed: {str(e)}"}


# === List Available Models (Optional Helper) ===
@app.get("/models")
def list_models():
    files = os.listdir(MODELS_DIR)
    model_list = sorted(set(f.split("_")[0] for f in files if f.endswith("_model.pkl")))
    return {"available_models": model_list}


# === Prometheus Monitoring ===
Instrumentator().instrument(app).expose(app)
