#predict
import pandas as pd
import joblib
import os
import re
import logging
import argparse
from datetime import datetime

# === Logging Setup ===
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/predict.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === Preprocessing Function ===
def preprocess_input(data: pd.DataFrame, train_columns: list) -> pd.DataFrame:
    logger.info(" Starting preprocessing...")
    
    data = data.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    if data["Date"].isnull().any():
        raise ValueError(" Invalid or missing date values in input.")

    # Expand date
    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month
    data["Day"] = data["Date"].dt.day
    data["DayOfWeek"] = data["Date"].dt.dayofweek

    # Add dummy fields
    data["Open"] = 1
    data["Customers"] = 0
    data["StateHoliday"] = "0"
    data["PromoInterval"] = "None"

    # Drop unused columns
    drop_cols = ["Date", "Customers", "Open"]
    data.drop(columns=drop_cols, errors="ignore", inplace=True)

    # One-hot encode
    cat_cols = ["StateHoliday", "StoreType", "Assortment", "PromoInterval"]
    data = pd.get_dummies(data, columns=cat_cols)

    # Clean column names
    data.columns = [re.sub(r"[^\w]", "_", str(col)) for col in data.columns]

    # Align with training columns
    missing_cols = set(train_columns) - set(data.columns)
    if missing_cols:
        logger.warning(f"⚠️ Adding missing columns: {missing_cols}")
        for col in missing_cols:
            data[col] = 0

    data = data.reindex(columns=train_columns, fill_value=0)

    if data.isnull().values.any():
        raise ValueError(" Null values found after preprocessing.")

    logger.info("Preprocessing complete.")
    return data

# === Batch Prediction Function ===
def predict_from_file(input_path: str, model_type: str = "xgboost"):
    logger.info(f" Reading input: {input_path}")

    model_path = f"models/{model_type}_model.pkl"
    columns_path = f"models/{model_type}_columns.pkl"

    if not os.path.exists(model_path) or not os.path.exists(columns_path):
        raise FileNotFoundError(f" Missing model or columns file for: {model_type}")

    model = joblib.load(model_path)
    train_columns = joblib.load(columns_path)

    raw_df = pd.read_csv(input_path)
    processed_df = preprocess_input(raw_df, train_columns)
    predictions = model.predict(processed_df)

    logger.info(f"Predictions completed for {len(predictions)} rows.")
    return predictions, raw_df

# === CLI Entry ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw/test.csv", help="Path to input CSV")
    parser.add_argument("--model", type=str, choices=["xgboost", "lightgbm"], default="xgboost", help="Model type to use")
    args = parser.parse_args()

    try:
        preds, input_df = predict_from_file(args.input, model_type=args.model)
        result_df = input_df.copy()
        result_df["Predicted Sales"] = preds

        output_file = f"predictions_{args.model}.csv"
        result_df.to_csv(output_file, index=False)
        logger.info(f"💾 Predictions saved to {output_file}")

    except Exception as e:
        logger.error(f" Prediction failed: {e}")
