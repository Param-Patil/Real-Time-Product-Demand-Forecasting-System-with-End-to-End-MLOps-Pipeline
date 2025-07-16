import os
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import joblib
import optuna
import re
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# === Paths ===
RAW_TRAIN_PATH = "data/raw/train.csv"
STORE_PATH = "data/raw/store.csv"
MODEL_DIR = "models"
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.pkl")
LGBM_MODEL_PATH = os.path.join(MODEL_DIR, "lightgbm_model.pkl")
XGB_COLUMNS_PATH = os.path.join(MODEL_DIR, "xgboost_columns.pkl")
LGBM_COLUMNS_PATH = os.path.join(MODEL_DIR, "lightgbm_columns.pkl")

# === Load & Merge ===
def load_and_merge_data():
    train = pd.read_csv(RAW_TRAIN_PATH, low_memory=False)
    store = pd.read_csv(STORE_PATH)
    data = pd.merge(train, store, on="Store", how="left")
    data["Date"] = pd.to_datetime(data["Date"])
    return data

# === Preprocessing ===
def preprocess(data):
    data = data[data["Open"] != 0].copy()
    data.fillna(0, inplace=True)

    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month
    data["Day"] = data["Date"].dt.day
    data["DayOfWeek"] = data["Date"].dt.dayofweek

    drop_cols = ["Date", "Customers", "Open"]
    data.drop(columns=drop_cols, inplace=True, errors='ignore')

    categorical_cols = ["StateHoliday", "StoreType", "Assortment", "PromoInterval"]
    data = pd.get_dummies(data, columns=categorical_cols)

    data.columns = [re.sub(r"[^\w]", "_", str(col)) for col in data.columns]
    return data

# === Train Model ===
def train_model(data, model_type="xgboost"):
    print(f"\n🔍 Running Optuna for {model_type.upper()}...")

    X = data.drop(columns=["Sales"])
    y = data["Sales"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    os.makedirs(MODEL_DIR, exist_ok=True)

    columns_path = XGB_COLUMNS_PATH if model_type == "xgboost" else LGBM_COLUMNS_PATH
    joblib.dump(X_train.columns.tolist(), columns_path)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }

        model = (
            xgb.XGBRegressor(**params, random_state=42, verbosity=0)
            if model_type == "xgboost"
            else lgb.LGBMRegressor(**params, random_state=42)
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)
    best_params = study.best_params
    print(f" Best Params for {model_type.upper()}: {best_params}")

    model = (
        xgb.XGBRegressor(**best_params, random_state=42)
        if model_type == "xgboost"
        else lgb.LGBMRegressor(**best_params, random_state=42)
    )
    model_path = XGB_MODEL_PATH if model_type == "xgboost" else LGBM_MODEL_PATH

    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, preds))
    mae = mean_absolute_error(y_val, preds)
    r2 = r2_score(y_val, preds)

    print(f" {model_type.upper()} → RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")

    # Save model and metrics
    joblib.dump(model, model_path)
    metrics = {
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "r2": round(r2, 4)
    }
    metrics_path = os.path.join(MODEL_DIR, f"{model_type}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)

    print(f" Model saved to {model_path}")
    print(f" Metrics saved to {metrics_path}")
    print(f" Columns saved to {columns_path}")

    return rmse

# === CLI Entrypoint ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["xgboost", "lightgbm", "all"], default="xgboost")
    args = parser.parse_args()

    print(" Loading and preprocessing data...")
    raw_data = load_and_merge_data()
    processed_data = preprocess(raw_data)

    if args.model == "xgboost":
        train_model(processed_data, model_type="xgboost")
    elif args.model == "lightgbm":
        train_model(processed_data, model_type="lightgbm")
    elif args.model == "all":
        rmse_xgb = train_model(processed_data, model_type="xgboost")
        rmse_lgb = train_model(processed_data, model_type="lightgbm")
        print(f"\n🏁 RMSE Comparison → XGBoost: {rmse_xgb:.2f} | LightGBM: {rmse_lgb:.2f}")
