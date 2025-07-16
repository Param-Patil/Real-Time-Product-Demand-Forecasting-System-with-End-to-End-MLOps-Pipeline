
##  Product Demand Forecasting System with MLOps 

An end-to-end Machine Learning system to forecast daily product sales across retail stores using historical data. Designed with full MLOps lifecycle in mind — from data ingestion to automated retraining and real-time monitoring.
Product Demand Forecasting System with MLOps — Built a full-stack ML pipeline (XGBoost/LightGBM) with FastAPI, Streamlit, Airflow, Docker, and Prometheus for real-time sales forecasting and automated retraining.
Developed a production-ready time series demand forecasting pipeline using XGBoost & LightGBM, achieving accurate sales predictions for Rossmann stores using historical, promotional, and seasonal features.
Implemented modular ML architecture with FastAPI for real-time inference, Streamlit dashboard for batch/single prediction UI, and Optuna for hyperparameter optimization.
Automated model retraining via Apache Airflow and deployed containerized services using Docker Compose with Prometheus-Grafana monitoring.
Enabled full MLOps lifecycle: data ingestion, model training, batch/real-time prediction, monitoring, and retraining, all deployable on AWS EC2 or Render.

Deployed App Links:
[Live Backend app](https://fastapi-c4xn.onrender.com),
[Live Frontend app](https://stramlit-dashboard.onrender.com)

 Dataset

This project uses the [Rossmann Store Sales dataset](https://www.kaggle.com/competitions/rossmann-store-sales/data) from Kaggle.

     The raw dataset files (train.csv, store.csv, etc.) are not included in this repository. Please download them manually from Kaggle and place them in the data/raw/ directory.

```markdown
#  Product Demand Forecasting System with MLOps

An end-to-end MLOps pipeline to forecast daily product sales for retail stores using historical data. This project covers the full lifecycle: data ingestion, model training, real-time prediction, monitoring, and deployment.

> Built with XGBoost, LightGBM, FastAPI, Streamlit, Docker, Airflow, and Prometheus — deployable on AWS EC2 or Render.

---

##  Objective

Forecast daily sales for Rossmann stores using historical sales, promotions, holidays, and store-level metadata. The system supports:

-  Real-time API predictions (FastAPI)
-  Interactive dashboard (Streamlit)
-  Automated retraining (Airflow)
-  Monitoring (Prometheus + Grafana)
-  Dockerized deployment (local + cloud)
- CI/CD automation (Docker + GitHub-ready)

---

##  Features

- End-to-end forecasting pipeline with feature engineering
- ML models: XGBoost & LightGBM (Optuna-tuned)
- REST API via FastAPI for single predictions
- Streamlit dashboard for single & batch predictions
- Airflow-based monthly retraining DAG
- Prometheus/Grafana for API monitoring
- Dockerized with `docker-compose`
- Central model store + deployment syncing via script

---

##  Project Structure

```bash
product-demand-forecasting-mlops/
├── api/
│   ├── main.py
│   └── models/
│       ├── xgboost_model.pkl
│       ├── xgboost_columns.pkl
│       ├── lightgbm_model.pkl
│       └── lightgbm_columns.pkl
│
├── dashboards/
│   ├── app.py
│   ├── batch_predict.py
│   └── models/
│       ├── xgboost_model.pkl
│       ├── xgboost_columns.pkl
│       ├── lightgbm_model.pkl
│       └── lightgbm_columns.pkl
│
├── models/
│   ├── xgboost_model.pkl
│   ├── xgboost_columns.pkl
│   ├── lightgbm_model.pkl
│   └── lightgbm_columns.pkl
│
├── data/
│   ├── raw/
│   │   └── train.csv         # (to be added by user)
│   │   └── store.csv         # (to be added by user)
│   └── processed/            # (optional - created during preprocessing)
│
├── prepare_deploy.py
├── requirements.txt
├── Dockerfile
└── .gitignore
<<<<<<< HEAD
=======

>>>>>>> 23034f2 ( Remove previously tracked files now ignored)
````
---

##  Train Model (CLI)

```bash
# Train both models:
python src/train.py --model all

# Train XGBoost or LightGBM individually:
python src/train.py --model xgboost
python src/train.py --model lightgbm
````

>  Outputs saved in `models/` directory.

---

## Copy Models to API & Dashboard

After training, sync model files to both the FastAPI and Streamlit folders:

```bash
python prepare_deploy.py
```

This copies:

* `models/*.pkl` → `api/models/`
* `models/*.pkl` → `dashboards/models/`

---

##  FastAPI API (Real-time Inference)

###  Run API locally

```bash
cd api/
uvicorn main:app --host 0.0.0.0 --port 8000
```

###  Health Check

```http
GET http://localhost:8000/
```

###  Predict Sales

```http
POST http://localhost:8000/predict?model_type=xgboost
```

**Sample Payload:**

```json
{
  "Store": 1,
  "DayOfWeek": 4,
  "Promo": 1,
  "SchoolHoliday": 0,
  "StoreType": "a",
  "Assortment": "a",
  "CompetitionDistance": 100.0,
  "Promo2": 0,
  "Date": "2025-07-01"
}
```

---

##  Streamlit Dashboard

###  Run the App

```bash
cd dashboards/
streamlit run app.py
```

### 💡 Features

*  Single record prediction form
*  CSV upload for batch prediction
*  Forecast visualizations with Plotly
* ⬇ Download predicted CSV results

---

##  Batch Prediction via CLI

```bash
python predict.py --input data/raw/test.csv --model xgboost
```

Outputs saved as `predictions_xgboost.csv`.

---

## Render Deployment
changes in main.py:  
MODELS_DIR = os.path.join(BASE_DIR, "models")      # for render change it to like this
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

changes in app.py:
API_BASE = os.getenv("API_URL", "http://localhost:8000")  to keep your fastapi render url like this 
API_BASE = os.getenv("API_URL", "https://api-js79.onrender.com")

##  Docker Deployment (Local or EC2)

> Requires `Docker` and `docker-compose`

```bash
docker-compose up --build -d
```

| Service   | URL                                            |
| --------- | ---------------------------------------------- |
| FastAPI   | [http://localhost:8000](http://localhost:8000) |
| Streamlit | [http://localhost:8501](http://localhost:8501) |

---

##  EC2 Deployment (Production-Ready)

```bash
sudo apt update && sudo apt install docker.io docker-compose -y
git clone https://github.com/<your-username>/product-demand-forecasting-mlops
cd product-demand-forecasting-mlops
python prepare_deploy.py
docker-compose up --build -d
```

>  Keep `.gitignore` configured to **allow model files** in `api/models/` and `dashboards/models/` for deployment.

---

##  License

Param Patil @IIT Roorkee Data Science

---

##  Author

**Patil Param**
*Machine Learning & MLOps Engineer*
📧 [paramhiralalpatil@gmail.com],
🔗 [GitHub](https://github.com/Param-Patil),
🔗 [LinkedIn](linkedin.com/in/param-hp)

---

> This project is a blueprint for building and deploying production-grade MLOps systems with real-time inference and monitoring.



```

