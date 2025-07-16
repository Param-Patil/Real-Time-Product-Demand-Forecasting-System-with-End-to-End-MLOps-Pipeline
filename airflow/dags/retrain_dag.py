#airflow/dags/*.py
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="retrain_model_pipeline",
    default_args=default_args,
    description="Retrain XGBoost/LightGBM models monthly",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@monthly",
    catchup=False
) as dag:

    retrain_script = BashOperator(
        task_id="retrain_model",
        bash_command="python /app/src/train.py --model all"
    )
