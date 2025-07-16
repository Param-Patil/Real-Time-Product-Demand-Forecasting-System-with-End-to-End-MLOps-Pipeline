from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_predict():
    payload = {
        "Store": 1,
        "DayOfWeek": 4,
        "Promo": 1,
        "SchoolHoliday": 0,
        "StoreType": "a",
        "Assortment": "a",
        "CompetitionDistance": 500.0,
        "Promo2": 1,
        "Date": "2015-06-15"
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predicted_sales" in response.json()
