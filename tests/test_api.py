import pytest
from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def build_valid_payload(**overrides):
    payload = {
        "Car_ID": 1,
        "Brand": "Toyota",
        "Model": "Corolla",
        "Year": 2018,
        "Kilometers_Driven": 50000,
        "Fuel_Type": "Petrol",
        "Transmission": "Manual",
        "Owner_Type": "First",
        "Mileage": 15,
        "Engine": 1498,
        "Power": 108,
        "Seats": 5,
    }
    payload.update(overrides)
    return {"features": payload}


def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_valid_input_returns_200_and_expected_schema():
    response = client.post("/predict", json=build_valid_payload())

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"prediction"}
    assert isinstance(body["prediction"], (int, float))


def test_predict_missing_field_returns_422():
    payload = build_valid_payload()
    payload["features"].pop("Transmission")

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_unknown_enum_returns_422():
    response = client.post("/predict", json=build_valid_payload(Fuel_Type="Electric"))

    assert response.status_code == 422


def test_newer_vehicle_is_predicted_more_expensive_than_older_vehicle():
    older = build_valid_payload(Year=2012)
    newer = build_valid_payload(Year=2020)

    older_response = client.post("/predict", json=older)
    newer_response = client.post("/predict", json=newer)

    assert older_response.status_code == 200
    assert newer_response.status_code == 200
    assert newer_response.json()["prediction"] > older_response.json()["prediction"]


def test_automatic_transmission_is_predicted_more_expensive_than_manual():
    manual = build_valid_payload(Transmission="Manual")
    automatic = build_valid_payload(Transmission="Automatic")

    manual_response = client.post("/predict", json=manual)
    automatic_response = client.post("/predict", json=automatic)

    assert manual_response.status_code == 200
    assert automatic_response.status_code == 200
    assert automatic_response.json()["prediction"] > manual_response.json()["prediction"]
