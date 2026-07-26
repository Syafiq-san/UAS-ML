import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .data_loader import load_config, load_data


class PredictionRequest(BaseModel):
    features: dict | None = None
    input: dict | None = None


class PredictionResponse(BaseModel):
    prediction: float | str


app = FastAPI(title="ML Prediction API")


def load_pipeline(path: str = "artifacts/pipeline.joblib"):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        raise RuntimeError(f"Pipeline tidak ditemukan di {path}. Jalankan src/train.py dulu.")


def load_allowed_values():
    df = load_data(load_config())
    allowed = {}
    for column in ["Brand", "Model", "Fuel_Type", "Transmission", "Owner_Type"]:
        if column in df.columns:
            allowed[column] = sorted({str(value) for value in df[column].dropna().tolist()})
    return allowed


@app.on_event("startup")
def startup_event():
    app.state.pipeline = load_pipeline()
    app.state.allowed_values = load_allowed_values()


@app.get("/")
def root():
    return {"service": "ml-prediction", "endpoints": ["/health", "/predict"]}


def ensure_app_state():
    if getattr(app.state, "pipeline", None) is None:
        app.state.pipeline = load_pipeline()
    if not hasattr(app.state, "allowed_values") or not getattr(app.state, "allowed_values", None):
        app.state.allowed_values = load_allowed_values()


@app.get("/health")
def health_check():
    ensure_app_state()
    pipeline = getattr(app.state, "pipeline", None)
    model_loaded = pipeline is not None and hasattr(pipeline, "predict")
    return {"status": "ok" if model_loaded else "error", "model_loaded": model_loaded}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    ensure_app_state()
    pipeline = getattr(app.state, "pipeline", None)

    if not hasattr(pipeline, "predict"):
        raise HTTPException(status_code=500, detail="Model pipeline tidak valid")

    payload = request.features or request.input or {}
    if not payload:
        raise HTTPException(status_code=422, detail="Harap kirim field features atau input")

    try:
        expected_columns = getattr(pipeline, "feature_names_in_", None)
        if expected_columns is None:
            raise HTTPException(status_code=500, detail="Model tidak memiliki daftar fitur yang terdefinisi")

        missing_columns = [column for column in expected_columns if column not in payload]
        if missing_columns:
            raise HTTPException(status_code=422, detail=f"Kolom fitur yang hilang: {missing_columns}")

        allowed_values = getattr(app.state, "allowed_values", {})
        for column, allowed in allowed_values.items():
            value = payload.get(column)
            if value is None:
                continue
            if str(value) not in allowed:
                raise HTTPException(status_code=422, detail=f"Nilai tidak valid untuk {column}: {value}")

        input_frame = pd.DataFrame([payload])
        input_frame = input_frame.reindex(columns=expected_columns)
        prediction = pipeline.predict(input_frame)[0]
        return {"prediction": float(prediction) if hasattr(prediction, "item") else prediction}
    except HTTPException:
        raise
    except Exception as exc:
        print(f"Prediction error: {exc}")
        raise HTTPException(status_code=400, detail=str(exc))
