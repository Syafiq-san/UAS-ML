import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


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


@app.on_event("startup")
def startup_event():
    app.state.pipeline = load_pipeline()


@app.get("/")
def root():
    return {"service": "ml-prediction", "endpoints": ["/health", "/predict"]}


@app.get("/health")
def health_check():
    model_loaded = hasattr(app.state.pipeline, "predict")
    return {"status": "ok" if model_loaded else "error", "model_loaded": model_loaded}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    pipeline = app.state.pipeline
    if not hasattr(pipeline, "predict"):
        raise HTTPException(status_code=500, detail="Model pipeline tidak valid")

    payload = request.features or request.input or {}
    if not payload:
        raise HTTPException(status_code=422, detail="Harap kirim field features atau input")

    try:
        input_frame = pd.DataFrame([payload])
        expected_columns = getattr(pipeline, "feature_names_in_", None)
        if expected_columns is not None:
            input_frame = input_frame.reindex(columns=expected_columns)
        prediction = pipeline.predict(input_frame)[0]
        return {"prediction": float(prediction) if hasattr(prediction, "item") else prediction}
    except Exception as exc:
        print(f"Prediction error: {exc}")
        raise HTTPException(status_code=400, detail=str(exc))
