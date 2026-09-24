import os
import logging
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.schemas import (
    PredictRequest, PredictResponse,
    BatchPredictRequest, BatchPredictResponse,
    HealthResponse,
)
from app.model import classifier_service

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("classifier-api")

API_KEY = os.getenv("API_KEY", "dev-key-change-me")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app = FastAPI(title="Text Classifier API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def load_model():
    logger.info("Loading model...")
    classifier_service.load()
    logger.info("Model loaded.")

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=classifier_service.is_loaded)

@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(verify_api_key)])
def predict(req: PredictRequest):
    if not classifier_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    logger.info("Prediction request received (len=%d chars)", len(req.text))
    label, confidence = classifier_service.predict(req.text)
    return PredictResponse(label=str(label), confidence=confidence)

@app.post("/batch-predict", response_model=BatchPredictResponse, dependencies=[Depends(verify_api_key)])
def batch_predict(req: BatchPredictRequest):
    if not classifier_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    logger.info("Batch prediction request received (n=%d)", len(req.texts))
    results = classifier_service.predict_batch(req.texts)
    return BatchPredictResponse(
        results=[PredictResponse(label=str(l), confidence=c) for l, c in results]
    )