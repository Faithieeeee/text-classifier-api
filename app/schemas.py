from pydantic import BaseModel, Field
from typing import List

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)

class PredictResponse(BaseModel):
    label: str
    confidence: float

class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=100)

class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    