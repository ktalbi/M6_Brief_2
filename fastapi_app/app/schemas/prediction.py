from pydantic import BaseModel, Field

class PredictResponse(BaseModel):
    prediction_id: int
    predicted_label: int
    probabilities: list[float]

class CorrectionRequest(BaseModel):
    prediction_id: int = Field(..., ge=1)
    true_label: int = Field(..., ge=0, le=9)

class CorrectionResponse(BaseModel):
    status: str
