from pydantic import BaseModel, Field
from typing import List


class TransactionRequest(BaseModel):
    customer_id: str = Field(..., json_schema_extra={"example": "CUST_00042"})
    amount: float = Field(..., gt=0, json_schema_extra={"example": 4500.0})
    merchant_category: str = Field(..., json_schema_extra={"example": "electronics"})
    timestamp: str = Field(..., json_schema_extra={"example": "2026-07-28T03:15:00"})
    lat: float = Field(..., json_schema_extra={"example": 19.07})
    lon: float = Field(..., json_schema_extra={"example": 72.87})


class RiskResponse(BaseModel):
    risk_score: float
    flagged: bool
    threshold_used: float
    top_reasons: List[str]
