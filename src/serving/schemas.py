"""Pydantic input/output schemas for the inference API."""
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    age: float = Field(..., ge=18, le=100, description="Customer age in years")
    tenure_months: float = Field(..., ge=0, le=240, description="Months as a customer")
    monthly_charge: float = Field(..., ge=0, le=500, description="Monthly bill in USD")
    num_products: float = Field(..., ge=1, le=20, description="Number of subscribed products")
    contract_type: str = Field(
        ...,
        description="Contract type: month-to-month | one_year | two_year",
    )
    payment_method: str = Field(
        ...,
        description="Payment method: electronic_check | mailed_check | bank_transfer | credit_card",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 45,
                "tenure_months": 12,
                "monthly_charge": 89.5,
                "num_products": 2,
                "contract_type": "month-to-month",
                "payment_method": "electronic_check",
            }
        }
    }


class PredictResponse(BaseModel):
    label: str = Field(..., description="Predicted class: churn | no_churn")
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    latency_ms: float = Field(..., description="Inference latency in milliseconds")


class HealthResponse(BaseModel):
    status: str
    model_run_id: str | None
    model_stage: str | None
