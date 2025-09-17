from typing import List, Literal, Annotated
from pydantic import BaseModel, Field, validator
from datetime import date
from langgraph.graph import add_messages


class ScoresModel(BaseModel):
    rfm_score: int = Field(..., ge=0, le=100, description="RFM score between 0-100")
    churn_risk: float = Field(..., ge=0.0, le=1.0, description="Churn risk between 0-1")
    priority: int = Field(..., ge=1, le=5, description="Priority level between 1-5")


class RecommendationModel(BaseModel):
    action: Literal["call", "email", "offer_bundle", "promo"] = Field(
        ..., description="Recommended action type"
    )
    reason: str = Field(..., min_length=1, description="Reason for the recommendation")


class AnalysisResponse(BaseModel):
    customer_id: str = Field(..., description="Customer identifier")
    scores: ScoresModel = Field(..., description="Customer scoring metrics")
    summary: str = Field(..., min_length=1, description="Purchase behavior summary")
    recommendations: List[RecommendationModel] = Field(
        ..., min_items=1, max_items=3, description="Top 3 recommended actions"
    )
    top_followups_today: List[str] = Field(
        ..., description="List of customer IDs to follow up today"
    )


class AnalyzeRequest(BaseModel):
    customer_id: str = Field(..., description="Customer ID to analyze")


class TopFollowupsRequest(BaseModel):
    date: str = Field(..., description="Date in YYYY-MM-DD format")

    @validator('date')
    def validate_date_format(cls, v):
        try:
            date.fromisoformat(v)
            return v
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD format')


# Internal models for graph state
class GraphState(BaseModel):
    customer_id: str
    customer_data: dict = Field(default_factory=dict)
    rfm_analysis: dict = Field(default_factory=dict)
    churn_analysis: dict = Field(default_factory=dict)
    summary: str = ""
    recommendations: List[dict] = Field(default_factory=list)
    top_followups: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    retry_count: int = 0
    merge_complete: bool = False
    final_response: dict = Field(default_factory=dict)