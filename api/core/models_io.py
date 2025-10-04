"""Pydantic request/response schemas used by the API.

These align with the UI so payloads remain
compatible across both direct API calls and the web client.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """
    What users need to provide for a biofilm removal prediction.

    All enzyme ratios will be normalized to sum to 1.0 internally.
    """
    # The three enzymes being tested
    DspB_ratio: float = Field(..., ge=0, description="Dispersin B ratio")
    DNase_I_ratio: float = Field(..., ge=0, description="DNase I ratio")
    ProK_ratio: float = Field(..., ge=0, description="Proteinase K ratio")

    # Experimental conditions
    Total_Volume: float = Field(100.0, ge=10, description="Total reaction volume (µL)")
    pH: float = Field(7.2, ge=6.8, le=8.5, description="Reaction pH")
    Temperature: float = Field(43.0, ge=30, le=50, description="Temperature (°C)")
    Reaction_Time: float = Field(24.0, ge=1, description="Reaction time (hours)")
    biofilm_age_hours: float = Field(24.0, ge=12, le=120, description="Biofilm age (hours)")


class PredictionResponse(BaseModel):
    """
    The prediction results we send back to users.
    """
    mean_prediction: float          # Predicted removal percentage
    prediction_interval_low: float  # Lower confidence bound
    prediction_interval_high: float # Upper confidence bound
    epistemic_uncertainty: float    # Model uncertainty estimate


class PriorExperiment(BaseModel):
    """Schema for previous experiment data used in optimization."""
    inputs: Dict[str, float]  # Experiment inputs (enzyme ratios, conditions)
    output: float            # Measured biofilm removal percentage


class OptimizationRequest(BaseModel):
    """
    Input schema for optimal mixture optimization requests.

    Uses Bayesian optimization to suggest enzyme ratios that maximize
    predicted removal while accounting for uncertainty.
    """
    fixed_conditions: Dict[str, Any]
    prior_experiments: Optional[List[PriorExperiment]] = None
    n_trials: int = Field(100, ge=10, le=1000)

