"""Miscellaneous endpoints for UI support.

Currently exposes `/training-schema-stats` which the frontend uses to
validate input ranges before sending prediction/optimization requests.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/training-schema-stats")
def get_training_schema_stats():
    """Return min/max ranges for UI-level validation matching the schema."""
    return {
        "DspB_ratio": {"min": 0.0, "max": 1.0},
        "DNase_I_ratio": {"min": 0.0, "max": 1.0},
        "ProK_ratio": {"min": 0.0, "max": 1.0},
        "Total_Volume": {"min": 10.0, "max": 10000.0},
        "pH": {"min": 6.8, "max": 8.2},
        "Temperature": {"min": 30.0, "max": 50.0},
        "Reaction_Time": {"min": 1.0, "max": 72.0},
        "biofilm_age_hours": {"min": 12.0, "max": 120.0},
    }