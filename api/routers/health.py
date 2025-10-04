"""Health and status endpoints.

Exposes:
- GET /health: lightweight health check
- GET /      : status page including model load flags
"""

from fastapi import APIRouter
from ..ml.engine import xgb_booster, rf_model

router = APIRouter()

@router.get("/health")
def health():
    """Container/ELB-friendly health probe endpoint."""
    return {"status": "healthy"}

@router.get("/")
def health_check():
    """Basic status with model availability flags for quick diagnostics."""
    return {
        "status": "OK" if xgb_booster is not None else "DEGRADED",
        "model": "XGB Biofilm v1.0.0",
        "xgb_loaded": bool(xgb_booster is not None),
        "rf_loaded": bool(rf_model is not None),
    }