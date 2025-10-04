"""Optimization and experiment suggestion endpoints.

Exposes:
- POST /optimal-mix: find best enzyme ratios/time using BO or fallback
- GET  /optimal-mix: quick default equal-ratio suggestion for UI init
- POST /suggest-experiments: uncertainty-driven active learning picks
"""


from fastapi import APIRouter, HTTPException
from typing import Any, Dict
from ..core.models_io import OptimizationRequest
from ..optimization.optimizer import find_optimal_mix, suggest_experiments, default_optimal_mix

router = APIRouter()

@router.post("/optimal-mix")
def optimal_mix(req: OptimizationRequest):
    try:
        return find_optimal_mix(
            fixed_conditions=req.fixed_conditions,
            n_trials=req.n_trials,
            prior_experiments=[
                {**pe.inputs, "output": pe.output} for pe in (req.prior_experiments or [])
            ] if req.prior_experiments else None,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Some clients may call GET /optimal-mix with query params; keep it for backward-compat
@router.get("/optimal-mix")
def optimal_mix_get() -> Dict[str, Any]:
    """Lightweight default optimal mixture for UI initialization."""
    return default_optimal_mix()

@router.post("/suggest-experiments")
def suggest_experiments_route(req: OptimizationRequest):
    try:
        return suggest_experiments(
            fixed_conditions=req.fixed_conditions,
            prior_experiments=[
                {**pe.inputs, "output": pe.output} for pe in (req.prior_experiments or [])
            ] if req.prior_experiments else None,
            n=5,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
