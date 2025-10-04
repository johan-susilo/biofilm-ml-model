"""Prediction and feature-importance endpoints.

Exposes:
- POST /predict: returns mean prediction with CI and epistemic uncertainty
- GET  /feature-importance: names/values for basic features used by the UI
"""

from fastapi import APIRouter, HTTPException
import numpy as np
from ..core.models_io import PredictionRequest, PredictionResponse
from ..ml.engine import _predict_with_uncertainty, _get_feature_importance
from ..core.config import FEATURES

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        # Normalize ratios and predict using engine helpers
        d = float(request.DspB_ratio)
        n = float(request.DNase_I_ratio)
        p = float(request.ProK_ratio)
        rt = float(request.Reaction_Time)

        prediction_pct, uncertainty_pct = _predict_with_uncertainty(d, n, p, rt)

        mean_pred = float(np.clip(prediction_pct, 0.0, 100.0))
        epistemic_uncertainty = float(uncertainty_pct)
        interval_half_width = 1.96 * epistemic_uncertainty

        return PredictionResponse(
            mean_prediction=mean_pred,
            prediction_interval_low=float(np.clip(mean_pred - interval_half_width, 0.0, 100.0)),
            prediction_interval_high=float(np.clip(mean_pred + interval_half_width, 0.0, 100.0)),
            epistemic_uncertainty=epistemic_uncertainty,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/feature-importance")
def feature_importance():
    # filter to basic features and rename for UI
    all_importances = _get_feature_importance()
    basic_importances = {k: float(v) for k, v in all_importances.items() if k in FEATURES}
    if not basic_importances:
        raise HTTPException(status_code=404, detail="Basic feature importances unavailable for current model")

    sorted_items = sorted(basic_importances.items(), key=lambda x: x[1], reverse=True)
    feature_name_map = {
        'dspb': 'Dispersin B',
        'dnase': 'DNase I',
        'prok': 'Proteinase K',
        'reaction_time': 'Reaction Time',
    }
    features = [feature_name_map.get(feat, feat) for feat, _ in sorted_items]
    values = [val for _, val in sorted_items]
    return {"features": features, "values": values}

