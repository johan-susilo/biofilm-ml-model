"""Model engine: loading, feature preparation, predictions, and importance.

This module ensure consistent predictions and uncertainty estimates while 
keeping the code organized inside the `api/` folder.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import joblib

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    xgb = None
    XGB_AVAILABLE = False

from ..core.config import XGB_PATHS, RF_PATHS, FEATURES

# Global model holders
xgb_booster: Optional[Any] = None
rf_model: Optional[Any] = None

def _try_load_xgb() -> Optional[Any]:
    """
    Load the XGBoost model from available locations.

    Returns:
        XGBoost model if found and loaded successfully, None otherwise
    """
    if not XGB_AVAILABLE:
        return None

    for path in XGB_PATHS:
        try:
            if Path(path).exists():
                booster = xgb.Booster()
                booster.load_model(str(path))
                return booster
        except Exception:
            continue
    return None


def _try_load_rf() -> Optional[Any]:
    """
    Load the Random Forest model for uncertainty estimation.

    Returns:
        Loaded Random Forest model if successful, None otherwise
    """
    for path in RF_PATHS:
        try:
            if Path(path).exists():
                return joblib.load(str(path))
        except Exception:
            continue
    return None


def _normalize(d: float, n: float, p: float) -> Tuple[float, float, float]:
    """
    Convert enzyme ratios to proportions that sum to 1.0.

    Args:
        d: Dispersin B ratio
        n: DNase I ratio
        p: Proteinase K ratio

    Returns:
        Normalized ratios (d, n, p)
    """
    total = float(d + n + p)
    if total <= 0:
        return (1.0 / 3, 1.0 / 3, 1.0 / 3)
    return d / total, n / total, p / total


def _build_features(d: float, n: float, p: float, rt: float) -> np.ndarray:
    """
    Create the feature array that our model expects.

    Args:
        d: Dispersin B ratio
        n: DNase I ratio
        p: Proteinase K ratio
        rt: Reaction time in hours

    Returns:
        Feature array ready for prediction
    """
    # Normalize enzyme ratios to sum to 1.0
    d, n, p = _normalize(d, n, p)

    # Create feature array: [dspb, dnase, prok, reaction_time]
    features = np.array([d, n, p, rt], dtype=float)
    return features.reshape(1, -1)


def _predict_with_uncertainty(d: float, n: float, p: float, rt: float) -> Tuple[float, float]:
    """
    Make prediction using XGBoost and estimate uncertainty using Random Forest.

    Args:
        d: Dispersin B ratio
        n: DNase I ratio
        p: Proteinase K ratio
        rt: Reaction time in hours

    Returns:
        Tuple of (prediction_percent, uncertainty_percent)

    Process:
    1. Build basic 4-feature array
    2. Make prediction using XGBoost model
    3. Calculate uncertainty from Random Forest ensemble variance
    4. Convert outputs to percentage scale
    """
    if xgb_booster is None:
        return 0.0, 0.0

    prediction = 0.0
    uncertainty = 0.0

    # Prepare feature array using basic features
    X_basic = _build_features(d, n, p, rt)

    # Make prediction using XGBoost
    if XGB_AVAILABLE:
        try:
            # First try with feature names for better compatibility
            dmat = xgb.DMatrix(X_basic, feature_names=FEATURES)
            prediction = float(xgb_booster.predict(dmat)[0])
        except Exception:
            try:
                # Fallback: try without feature names
                dmat = xgb.DMatrix(X_basic)
                prediction = float(xgb_booster.predict(dmat)[0])
            except Exception:
                # If both fail, return 0 (model incompatible)
                prediction = 0.0

    # Enhanced uncertainty calculation using Random Forest ensemble variance
    try:
        if rf_model is not None and hasattr(rf_model, "estimators_"):
            # Use basic features for uncertainty estimation
            tree_predictions = [float(estimator.predict(X_basic)[0]) for estimator in rf_model.estimators_]
            if tree_predictions:
                uncertainty = float(np.std(tree_predictions))

                # If uncertainty is too low, add distance-based uncertainty
                if uncertainty < 0.01:
                    # Calculate distance from "center" ratios (0.33, 0.33, 0.33, 24h)
                    center_distance = np.sqrt((d - 0.33)**2 + (n - 0.33)**2 + (p - 0.33)**2 + ((rt - 24.0)/48.0)**2)
                    uncertainty = max(uncertainty, center_distance * 0.1)  # Scale distance to reasonable uncertainty
    except Exception:
        # Fallback uncertainty based on distance from explored regions
        center_distance = np.sqrt((d - 0.33)**2 + (n - 0.33)**2 + (p - 0.33)**2 + ((rt - 24.0)/48.0)**2)
        uncertainty = center_distance * 0.05  # Minimal uncertainty for unexplored regions

    # Convert to percentage scale
    prediction_pct = max(0.0, min(100.0, prediction * 100.0))
    uncertainty_pct = max(0.0, uncertainty * 100.0)

    return prediction_pct, uncertainty_pct


def _get_feature_importance() -> Dict[str, float]:
    """
    Extract feature importance scores from the XGBoost model.

    Returns:
        Dictionary mapping basic feature names to importance scores (F-scores)
    """
    if xgb_booster is None:
        # Return uniform importance for features if no model is loaded
        return {k: 1.0 / len(FEATURES) for k in FEATURES}

    try:
        # Get F-score importance from XGBoost
        importance_dict = xgb_booster.get_fscore()
        if isinstance(importance_dict, dict) and importance_dict:
            # Filter to only basic features and return those
            basic_importance = {}
            for feature in FEATURES:
                if feature in importance_dict:
                    basic_importance[feature] = float(importance_dict[feature])
                else:
                    basic_importance[feature] = 1.0 / len(FEATURES)
            if basic_importance:
                return basic_importance
    except Exception:
        pass

    # Fallback to uniform importance for basic features
    return {k: 1.0 / len(FEATURES) for k in FEATURES}

# Model init hook (can be called at startup)
def load_models() -> None:
    global xgb_booster, rf_model
    xgb_booster = _try_load_xgb()
    rf_model = _try_load_rf()
