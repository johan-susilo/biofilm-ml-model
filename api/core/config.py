"""Service-wide configuration.

Defines model file locations and the canonical order of basic features
expected by the ML models and frontend.
"""

from pathlib import Path

# Keep these identical to your original constants
XGB_PATHS = [Path("ml-model/xgb_biofilm_model.json")]
RF_PATHS  = [Path("ml-model/rf_uncertainty_model.joblib")]

# The features the model expects (order matters)
FEATURES = ["dspb", "dnase", "prok", "reaction_time"]
