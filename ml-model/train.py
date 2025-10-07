"""Train models for the Biofilm Prediction API using data/polished.csv.

Best practices for reliability and consistency:
- Deterministic seeds across libraries
- Enzyme feature normalization consistent with inference
- Target scaling to [0, 1] if dataset is 0..100
- Nested CV hyperparameter tuning with Optuna (inner xgb.cv)
- Final model retraining with optimal rounds
- Separate RandomForest model for uncertainty estimation

Outputs under ml-model/ (repo root):
- xgb_biofilm_model.json (XGBoost Booster)
- rf_uncertainty_model.joblib (RandomForestRegressor)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold


# ---------------------------
# Configuration
# ---------------------------
SEED = 42
N_TRIALS = 100  # Adjust for speed vs. thoroughness
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 5
FEATURES = ["dspb", "dnase", "prok", "reaction_time"]
ENZYME_FEATURES = ["dspb", "dnase", "prok"]

np.random.seed(SEED)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_dataset() -> pd.DataFrame:
    # Prefer repo-root data/polished.csv regardless of CWD
    candidates = [
        _project_root() / "data" / "polished.csv",
        Path("../data/polished.csv").resolve(),
        Path("data/polished.csv").resolve(),
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            print(f"Loaded dataset: {p} -> {df.shape[0]} rows, {df.shape[1]} cols")
            return df

    raise FileNotFoundError("polished.csv not found in expected locations.")


def preprocess(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    # Basic column validation
    missing = [c for c in FEATURES + ["degrade"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    # Drop obvious bad rows and coerce numeric
    for c in FEATURES + ["degrade"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=FEATURES + ["degrade"]).reset_index(drop=True)

    # Normalize enzyme ratios to sum==1 (avoid divide by zero)
    totals = df[ENZYME_FEATURES].sum(axis=1).replace(0, np.nan)
    df.loc[:, ENZYME_FEATURES] = df[ENZYME_FEATURES].div(totals, axis=0)
    df.loc[:, ENZYME_FEATURES] = df[ENZYME_FEATURES].fillna(1.0 / 3)

    # Harmonize target scale for consistency with API
    y = df["degrade"].astype(float).values
    if np.nanmax(y) > 1.5:  # if percentages 0..100
        y = y / 100.0
    # Do not clip negatives: the API clamps at display time only

    X = df[FEATURES].astype(float).values
    return X, y


def tune_with_cv(dtrain: xgb.DMatrix) -> Dict:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "random_state": SEED,
            "seed": SEED,
            # conservative, generalizable space
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "nthread": -1,
        }
        num_boost_round = trial.suggest_int("n_estimators", 200, 1200)
        cv = xgb.cv(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            nfold=N_INNER_SPLITS,
            seed=SEED,
            as_pandas=True,
            early_stopping_rounds=50,
            shuffle=True,
            verbose_eval=False,
        )
        trial.set_user_attr("best_rounds", int(cv.shape[0]))
        return float(cv["test-rmse-mean"].iloc[-1])

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False, n_jobs=-1)
    best = dict(study.best_params)
    best_rounds = int(study.best_trial.user_attrs["best_rounds"])  # type: ignore[index]
    best.pop("n_estimators", None)
    return {"params": best, "rounds": best_rounds, "study": study}


def nested_cv_eval(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    kf = KFold(n_splits=N_OUTER_SPLITS, shuffle=True, random_state=SEED)
    r2s, rmses = [], []
    print(f"\nStarting Nested Cross-Validation ({N_OUTER_SPLITS} outer folds)...")
    for i, (tr, te) in enumerate(kf.split(X, y), start=1):
        print(f"  Fold {i}/{N_OUTER_SPLITS}")
        dtrain = xgb.DMatrix(X[tr], label=y[tr], feature_names=FEATURES)
        dtest = xgb.DMatrix(X[te], label=y[te], feature_names=FEATURES)

        tuned = tune_with_cv(dtrain)
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "random_state": SEED,
            "seed": SEED,
            "nthread": -1,
            **tuned["params"],
        }
        model = xgb.train(params=params, dtrain=dtrain, num_boost_round=tuned["rounds"])
        preds = model.predict(dtest)
        r2s.append(r2_score(y[te], preds))
        rmses.append(mean_squared_error(y[te], preds, squared=False))
        print(f"    R²={r2s[-1]:.3f} RMSE={rmses[-1]:.3f} rounds={tuned['rounds']}")

    return {
        "r2_mean": float(np.mean(r2s)),
        "r2_std": float(np.std(r2s)),
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
    }


def train_final_models(X: np.ndarray, y: np.ndarray) -> Tuple[xgb.Booster, RandomForestRegressor, Dict]:
    d_full = xgb.DMatrix(X, label=y, feature_names=FEATURES)
    tuned = tune_with_cv(d_full)
    print(f"\nBest params (final): {tuned['params']}")
    print(f"Best rounds (final): {tuned['rounds']}")

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "random_state": SEED,
        "seed": SEED,
        "nthread": -1,
        **tuned["params"],
    }
    xgb_model = xgb.train(params=params, dtrain=d_full, num_boost_round=tuned["rounds"])

    # More trees provide more stable std across estimators
    rf_model = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
    rf_model.fit(X, y)
    return xgb_model, rf_model, tuned


def save_models(xgb_model: xgb.Booster, rf_model: RandomForestRegressor) -> None:
    out_dir = _project_root() / "ml-model"
    out_dir.mkdir(parents=True, exist_ok=True)
    xgb_path = out_dir / "xgb_biofilm_model.json"
    rf_path = out_dir / "rf_uncertainty_model.joblib"
    xgb_model.save_model(str(xgb_path))
    joblib.dump(rf_model, str(rf_path))
    print(f"Saved XGB to {xgb_path}")
    print(f"Saved RF  to {rf_path}")


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    print("Biofilm Model Training")
    print("=" * 60)

    df = load_dataset()
    X, y = preprocess(df)
    print(f"Using features: {FEATURES}")
    print(f"Target stats (scaled): min={y.min():.3f} max={y.max():.3f} mean={y.mean():.3f}")

    metrics = nested_cv_eval(X, y)
    print("\nNested CV Summary (target 0..1):")
    print(f"  R²:   {metrics['r2_mean']:.3f} ± {metrics['r2_std']:.3f}")
    print(f"  RMSE: {metrics['rmse_mean']:.3f} ± {metrics['rmse_std']:.3f}")

    xgb_model, rf_model, tuned = train_final_models(X, y)
    save_models(xgb_model, rf_model)

    # Simple feature importance printout (F-score)
    try:
        importance = xgb_model.get_fscore()
        if importance:
            ordered = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
            print("\nFeature importance (F-score):")
            for k, v in ordered:
                print(f"  {k}: {v}")
    except Exception:
        pass

    print("\nDone.")


if __name__ == "__main__":
    main()
