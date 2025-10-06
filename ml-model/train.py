import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# --- Global Seed for Reproducibility ---
SEED = 42
TRIAL = 100 # Reduced for faster demonstration, you can set it back to 500
np.random.seed(SEED)

# --- Nested CV Configuration ---
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 5 # Used by xgb.cv internally

print(" XGBOOST OPTIMIZATION (with Final Production Fixes)")
print("=" * 60)

# --- Load and prepare data ---
try:
    df = pd.read_csv("polished.csv")
except FileNotFoundError:
    print("Error: 'polished.csv' not found. Please ensure the dataset is available.")
    data = {'dspb': np.random.rand(100)*10, 'dnase': np.random.rand(100)*10, 'prok': np.random.rand(100)*10, 'reaction_time': np.random.uniform(1, 48, 100), 'degrade': np.random.rand(100) * 100}
    df = pd.DataFrame(data)
    print(" Using a dummy dataset for demonstration.")

print(f" Dataset Loaded: {df.shape[0]} samples, {df.shape[1]} features")

# --- Feature Engineering & Normalization ---
features = ['dspb', 'dnase', 'prok', 'reaction_time']
enzyme_features = ['dspb', 'dnase', 'prok']

print("   normalizing enzyme features to ensure train/inference consistency...")
enzyme_totals = df[enzyme_features].sum(axis=1)
# Avoid division by zero for rows where all enzymes are 0
df[enzyme_features] = df[enzyme_features].div(enzyme_totals.replace(0, 1), axis=0)

X = df[features].values
y = df['degrade'].values

print(f"Using {len(features)} features: {features}")

# --- Nested Cross-Validation for Robust Evaluation ---
outer_cv = KFold(n_splits=N_OUTER_SPLITS, shuffle=True, random_state=SEED)
outer_fold_scores = []

print(f"\n Starting Nested Cross-Validation ({N_OUTER_SPLITS} outer folds)...")

for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
    print(f"\n===== Outer Fold {fold_idx + 1}/{N_OUTER_SPLITS} =====")
    X_train_outer, X_test_outer = X[train_idx], X[test_idx]
    y_train_outer, y_test_outer = y[train_idx], y[test_idx]
    
    dtrain_fold = xgb.DMatrix(X_train_outer, label=y_train_outer, feature_names=features)

    def xgb_objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'random_state': SEED,
            'max_depth': trial.suggest_int('max_depth', 3, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
        }
        n_estimators = trial.suggest_int('n_estimators', 100, 700)
        cv_results = xgb.cv(
            params=params, 
            dtrain=dtrain_fold, 
            num_boost_round=n_estimators, 
            nfold=N_INNER_SPLITS, 
            seed=SEED, 
            as_pandas=True,
            early_stopping_rounds=30,
            shuffle=True
        )
        best_rounds = int(cv_results.shape[0])
        trial.set_user_attr("best_rounds", best_rounds)
        return cv_results['test-rmse-mean'].iloc[-1]

    print("  Tuning hyperparameters (inner CV)...")
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(xgb_objective, n_trials=TRIAL, show_progress_bar=False, n_jobs=-1)

    best_params_fold = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'random_state': SEED,
    }
    best_params_fold.update(study.best_params) # Add tuned params
    best_params_fold.pop('n_estimators', None)
    
    best_rounds_fold = int(study.best_trial.user_attrs["best_rounds"])
    
    final_model_fold = xgb.train(
        params=best_params_fold,
        dtrain=dtrain_fold,
        num_boost_round=best_rounds_fold
    )
    
    dtest_fold = xgb.DMatrix(X_test_outer, feature_names=features)
    preds = final_model_fold.predict(dtest_fold)
    
    r2 = r2_score(y_test_outer, preds)
    rmse = np.sqrt(mean_squared_error(y_test_outer, preds))
    outer_fold_scores.append({'r2': r2, 'rmse': rmse})
    
    print(f"   Fold {fold_idx + 1} Results -> R²: {r2:.3f}, RMSE: {rmse:.3f} (trained for {best_rounds_fold} rounds)")

print("\n" + "="*60)
print(" Nested Cross-Validation Final Performance")
print("="*60)
avg_r2 = np.mean([score['r2'] for score in outer_fold_scores])
std_r2 = np.std([score['r2'] for score in outer_fold_scores])
avg_rmse = np.mean([score['rmse'] for score in outer_fold_scores])
std_rmse = np.std([score['rmse'] for score in outer_fold_scores])

print(f"Average R²:   {avg_r2:.3f} ± {std_r2:.3f}")
print(f"Average RMSE: {avg_rmse:.3f} ± {std_rmse:.3f}")

print("\n Training Final Models on the ENTIRE Dataset for Deployment...")
d_full = xgb.DMatrix(X, label=y, feature_names=features)

def final_objective(trial):
    params = {
        'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'random_state': SEED,
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
    }
    n_estimators = trial.suggest_int('n_estimators', 100, 700)
    cv_results = xgb.cv(
        params=params, dtrain=d_full, num_boost_round=n_estimators, 
        nfold=N_INNER_SPLITS, seed=SEED, as_pandas=True,
        early_stopping_rounds=30,
        shuffle=True 
    )
    best_rounds = int(cv_results.shape[0])
    trial.set_user_attr("best_rounds", best_rounds)
    return cv_results['test-rmse-mean'].iloc[-1]

print("  Re-tuning hyperparameters on all data...")
final_sampler = optuna.samplers.TPESampler(seed=SEED)
final_study = optuna.create_study(direction='minimize', sampler=final_sampler)
final_study.optimize(final_objective, n_trials=TRIAL, show_progress_bar=False, n_jobs=-1)

best_params_final = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'random_state': SEED,
}
best_params_final.update(final_study.best_params)
best_params_final.pop('n_estimators', None)
best_rounds_final = int(final_study.best_trial.user_attrs["best_rounds"])

print(f"\nBest params for final model: {final_study.best_params}")
print(f"Optimal rounds for final model: {best_rounds_final}")

xgb_model = xgb.train(
    params=best_params_final,
    dtrain=d_full,
    num_boost_round=best_rounds_final
)
print("  Final XGBoost model trained.")

rf_model = RandomForestRegressor(n_estimators=100, random_state=SEED)
rf_model.fit(X, y)
print("   Final Random Forest model trained.")

def predict_with_uncertainty(dspb, dnase, prok, reaction_time):
    # This normalization is now consistent with the training data
    total = max(dspb + dnase + prok, 1e-12)
    dspb_n, dnase_n, prok_n = dspb / total, dnase / total, prok / total
    row = np.array([[dspb_n, dnase_n, prok_n, reaction_time]])
    drow = xgb.DMatrix(row, feature_names=features)
    xgb_pred = float(xgb_model.predict(drow)[0])
    tree_preds = [tree.predict(row)[0] for tree in rf_model.estimators_]
    uncertainty = float(np.std(tree_preds))
    return xgb_pred, uncertainty

def find_best_mixture(reaction_time=24.0):
    def objective(trial):
        dspb = trial.suggest_float('dspb', 0.0, 1.0)
        dnase = trial.suggest_float('dnase', 0.0, 1.0 - dspb)
        prok = 1.0 - dspb - dnase
        pred, _ = predict_with_uncertainty(dspb, dnase, prok, reaction_time)
        return pred
    
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study_mix = optuna.create_study(direction='maximize', sampler=sampler)
    study_mix.optimize(objective, n_trials=TRIAL, show_progress_bar=False)

    best_dspb = study_mix.best_params['dspb']
    best_dnase = study_mix.best_params['dnase']
    best_prok = 1.0 - best_dspb - best_dnase
    pred, uncertainty = predict_with_uncertainty(best_dspb, best_dnase, best_prok, reaction_time)
    
    print(f"\n Best mixture for {reaction_time}h reaction:")
    print(f"  DSPB: {best_dspb:.3f}, DNase: {best_dnase:.3f}, Prok: {best_prok:.3f}")
    print(f"  Predicted degradation: {pred:.3f} ± {uncertainty:.3f}")
    return (best_dspb, best_dnase, best_prok), pred, uncertainty

best_mixture, best_pred, best_unc = find_best_mixture()

def suggest_next_experiments(reaction_time=24.0, n_suggestions=5):
    print(f"\n Active Learning Suggestions for {reaction_time}h reaction:")
    candidates = []
    random_points = np.random.dirichlet(np.ones(3), size=2000)
    for point in random_points:
        dspb, dnase, prok = point[0], point[1], point[2]
        pred, uncertainty = predict_with_uncertainty(dspb, dnase, prok, reaction_time)
        acquisition_score = pred + 1.5 * uncertainty
        candidates.append({'dspb': dspb, 'dnase': dnase, 'prok': prok, 'prediction': pred, 'uncertainty': uncertainty, 'score': acquisition_score})
    
    candidates.sort(key=lambda x: x['score'], reverse=True)
    for i, candidate in enumerate(candidates[:n_suggestions]):
        print(f"\nExperiment {i+1}: DSPB: {candidate['dspb']:.3f}, DNase: {candidate['dnase']:.3f}, Prok: {candidate['prok']:.3f}")
        print(f"  Predicted: {candidate['prediction']:.3f}, Uncertainty: {candidate['uncertainty']:.3f}, Score: {candidate['score']:.3f}")
    return candidates

next_experiments = suggest_next_experiments()

print(f"\n XGBoost Feature Importance:")
importance_scores = xgb_model.get_fscore()
sorted_importance = sorted(importance_scores.items(), key=lambda item: item[1], reverse=True)
for feature, importance in sorted_importance:
    print(f"  {feature}: {importance}")

print("\n Saving models...")
xgb_model_filename = "xgb_biofilm_model.json"
xgb_model.save_model(xgb_model_filename)
print(f"- XGBoost model saved to: {xgb_model_filename}")

rf_model_filename = "rf_uncertainty_model.joblib"
joblib.dump(rf_model, rf_model_filename)
print(f"- Random Forest model saved to: {rf_model_filename}")

print(f"\n SUMMARY:")
print(f"- Model performance robustly estimated with {N_OUTER_SPLITS}-fold nested CV.")
print(f"- Final models trained on all data for prediction.")
print(f"- Best 3-enzyme mixture identified for 24h reaction.")
print(f"- {len(next_experiments)} new experiments suggested by active learning.")