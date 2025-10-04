"""Optimization (optimal mix) and experiment suggestion logic.

Implements two key capabilities used by the frontend:
- Finding a high-performing enzyme mixture using Bayesian optimization
  (falling back to a randomized search if Optuna isn't available)
- Suggesting diverse, high-uncertainty experiments for active learning
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Re-use engine helpers for predictions
from ..ml.engine import _predict_with_uncertainty, _normalize


def find_optimal_mix(
    fixed_conditions: Dict[str, Any],
    n_trials: int = 100,
    prior_experiments: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Find optimal enzyme mixture and reaction time using Bayesian optimization
    (or random search fallback). 
    """

    def acquisition_function(ratios: Tuple[float, float, float], reaction_time: float) -> Tuple[float, float, float]:
        d, n, p = _normalize(*ratios)
        prediction, uncertainty = _predict_with_uncertainty(d, n, p, reaction_time)
        acquisition_score = float(prediction + 1.5 * uncertainty)
        return acquisition_score, float(prediction), float(uncertainty)

    # Collect prior experiments to avoid suggesting similar mixtures
    tested_combinations: List[Tuple[np.ndarray, float]] = []
    if prior_experiments:
        for experiment in prior_experiments:
            ratios = np.array([
                float(experiment.get('DspB_ratio', 0.0)),
                float(experiment.get('DNase_I_ratio', 0.0)),
                float(experiment.get('ProK_ratio', 0.0)),
            ], dtype=float)
            total = ratios.sum() + 1e-9
            reaction_time = float(experiment.get('Reaction_Time', 24.0))
            tested_combinations.append((ratios / total, reaction_time))

    def is_too_similar_to_tested(ratios: np.ndarray, reaction_time: float, threshold: float = 0.12) -> bool:
        for tested_ratios, tested_time in tested_combinations:
            if (np.sum(np.abs(ratios - tested_ratios)) < threshold and
                abs(reaction_time - tested_time) < 3.0):
                return True
        return False

    best_result = {
        "ratios": (1/3, 1/3, 1/3),
        "reaction_time": 24.0,
        "score": -1.0,
        "prediction": 0.0,
        "uncertainty": 0.0,
    }

    # Try Optuna; fallback to random search
    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            dspb = trial.suggest_float("dspb", 0.0, 1.0)
            dnase = trial.suggest_float("dnase", 0.0, 1.0 - dspb)
            prok = 1.0 - dspb - dnase
            reaction_time = trial.suggest_float("reaction_time", 1.0, 72.0)
            score, pred, uncert = acquisition_function((dspb, dnase, prok), reaction_time)
            return -score

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(
                seed=42,
                n_startup_trials=min(10, n_trials // 4),
                n_ei_candidates=24,
                warn_independent_sampling=False,
            ),
        )
        n_trials_run = min(int(n_trials), 100)
        study.optimize(objective, n_trials=n_trials_run, show_progress_bar=False)

        best_dspb = float(study.best_params["dspb"]) if study.best_params else 1/3
        best_dnase = float(study.best_params["dnase"]) if study.best_params else 1/3
        best_prok = float(1.0 - best_dspb - best_dnase)
        best_reaction_time = float(study.best_params.get("reaction_time", 24.0)) if study.best_params else 24.0

        score, prediction, uncertainty = acquisition_function(
            (best_dspb, best_dnase, best_prok), best_reaction_time
        )

        best_result.update({
            "ratios": (best_dspb, best_dnase, best_prok),
            "reaction_time": best_reaction_time,
            "score": score,
            "prediction": prediction,
            "uncertainty": uncertainty,
        })

    except Exception:
        rng = np.random.default_rng(42)
        for _ in range(int(n_trials)):
            ratios = rng.dirichlet(alpha=[1.5, 1.5, 1.5])
            ratios_array = np.array([float(ratios[0]), float(ratios[1]), float(ratios[2])], dtype=float)
            reaction_time = rng.uniform(1.0, 72.0)

            if is_too_similar_to_tested(ratios_array, reaction_time):
                continue

            score, prediction, uncertainty = acquisition_function(
                (ratios_array[0], ratios_array[1], ratios_array[2]), reaction_time
            )

            if score > best_result["score"]:
                best_result.update({
                    "ratios": (float(ratios_array[0]), float(ratios_array[1]), float(ratios_array[2])),
                    "reaction_time": reaction_time,
                    "score": score,
                    "prediction": prediction,
                    "uncertainty": uncertainty,
                })

    # Convert optimal ratios to integer counts (out of 100)
    d, n, p = best_result["ratios"]
    integer_counts = [int(round(d * 100)), int(round(n * 100)), int(round(p * 100))]
    total_count = sum(integer_counts)
    if total_count != 100:
        max_index = int(np.argmax(integer_counts))
        integer_counts[max_index] += (100 - total_count)

    return {
        "ratios": [float(d), float(n), float(p)],
        "integer_counts": integer_counts,
        "predicted": float(np.clip(best_result["prediction"], 0.0, 100.0)),
        "uncertainty": float(max(0.0, best_result["uncertainty"])),
        "recommended_reaction_time": float(best_result["reaction_time"]),
        "total_stock_concentration_mg_ml": 1.0,
    }


def default_optimal_mix() -> Dict[str, Any]:
    """Lightweight default optimal mixture for UI initialization."""
    d, n, p = (1/3, 1/3, 1/3)
    integer_counts = [33, 33, 34]
    rt = 24.0
    pred, uncert = _predict_with_uncertainty(d, n, p, rt)
    return {
        "ratios": [d, n, p],
        "integer_counts": integer_counts,
        "predicted": float(np.clip(pred, 0.0, 100.0)),
        "uncertainty": float(max(0.0, uncert)),
        "recommended_reaction_time": float(rt),
        "total_stock_concentration_mg_ml": 1.0,
    }


def suggest_experiments(
    fixed_conditions: Dict[str, Any],
    prior_experiments: Optional[List[Dict[str, Any]]] = None,
    n: int = 5,
) -> Dict[str, Any]:
    """
    Suggest new experiments using active learning strategy.
    Returns a dict with keys: suggestions (list) and message (str).
    """

    def uncertainty_acquisition_function(ratios: Tuple[float, float, float], reaction_time: float) -> Tuple[float, float, float]:
        d, n, p = _normalize(*ratios)
        prediction, uncertainty = _predict_with_uncertainty(d, n, p, reaction_time)

        exploration_bonus = 0.0
        max_ratio = max(d, n, p)
        if max_ratio > 0.7:
            exploration_bonus += 0.02
        if reaction_time < 6.0 or reaction_time > 48.0:
            exploration_bonus += 0.01

        uncertainty_score = float(uncertainty + exploration_bonus + 0.05 * prediction)
        return uncertainty_score, float(prediction), float(uncertainty)

    tested_combinations: List[Tuple[np.ndarray, float]] = []
    if prior_experiments:
        for experiment in prior_experiments:
            ratios = np.array([
                float(experiment.get('DspB_ratio', 0.0)),
                float(experiment.get('DNase_I_ratio', 0.0)),
                float(experiment.get('ProK_ratio', 0.0)),
            ], dtype=float)
            total = ratios.sum() + 1e-9
            reaction_time = float(experiment.get('Reaction_Time', 24.0))
            tested_combinations.append((ratios / total, reaction_time))

    def is_too_similar_to_tested(ratios: np.ndarray, reaction_time: float, threshold: float = 0.08) -> bool:
        for tested_ratios, tested_time in tested_combinations:
            if (np.sum(np.abs(ratios - tested_ratios)) < threshold and
                abs(reaction_time - tested_time) < 2.0):
                return True
        return False

    suggestions: List[Dict[str, Any]] = []

    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            dspb = trial.suggest_float("dspb", 0.0, 1.0)
            dnase = trial.suggest_float("dnase", 0.0, 1.0 - dspb)
            prok = 1.0 - dspb - dnase
            reaction_time = trial.suggest_float("reaction_time", 1.0, 72.0)

            ratios_array = np.array([dspb, dnase, prok])
            if is_too_similar_to_tested(ratios_array, reaction_time):
                return 0.0

            score, pred, uncert = uncertainty_acquisition_function((dspb, dnase, prok), reaction_time)
            return -score

        for seed in [42, 123, 456]:
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=seed, warn_independent_sampling=False),
            )
            study.optimize(objective, n_trials=50, show_progress_bar=False)

            if study.best_params:
                best_dspb = float(study.best_params["dspb"])  # type: ignore[index]
                best_dnase = float(study.best_params["dnase"])  # type: ignore[index]
                best_prok = float(1.0 - best_dspb - best_dnase)
                best_reaction_time = float(study.best_params["reaction_time"])  # type: ignore[index]

                score, prediction, uncertainty = uncertainty_acquisition_function(
                    (best_dspb, best_dnase, best_prok), best_reaction_time
                )

                suggestions.append({
                    "dspb": best_dspb,
                    "dnase": best_dnase,
                    "prok": best_prok,
                    "reaction_time": best_reaction_time,
                    "predicted": float(np.clip(prediction, 0.0, 100.0)),
                    "uncertainty": uncertainty,
                    "score": score,
                })

    except Exception:
        rng = np.random.default_rng(42)

        sampling_strategies = [
            lambda: rng.dirichlet(alpha=[1.0, 1.0, 1.0]),
            lambda: rng.dirichlet(alpha=[5.0, 1.0, 1.0]),
            lambda: rng.dirichlet(alpha=[1.0, 5.0, 1.0]),
            lambda: rng.dirichlet(alpha=[1.0, 1.0, 5.0]),
            lambda: rng.dirichlet(alpha=[2.0, 2.0, 1.0]),
            lambda: rng.dirichlet(alpha=[2.0, 1.0, 2.0]),
            lambda: rng.dirichlet(alpha=[1.0, 2.0, 2.0]),
        ]

        for strategy_idx in range(len(sampling_strategies) * 30):
            strategy = sampling_strategies[strategy_idx % len(sampling_strategies)]
            ratios = strategy()
            dspb, dnase, prok = float(ratios[0]), float(ratios[1]), float(ratios[2])

            if strategy_idx % 3 == 0:
                reaction_time = rng.uniform(1.0, 12.0)
            elif strategy_idx % 3 == 1:
                reaction_time = rng.uniform(12.0, 36.0)
            else:
                reaction_time = rng.uniform(36.0, 72.0)

            ratios_array = np.array([dspb, dnase, prok])
            if is_too_similar_to_tested(ratios_array, reaction_time):
                continue

            score, prediction, uncertainty = uncertainty_acquisition_function(
                (dspb, dnase, prok), reaction_time
            )

            suggestions.append({
                "dspb": dspb,
                "dnase": dnase,
                "prok": prok,
                "reaction_time": reaction_time,
                "predicted": float(np.clip(prediction, 0.0, 100.0)),
                "uncertainty": uncertainty,
                "score": score,
            })

    suggestions.sort(key=lambda x: x["score"], reverse=True)

    unique_suggestions: List[Dict[str, Any]] = []
    for suggestion in suggestions:
        is_unique = True
        for existing in unique_suggestions:
            ratios_diff = abs(suggestion["dspb"] - existing["dspb"]) + \
                         abs(suggestion["dnase"] - existing["dnase"]) + \
                         abs(suggestion["prok"] - existing["prok"])
            time_diff = abs(suggestion["reaction_time"] - existing["reaction_time"])

            if ratios_diff < 0.1 and time_diff < 3.0:
                is_unique = False
                break

        if is_unique:
            unique_suggestions.append(suggestion)
        if len(unique_suggestions) >= 3:
            break

    return {
        "suggestions": unique_suggestions[: min(n, 3)],
        "message": f"Found {len(unique_suggestions[: min(n, 3)])} diverse experiment suggestions",
    }