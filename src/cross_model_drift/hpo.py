from __future__ import annotations

from collections.abc import Callable
from typing import Any

import optuna
import pandas as pd

from cross_model_drift.features import target_vector
from cross_model_drift.metrics import quality_metrics
from cross_model_drift.models import train_lightgbm

HPO_PARAM_SPACE = (
    "learning_rate",
    "num_leaves",
    "max_depth",
    "min_child_samples",
    "feature_fraction",
    "bagging_fraction",
    "lambda_l1",
    "lambda_l2",
)


def suggest_lgbm_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    }


def make_pr_auc_objective(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    num_boost_round: int = 250,
    threshold: float = 0.5,
) -> Callable[[optuna.Trial], float]:
    y_train = target_vector(train)
    y_valid = target_vector(valid)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_lgbm_params(trial)
        model = train_lightgbm(
            train,
            y_train,
            valid,
            y_valid,
            params=params,
            num_boost_round=num_boost_round,
            threshold=threshold,
        )
        scores = model.predict_proba(valid)
        metrics = quality_metrics(y_valid, scores, threshold=threshold)
        trial.set_user_attr("metrics", metrics)
        return float(metrics["pr_auc"])

    return objective


def run_optuna_hpo(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    n_trials: int = 30,
    timeout: int | None = None,
    seed: int = 42,
    num_boost_round: int = 250,
    threshold: float = 0.5,
) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="lgbm_pr_auc")
    study.optimize(
        make_pr_auc_objective(
            train,
            valid,
            num_boost_round=num_boost_round,
            threshold=threshold,
        ),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False,
    )
    return study
