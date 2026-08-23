from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cross_model_drift.features import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES, feature_matrix

ModelKind = Literal["logistic", "lightgbm"]


class Predictor(Protocol):
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray: ...


@dataclass
class TrainedModel:
    kind: ModelKind
    estimator: Any
    features: tuple[str, ...] = MODEL_FEATURES
    threshold: float = 0.5
    params: dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = feature_matrix(frame, self.features)
        if self.kind == "lightgbm":
            booster: lgb.Booster = self.estimator
            return np.asarray(booster.predict(matrix), dtype="float64")
        proba = self.estimator.predict_proba(matrix)
        return np.asarray(proba[:, 1], dtype="float64")

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(frame) >= self.threshold).astype("int8")

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": self.kind,
            "features": self.features,
            "threshold": self.threshold,
            "params": self.params,
        }
        if self.kind == "lightgbm":
            model_path = out.with_suffix(".txt")
            self.estimator.save_model(str(model_path))
            payload["booster_path"] = model_path.name
            joblib.dump(payload, out)
        else:
            payload["estimator"] = self.estimator
            joblib.dump(payload, out)
        return out


def load_model(path: str | Path) -> TrainedModel:
    stored = Path(path)
    payload = joblib.load(stored)
    kind: ModelKind = payload["kind"]
    if kind == "lightgbm":
        booster_path = stored.with_name(payload["booster_path"])
        estimator = lgb.Booster(model_file=str(booster_path))
    else:
        estimator = payload["estimator"]
    return TrainedModel(
        kind=kind,
        estimator=estimator,
        features=tuple(payload["features"]),
        threshold=float(payload["threshold"]),
        params=dict(payload.get("params") or {}),
    )


def logistic_pipeline(random_state: int = 42, max_iter: int = 400) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(NUMERIC_FEATURES)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                list(CATEGORICAL_FEATURES),
            ),
        ]
    )
    clf = LogisticRegression(
        max_iter=max_iter,
        class_weight="balanced",
        solver="lbfgs",
        random_state=random_state,
    )
    return Pipeline([("preprocess", preprocess), ("clf", clf)])


def train_logistic(
    train: pd.DataFrame,
    y: pd.Series,
    *,
    threshold: float = 0.5,
    random_state: int = 42,
) -> TrainedModel:
    pipe = logistic_pipeline(random_state=random_state)
    matrix = feature_matrix(train)
    pipe.fit(matrix, y)
    return TrainedModel(
        kind="logistic",
        estimator=pipe,
        threshold=threshold,
        params={"random_state": random_state, "class_weight": "balanced"},
    )


DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,
    "verbosity": -1,
    "n_jobs": -1,
}


def train_lightgbm(
    train: pd.DataFrame,
    y_train: pd.Series,
    valid: pd.DataFrame | None = None,
    y_valid: pd.Series | None = None,
    *,
    params: dict[str, Any] | None = None,
    num_boost_round: int = 300,
    early_stopping_rounds: int = 40,
    threshold: float = 0.5,
) -> TrainedModel:
    merged = {**DEFAULT_LGBM_PARAMS, **(params or {})}
    train_set = lgb.Dataset(feature_matrix(train), label=y_train)
    valid_sets = [train_set]
    valid_names = ["train"]
    callbacks: list[Any] = [lgb.log_evaluation(period=50)]
    if valid is not None and y_valid is not None:
        valid_set = lgb.Dataset(feature_matrix(valid), label=y_valid, reference=train_set)
        valid_sets.append(valid_set)
        valid_names.append("valid")
        callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
    booster = lgb.train(
        merged,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    return TrainedModel(
        kind="lightgbm",
        estimator=booster,
        threshold=threshold,
        params=merged,
    )
