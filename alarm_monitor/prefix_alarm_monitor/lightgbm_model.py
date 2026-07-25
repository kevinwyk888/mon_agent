"""LightGBM prefix scorer with soft success targets."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class LightGBMPrefixScorer:
    """Small gradient-boosted tree model for tabular prefix features."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = int(seed)
        self.model = None

    def fit(
        self,
        x_train: np.ndarray,
        y_train_success: np.ndarray,
        x_val: np.ndarray,
        y_val_success: np.ndarray,
    ) -> "LightGBMPrefixScorer":
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise RuntimeError(
                "LightGBM is required; install the project with the alarm-monitor extra"
            ) from exc

        train_data = lgb.Dataset(
            np.asarray(x_train, dtype=np.float64),
            label=np.asarray(y_train_success, dtype=np.float64),
            free_raw_data=True,
        )
        val_data = lgb.Dataset(
            np.asarray(x_val, dtype=np.float64),
            label=np.asarray(y_val_success, dtype=np.float64),
            reference=train_data,
            free_raw_data=True,
        )
        self.model = lgb.train(
            {
                "objective": "cross_entropy",
                "metric": "cross_entropy",
                "learning_rate": 0.04,
                "num_leaves": 15,
                "max_depth": 4,
                "min_data_in_leaf": 20,
                "bagging_fraction": 0.9,
                "bagging_freq": 1,
                "feature_fraction": 0.9,
                "lambda_l2": 1.0,
                "seed": self.seed,
                "deterministic": True,
                "num_threads": 4,
                "verbosity": -1,
            },
            train_data,
            num_boost_round=400,
            valid_sets=[val_data],
            valid_names=["validation"],
            callbacks=[lgb.early_stopping(40, verbose=False)],
        )
        return self

    @classmethod
    def load(cls, path: Path, seed: int = 42) -> "LightGBMPrefixScorer":
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise RuntimeError(
                "LightGBM is required; install the project with the alarm-monitor extra"
            ) from exc
        scorer = cls(seed=seed)
        scorer.model = lgb.Booster(model_file=str(path))
        return scorer

    def predict_success_prob(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LightGBM scorer has not been fit")
        predictions = self.model.predict(
            np.asarray(features, dtype=np.float64),
            num_iteration=self.model.best_iteration,
        )
        return np.clip(np.asarray(predictions, dtype=np.float64), 1e-6, 1.0 - 1e-6)

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("LightGBM scorer has not been fit")
        self.model.save_model(str(path), num_iteration=self.model.best_iteration)
