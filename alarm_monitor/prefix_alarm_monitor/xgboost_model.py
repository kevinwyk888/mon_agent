"""XGBoost prefix scorer with soft success targets."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class XGBoostPrefixScorer:
    """Small histogram-based boosted-tree model for tabular prefix features."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = int(seed)
        self.model = None

    @staticmethod
    def _xgboost():
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise RuntimeError(
                "XGBoost is required; install the project with the alarm-monitor extra"
            ) from exc
        return xgb

    def fit(
        self,
        x_train: np.ndarray,
        y_train_success: np.ndarray,
        x_val: np.ndarray,
        y_val_success: np.ndarray,
    ) -> "XGBoostPrefixScorer":
        xgb = self._xgboost()
        train_data = xgb.DMatrix(
            np.asarray(x_train, dtype=np.float32),
            label=np.asarray(y_train_success, dtype=np.float32),
        )
        val_data = xgb.DMatrix(
            np.asarray(x_val, dtype=np.float32),
            label=np.asarray(y_val_success, dtype=np.float32),
        )
        self.model = xgb.train(
            {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
                "learning_rate": 0.04,
                "max_depth": 4,
                "min_child_weight": 20.0,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_lambda": 1.0,
                "seed": self.seed,
                "nthread": 4,
            },
            train_data,
            num_boost_round=400,
            evals=[(val_data, "validation")],
            early_stopping_rounds=40,
            verbose_eval=False,
        )
        return self

    @classmethod
    def load(cls, path: Path, seed: int = 42) -> "XGBoostPrefixScorer":
        xgb = cls._xgboost()
        scorer = cls(seed=seed)
        scorer.model = xgb.Booster()
        scorer.model.load_model(path)
        return scorer

    def predict_success_prob(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("XGBoost scorer has not been fit")
        xgb = self._xgboost()
        data = xgb.DMatrix(np.asarray(features, dtype=np.float32))
        best_iteration = getattr(self.model, "best_iteration", None)
        predictions = self.model.predict(
            data,
            iteration_range=(0, best_iteration + 1) if best_iteration is not None else (0, 0),
        )
        return np.clip(np.asarray(predictions, dtype=np.float64), 1e-6, 1.0 - 1e-6)

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("XGBoost scorer has not been fit")
        self.model.save_model(path)