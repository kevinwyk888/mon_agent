"""A dependency-light two-hidden-layer neural prefix scorer."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-values))


def _binary_cross_entropy(targets: np.ndarray, predictions: np.ndarray) -> float:
    predictions = np.clip(predictions, 1e-8, 1.0 - 1e-8)
    return float(
        -np.mean(targets * np.log(predictions) + (1.0 - targets) * np.log(1.0 - predictions))
    )


class LightweightMLPScorer:
    """Two-layer MLP trained with Adam and soft-label binary cross entropy."""

    def __init__(
        self,
        hidden_sizes: tuple[int, int] = (32, 16),
        learning_rate: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 512,
        l2: float = 1e-4,
        patience: int = 12,
        seed: int = 42,
    ) -> None:
        self.hidden_sizes = hidden_sizes
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.l2 = float(l2)
        self.patience = int(patience)
        self.seed = int(seed)
        self.parameters: Dict[str, np.ndarray] = {}
        self.history: list[Dict[str, float]] = []
        self.best_epoch = 0

    def _initialize(self, num_features: int, rng: np.random.Generator) -> None:
        hidden_1, hidden_2 = self.hidden_sizes
        self.parameters = {
            "w1": rng.normal(0.0, np.sqrt(2.0 / num_features), (num_features, hidden_1)),
            "b1": np.zeros(hidden_1, dtype=np.float64),
            "w2": rng.normal(0.0, np.sqrt(2.0 / hidden_1), (hidden_1, hidden_2)),
            "b2": np.zeros(hidden_2, dtype=np.float64),
            "w3": rng.normal(0.0, np.sqrt(1.0 / hidden_2), (hidden_2, 1)),
            "b3": np.zeros(1, dtype=np.float64),
        }

    def _forward(self, features: np.ndarray) -> tuple[np.ndarray, ...]:
        z1 = features @ self.parameters["w1"] + self.parameters["b1"]
        hidden_1 = np.maximum(z1, 0.0)
        z2 = hidden_1 @ self.parameters["w2"] + self.parameters["b2"]
        hidden_2 = np.maximum(z2, 0.0)
        logits = hidden_2 @ self.parameters["w3"] + self.parameters["b3"]
        predictions = _sigmoid(logits[:, 0])
        return z1, hidden_1, z2, hidden_2, predictions

    def fit(
        self,
        x_train: np.ndarray,
        y_train_success: np.ndarray,
        x_val: np.ndarray,
        y_val_success: np.ndarray,
    ) -> "LightweightMLPScorer":
        x_train = np.asarray(x_train, dtype=np.float64)
        y_train_success = np.asarray(y_train_success, dtype=np.float64)
        x_val = np.asarray(x_val, dtype=np.float64)
        y_val_success = np.asarray(y_val_success, dtype=np.float64)
        rng = np.random.default_rng(self.seed)
        self._initialize(x_train.shape[1], rng)

        first_moment = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        second_moment = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        best_parameters = {name: value.copy() for name, value in self.parameters.items()}
        best_loss = float("inf")
        stale_epochs = 0
        update_step = 0

        for epoch in range(1, self.epochs + 1):
            order = rng.permutation(x_train.shape[0])
            for start in range(0, x_train.shape[0], self.batch_size):
                batch_indices = order[start : start + self.batch_size]
                features = x_train[batch_indices]
                targets = y_train_success[batch_indices]
                z1, hidden_1, z2, hidden_2, predictions = self._forward(features)

                grad_logits = (predictions - targets)[:, None] / max(len(batch_indices), 1)
                gradients = {
                    "w3": hidden_2.T @ grad_logits + self.l2 * self.parameters["w3"],
                    "b3": grad_logits.sum(axis=0),
                }
                grad_hidden_2 = grad_logits @ self.parameters["w3"].T
                grad_z2 = grad_hidden_2 * (z2 > 0.0)
                gradients["w2"] = hidden_1.T @ grad_z2 + self.l2 * self.parameters["w2"]
                gradients["b2"] = grad_z2.sum(axis=0)
                grad_hidden_1 = grad_z2 @ self.parameters["w2"].T
                grad_z1 = grad_hidden_1 * (z1 > 0.0)
                gradients["w1"] = features.T @ grad_z1 + self.l2 * self.parameters["w1"]
                gradients["b1"] = grad_z1.sum(axis=0)

                update_step += 1
                for name, gradient in gradients.items():
                    first_moment[name] = 0.9 * first_moment[name] + 0.1 * gradient
                    second_moment[name] = 0.999 * second_moment[name] + 0.001 * gradient**2
                    corrected_first = first_moment[name] / (1.0 - 0.9**update_step)
                    corrected_second = second_moment[name] / (1.0 - 0.999**update_step)
                    self.parameters[name] -= self.learning_rate * corrected_first / (
                        np.sqrt(corrected_second) + 1e-8
                    )

            train_loss = _binary_cross_entropy(
                y_train_success, self.predict_success_prob(x_train)
            )
            val_loss = _binary_cross_entropy(y_val_success, self.predict_success_prob(x_val))
            self.history.append(
                {"epoch": float(epoch), "train_bce": train_loss, "val_bce": val_loss}
            )
            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                self.best_epoch = epoch
                best_parameters = {
                    name: value.copy() for name, value in self.parameters.items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break

        self.parameters = best_parameters
        return self

    def predict_success_prob(self, features: np.ndarray) -> np.ndarray:
        if not self.parameters:
            raise RuntimeError("MLP scorer has not been fit")
        return self._forward(np.asarray(features, dtype=np.float64))[-1]

    @classmethod
    def load(cls, path: Path, seed: int = 42) -> "LightweightMLPScorer":
        with np.load(path) as artifact:
            hidden_sizes = (int(artifact["w1"].shape[1]), int(artifact["w2"].shape[1]))
            scorer = cls(hidden_sizes=hidden_sizes, seed=seed)
            scorer.parameters = {
                name: np.asarray(artifact[name], dtype=np.float64)
                for name in ("w1", "b1", "w2", "b2", "w3", "b3")
            }
            scorer.best_epoch = int(artifact["best_epoch"][0])
        return scorer

    def save(self, path: Path) -> None:
        if not self.parameters:
            raise RuntimeError("MLP scorer has not been fit")
        np.savez_compressed(
            path,
            **self.parameters,
            best_epoch=np.asarray([self.best_epoch], dtype=np.int64),
        )
