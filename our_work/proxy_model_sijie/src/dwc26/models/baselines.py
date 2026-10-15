"""Benchmarks B0 and B1 (spec task2 section 2.11; PRD section 6; V6).

They are not proxies: `extrapolates_time = False`, `forecast` and a rollout
beyond T_fit raise NotImplementedError, `predict_std` is None, and they are
registered in BENCHMARKS only, so `load_model(..., proxies_only=True)` keeps
them out of Tasks 4 and 5 (V13).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from our_work.proxy_model_sijie.src.dwc26.data.check import ContractError
from our_work.proxy_model_sijie.src.dwc26.models.base import ModelBase

NO_CONTINUATION = "benchmark: no time continuation (PRD §6)"


class _Benchmark(ModelBase):
    extrapolates_time = False

    def _rollout(self, X, n_total):
        if n_total > self.T_fit:
            raise NotImplementedError(NO_CONTINUATION)
        return super()._rollout(X, n_total)

    def forecast(self, X, n_steps):
        raise NotImplementedError(NO_CONTINUATION)

    def _curves(self, X: np.ndarray) -> np.ndarray:
        """(n, T_fit, 3) predicted curves."""
        raise NotImplementedError

    def _raw_rollout(self, X, n_total):
        return np.ascontiguousarray(self._curves(X)[:, self.k0:n_total])


class MeanCurveBenchmark(_Benchmark):
    """B0: the mean training curve, the same for every x."""

    name = "b0"

    def _fit(self, X, Y, cfg, val):
        self.mean_curve = Y.mean(axis=0)

    def _curves(self, X):
        return np.broadcast_to(self.mean_curve, (X.shape[0], *self.mean_curve.shape))

    def _save_artefacts(self, path: Path) -> list[str]:
        with open(path / "b0.npz", "wb") as fh:
            np.savez(fh, mean_curve=self.mean_curve)
        return ["b0.npz"]

    def _load_artefacts(self, path: Path, meta: dict) -> None:
        with np.load(path / "b0.npz", allow_pickle=False) as z:
            self.mean_curve = z["mean_curve"]


class KNNBenchmark(_Benchmark):
    """B1: KNeighborsRegressor(3, weights='distance', algorithm='brute') from scaled X
    to Y reshaped to (n, T_fit * 3)."""

    name = "b1"

    def _build(self) -> None:
        from sklearn.neighbors import KNeighborsRegressor

        knn = self.settings["knn"]
        if len(self.X_fit) < knn["n_neighbors"]:
            raise ContractError(f"b1: {len(self.X_fit)} training cases < n_neighbors = {knn['n_neighbors']}")
        self.knn = KNeighborsRegressor(n_neighbors=knn["n_neighbors"], weights=knn["weights"],
                                       algorithm=knn["algorithm"], metric="euclidean")
        self.knn.fit(self.X_fit, self.Y_fit.reshape(len(self.Y_fit), -1))

    def _fit(self, X, Y, cfg, val):
        if "knn" not in cfg:
            raise ContractError("b1: the fit settings have no 'knn' block")
        self.X_fit, self.Y_fit = X.copy(), Y.copy()
        self._build()

    def _curves(self, X):
        return self.knn.predict(X).reshape(X.shape[0], self.T_fit, -1)

    def _save_artefacts(self, path: Path) -> list[str]:
        with open(path / "b1.npz", "wb") as fh:
            np.savez(fh, X_fit=self.X_fit, Y_fit=self.Y_fit)
        return ["b1.npz"]

    def _load_artefacts(self, path: Path, meta: dict) -> None:
        with np.load(path / "b1.npz", allow_pickle=False) as z:
            self.X_fit, self.Y_fit = z["X_fit"], z["Y_fit"]
        self._build()
