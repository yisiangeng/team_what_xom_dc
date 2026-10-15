"""Proxy protocol, onset, clipping and V9 input checks (spec task2 sections 1.3, 1.8, 2.2, 2.12).

`ModelBase` holds what every model shares: the input checks, the onset k0,
the zeros before k0, the clip at 0 and its count, `predict`, `forecast`, and
the `meta.json` part of save and load (section 1.5). A subclass implements
`_fit`, `_raw_rollout` (raw values of the steps k0..n_total-1), `_raw_std`,
`_save_artefacts` and `_load_artefacts`.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from importlib import metadata
from numbers import Integral
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from our_work.proxy_model_sijie.src.dwc26.data.check import ContractError, assert_array
from our_work.proxy_model_sijie.src.dwc26.data.load import FLUIDS

FORMAT_VERSION = 1
META_PACKAGES = {"numpy": "numpy", "scipy": "scipy", "torch": "torch", "scikit-learn": "scikit-learn"}
CONTEXT_KEYS = ("run_id", "proxy_id", "t_cut", "settings_from", "x_names", "fluids", "y_factor", "x_lower",
                "x_upper", "prepare_run_id", "arrays_sha256")
META_KEYS = ("format_version", "model", "class", "variant", "proxy_id", "run_id", "extrapolates_time",
             "settings", "k0", "T_fit", "t_cut", "settings_from", "p", "x_names", "fluids", "y_factor",
             "x_lower", "x_upper", "prepare_run_id", "arrays_sha256", "packages", "files", "created")


class TrialFailed(RuntimeError):
    """Numerical failure of one trial (NaN loss, singular basis)."""


@runtime_checkable
class Proxy(Protocol):
    """PRD section 3, extended by D-017."""

    name: str
    variant: str
    extrapolates_time: bool
    T_fit: int
    k0: int
    last_n_clipped: int

    def fit(self, X: np.ndarray, Y: np.ndarray, cfg: dict, *,
            val: tuple[np.ndarray, np.ndarray] | None = None) -> "Proxy": ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def predict_std(self, X: np.ndarray) -> np.ndarray | None: ...
    def rollout(self, X: np.ndarray, n_total: int) -> np.ndarray: ...
    def forecast(self, X: np.ndarray, n_steps: int) -> np.ndarray: ...
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "Proxy": ...


def onset_index(Y: np.ndarray) -> int:
    """k0 = first date with a value > 0 in any case and fluid (D-013).
    ContractError if every value is 0 or if T - k0 < 3."""
    Y = np.asarray(Y)
    assert_array("onset_index Y", Y, (None, None, len(FLUIDS)), nonneg=True)
    active = (Y > 0).any(axis=(0, 2))
    if not active.any():
        raise ContractError("onset_index: every training value is 0")
    k0 = int(np.argmax(active))
    T = Y.shape[1]
    if T - k0 < 3:
        raise ContractError(f"onset_index: T - k0 = {T} - {k0} < 3; DMD needs at least two snapshot pairs")
    return k0


def clip_nonneg(a: np.ndarray) -> tuple[np.ndarray, int]:
    """(max(a, 0), count of a < 0). Values that are not < 0 keep their bits."""
    a = np.asarray(a)
    neg = a < 0
    return np.where(neg, 0.0, a), int(np.count_nonzero(neg))


def check_int(v, name: str, minimum: int = 1) -> int:
    if isinstance(v, bool) or not isinstance(v, Integral):
        raise ContractError(f"{name} must be an integer, got {type(v).__name__}")
    if v < minimum:
        raise ContractError(f"{name} must be >= {minimum}, got {v}")
    return int(v)


def _check_box(name: str, X: np.ndarray) -> None:
    if not np.isfinite(X).all():
        raise ContractError(f"{name}: {int((~np.isfinite(X)).sum())} non-finite values")
    if (X < 0).any() or (X > 1).any():
        raise ContractError(f"{name}: values outside [0, 1] (min {X.min()}, max {X.max()}); "
                            "evaluate the prior first and call the proxy only inside the box (D-003)")


def check_X(X: np.ndarray, p: int) -> np.ndarray:
    """V9 input of predict, predict_std, rollout, forecast: (n, p), n >= 1, finite, 0 <= X <= 1."""
    X = np.asarray(X)
    if X.dtype.kind not in "fiu":
        raise ContractError(f"X: numeric array expected, got dtype {X.dtype}")
    X = X.astype(np.float64, copy=False)
    if X.ndim != 2 or X.shape[1] != p or X.shape[0] < 1:
        raise ContractError(f"X: shape {X.shape}, expected (n >= 1, {p})")
    _check_box("X", X)
    return X


def check_XY(X: np.ndarray, Y: np.ndarray) -> None:
    """V9 input of fit: X (n, p) float64 in [0, 1]; Y (n, T, 3) float64, finite, >= 0."""
    if not isinstance(X, np.ndarray) or X.dtype != np.float64 or X.ndim != 2 or X.shape[0] < 1:
        raise ContractError(f"fit X: expected a float64 (n, p) array, got "
                            f"{getattr(X, 'dtype', type(X).__name__)} {getattr(X, 'shape', '')}")
    _check_box("fit X", X)
    assert_array("fit Y", Y, (X.shape[0], None, len(FLUIDS)), nonneg=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def package_versions() -> dict[str, str]:
    out = {}
    for key, dist in META_PACKAGES.items():
        try:
            out[key] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[key] = "not installed"
    return out


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def class_path(cls: type) -> str:
    return f"{cls.__module__}:{cls.__name__}"


class ModelBase:
    """Shared part of every model (suggested structure of spec section 1.3)."""

    name: str = ""
    extrapolates_time: bool = False

    def __init__(self) -> None:
        self.variant = "main"
        self.T_fit: int | None = None
        self.k0: int | None = None
        self.p: int | None = None
        self.settings: dict = {}
        self.last_n_clipped = 0
        self.context: dict = {k: None for k in CONTEXT_KEYS}

    # --- fit ---------------------------------------------------------------------
    def fit(self, X: np.ndarray, Y: np.ndarray, cfg: dict, *,
            val: tuple[np.ndarray, np.ndarray] | None = None) -> "ModelBase":
        check_XY(X, Y)
        if val is not None:
            if not isinstance(val, tuple) or len(val) != 2:
                raise ContractError("fit: val must be a tuple (X_val, Y_val)")
            check_XY(*val)
            if val[0].shape[1] != X.shape[1] or val[1].shape[1] != Y.shape[1]:
                raise ContractError(f"fit: val has p = {val[0].shape[1]} and T = {val[1].shape[1]}, "
                                    f"fit has p = {X.shape[1]} and T = {Y.shape[1]}")
        k0 = onset_index(Y)
        self.k0, self.T_fit, self.p = k0, int(Y.shape[1]), int(X.shape[1])
        self.settings = dict(cfg)
        self._fit(X, Y, cfg, val)
        self.last_n_clipped = 0
        return self

    def _fit(self, X, Y, cfg, val) -> None:
        raise NotImplementedError

    def _require_fitted(self) -> None:
        if self.T_fit is None:
            raise ContractError(f"{self.name}: the model is not fitted")

    # --- prediction ----------------------------------------------------------------
    def _raw_rollout(self, X: np.ndarray, n_total: int) -> np.ndarray:
        """(n, n_total - k0, 3) raw values of the steps k0..n_total-1."""
        raise NotImplementedError

    def _raw_std(self, X: np.ndarray) -> np.ndarray | None:
        """(n, T_fit - k0, 3) std of the steps k0..T_fit-1, or None."""
        return None

    def _rollout(self, X: np.ndarray, n_total: int) -> tuple[np.ndarray, np.ndarray]:
        n, k0 = X.shape[0], self.k0
        out = np.zeros((n, n_total, len(FLUIDS)))
        neg = np.zeros(n_total, dtype=np.int64)
        if n_total > k0:
            raw = self._raw_rollout(X, n_total)
            assert_array(f"{self.name}.raw_rollout", raw, (n, n_total - k0, len(FLUIDS)))
            negm = raw < 0
            out[:, k0:] = np.where(negm, 0.0, raw)
            neg[k0:] = negm.sum(axis=(0, 2))
        assert_array(f"{self.name}.rollout", out, (n, n_total, len(FLUIDS)), nonneg=True)
        return out, neg

    def rollout(self, X: np.ndarray, n_total: int) -> np.ndarray:
        """(n, n_total, 3): steps 0..n_total-1, clipped at 0, exactly 0 for k < k0."""
        self._require_fitted()
        n_total = check_int(n_total, "n_total")
        X = check_X(X, self.p)
        out, neg = self._rollout(X, n_total)
        self.last_n_clipped = int(neg.sum())
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        """(n, T_fit, 3) scaled, >= 0."""
        self._require_fitted()
        return self.rollout(X, self.T_fit)

    def forecast(self, X: np.ndarray, n_steps: int) -> np.ndarray:
        """(n, n_steps, 3): steps T_fit..T_fit+n_steps-1 of the rollout. `last_n_clipped`
        counts the returned steps."""
        self._require_fitted()
        n_steps = check_int(n_steps, "n_steps")
        X = check_X(X, self.p)
        out, neg = self._rollout(X, self.T_fit + n_steps)
        self.last_n_clipped = int(neg[self.T_fit:].sum())
        return np.ascontiguousarray(out[:, self.T_fit:])

    def predict_std(self, X: np.ndarray) -> np.ndarray | None:
        """(n, T_fit, 3) scaled, >= 0, exactly 0 for k < k0; None for benchmarks."""
        self._require_fitted()
        X = check_X(X, self.p)
        s = self._raw_std(X)
        if s is None:
            return None
        n, k0 = X.shape[0], self.k0
        assert_array(f"{self.name}.raw_std", s, (n, self.T_fit - k0, len(FLUIDS)), nonneg=True)
        out = np.zeros((n, self.T_fit, len(FLUIDS)))
        out[:, k0:] = s
        return out

    # --- save and load ---------------------------------------------------------------
    def set_context(self, **kw) -> None:
        """Run and data context written to meta.json (scalers, run ID, window)."""
        unknown = set(kw) - set(CONTEXT_KEYS)
        if unknown:
            raise ContractError(f"set_context: unknown keys {sorted(unknown)}")
        self.context.update({k: _jsonable(v) for k, v in kw.items()})

    def _save_artefacts(self, path: Path) -> list[str]:
        """Writes the model files into path; returns their names."""
        raise NotImplementedError

    def _meta_extra(self) -> dict:
        return {}

    def save(self, path: Path) -> None:
        self._require_fitted()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        names = self._save_artefacts(path)
        files = {n: sha256_file(path / n) for n in names}
        ctx = self.context
        meta = {
            "format_version": FORMAT_VERSION,
            "model": self.name,
            "class": class_path(type(self)),
            "variant": self.variant,
            "proxy_id": ctx.get("proxy_id"),
            "run_id": ctx.get("run_id"),
            "extrapolates_time": bool(self.extrapolates_time),
            "settings": _jsonable(self.settings),
            "k0": int(self.k0),
            "T_fit": int(self.T_fit),
            "t_cut": ctx.get("t_cut"),
            "settings_from": ctx.get("settings_from"),
            "p": int(self.p),
            "x_names": ctx.get("x_names"),
            "fluids": ctx.get("fluids") or list(FLUIDS),
            "y_factor": ctx.get("y_factor"),
            "x_lower": ctx.get("x_lower"),
            "x_upper": ctx.get("x_upper"),
            "prepare_run_id": ctx.get("prepare_run_id"),
            "arrays_sha256": ctx.get("arrays_sha256"),
            "packages": package_versions(),
            "files": files,
            "created": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        }
        meta.update(_jsonable(self._meta_extra()))
        (path / "meta.json").write_text(json.dumps(meta, indent=1, sort_keys=False), encoding="utf-8")

    def _load_artefacts(self, path: Path, meta: dict) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path) -> "ModelBase":
        path = Path(path)
        meta = read_meta(path)
        if meta.get("format_version") != FORMAT_VERSION:
            raise ContractError(f"{path}: format_version {meta.get('format_version')}, expected {FORMAT_VERSION}")
        if meta.get("class") != class_path(cls) or meta.get("model") != cls.name:
            raise ContractError(f"{path}: meta.json is for {meta.get('class')}, not {class_path(cls)}")
        files = meta.get("files")
        if not isinstance(files, dict) or not files:
            raise ContractError(f"{path}: meta.json lists no files")
        for name, sha in files.items():
            f = path / name
            if not f.is_file():
                raise ContractError(f"{path}: missing artefact {name}")
            if sha256_file(f) != sha:
                raise ContractError(f"{path}: sha256 of {name} differs from meta.json; the file was changed")
        m = cls()
        m.variant = meta["variant"]
        m.k0, m.T_fit, m.p = int(meta["k0"]), int(meta["T_fit"]), int(meta["p"])
        m.settings = meta["settings"]
        m.context = {k: meta.get(k) for k in CONTEXT_KEYS}
        m._load_artefacts(path, meta)
        return m


def read_meta(path: Path) -> dict:
    f = Path(path) / "meta.json"
    if not f.is_file():
        raise ContractError(f"no meta.json in {Path(path).as_posix()}")
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ContractError(f"{f.as_posix()}: invalid JSON: {e}") from e
