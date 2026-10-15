"""Model registry and proxy loading (spec task2 section 1.3; PRD section 3; D-014, D-017).

Imports are lazy, so that `dmd`, `b0` and `b1` run without importing torch.
"""

from __future__ import annotations

import importlib
import sqlite3
from pathlib import Path

import numpy as np

from our_work.proxy_model_sijie.src.dwc26.config import MODEL_NAMES as CONFIG_MODEL_NAMES
from our_work.proxy_model_sijie.src.dwc26.config import ConfigError
from our_work.proxy_model_sijie.src.dwc26.data.check import ContractError
from our_work.proxy_model_sijie.src.dwc26.data.scale import load_scalers
from our_work.proxy_model_sijie.src.dwc26.models.base import read_meta

PROXIES = {"rnn": "dwc26.models.rnn:RNNProxy", "lstm": "dwc26.models.lstm:LSTMProxy",
           "dmd": "dwc26.models.dmd:DMDProxy"}
BENCHMARKS = {"b0": "dwc26.models.baselines:MeanCurveBenchmark",
              "b1": "dwc26.models.baselines:KNNBenchmark"}
MODEL_NAMES = (*PROXIES, *BENCHMARKS)
if MODEL_NAMES != CONFIG_MODEL_NAMES:  # pragma: no cover - guard against drift
    raise ImportError("dwc26.config.MODEL_NAMES differs from the registry")


def get_class(name: str) -> type:
    """Imports the class of a registered model. Unknown name -> ConfigError.
    ContractError unless extrapolates_time is True for PROXIES, False for BENCHMARKS (V13)."""
    if name in PROXIES:
        target, want = PROXIES[name], True
    elif name in BENCHMARKS:
        target, want = BENCHMARKS[name], False
    else:
        raise ConfigError(f"unknown model {name!r}; registered: {list(MODEL_NAMES)}")
    module, cls_name = target.split(":")
    cls = getattr(importlib.import_module(module), cls_name)
    if cls.extrapolates_time is not want:
        raise ContractError(f"{target}: extrapolates_time must be {want} (V13)")
    if cls.name != name:
        raise ContractError(f"{target}: name {cls.name!r} differs from the registry name {name!r}")
    return cls


def parse_proxy_id(proxy_id: str) -> tuple[str, str]:
    """'<run_id>/<variant>' -> (run_id, variant) (D-014)."""
    parts = str(proxy_id).split("/")
    if len(parts) != 2 or not all(p.strip() for p in parts) or any(p != p.strip() for p in parts):
        raise ConfigError(f"proxy ID {proxy_id!r} is not '<run_id>/<variant>'")
    return parts[0], parts[1]


def load_model(proxy_id: str, models_dir: Path, *, con: sqlite3.Connection | None = None,
               proxies_only: bool = True):
    """Loads <models_dir>/<run_id>/<variant>/. With proxies_only, a benchmark raises
    ContractError (Tasks 4 and 5). With con, the scalers in meta.json must equal
    load_scalers(con) exactly (V4)."""
    run_id, variant = parse_proxy_id(proxy_id)
    path = Path(models_dir) / run_id / variant
    meta = read_meta(path)
    name = meta.get("model")
    if proxies_only and name not in PROXIES:
        raise ContractError(f"{proxy_id}: model {name!r} is a benchmark, not a proxy; it cannot be used for "
                            "history matching or forecasting (PRD §6, V13)")
    cls = get_class(name)
    model = cls.load(path)
    if meta.get("run_id") != run_id or model.variant != variant:
        raise ContractError(f"{proxy_id}: meta.json names run {meta.get('run_id')!r} and variant "
                            f"{model.variant!r}")
    if con is not None:
        xs, ys = load_scalers(con)
        for key, ref in (("y_factor", ys.factor), ("x_lower", xs.lower), ("x_upper", xs.upper)):
            got = meta.get(key)
            if got is None or not np.array_equal(np.asarray(got, dtype=np.float64), ref):
                raise ContractError(f"{proxy_id}: {key} in meta.json differs from the scalers table (V4)")
    return model
