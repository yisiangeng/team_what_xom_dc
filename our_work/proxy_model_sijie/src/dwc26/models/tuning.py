"""Trial plan and resolved fit settings (spec task2 section 2.14; D-015).

- `lhs` (RNN, LSTM): one Latin hypercube of N = n_trials / (number of input
  options) points for each option (stratum), seeded with [seed, stratum].
  Source: McKay et al. (1979).
- `grid` (DMD): the Cartesian product of the choice dimensions, last fastest.
- B0 and B1: no plan.

The plan is fixed before any fit, so it is reproducible (V7) and the fits can
run in parallel.
"""

from __future__ import annotations

import copy
import itertools
import math

import numpy as np

from our_work.proxy_model_sijie.src.dwc26.config import ConfigError

NEURAL = ("rnn", "lstm")
NEURAL_FIXED = ("seeds", "max_epochs", "patience", "min_delta", "grad_clip_norm", "adam", "dtype",
                "predict_batch")


def lhs_unit(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """(n, d) Latin hypercube in [0, 1): for each dimension in order,
    u = (permutation(n) + uniform(n)) / n."""
    u = np.empty((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        jit = rng.random(n)
        u[:, j] = (perm + jit) / n
    return u


def _map_value(spec: dict, u: float):
    if spec["type"] == "choice":
        vals = spec["values"]
        m = len(vals)
        return vals[min(int(math.floor(u * m)), m - 1)]
    if spec["type"] == "loguniform":
        lo, hi = math.log(spec["low"]), math.log(spec["high"])
        return float(math.exp(lo + u * (hi - lo)))
    raise ConfigError(f"unknown dimension type {spec['type']!r}")


def make_plan(cfg: dict) -> list[dict]:
    """[{'trial': int, 'variant': str, 'settings': {dim: value}}], trials 0..N-1 in stratum order."""
    model = cfg["model"]
    if model not in NEURAL and model != "dmd":
        return []
    tu = cfg["tuning"]
    n_trials, space = tu["n_trials"], tu["space"]
    if not 1 <= n_trials <= 30:
        raise ConfigError(f"tuning.n_trials = {n_trials} is outside [1, 30]")
    dims = list(space)
    plan = []
    if tu["sampler"] == "lhs":
        strata = list(cfg["network"]["input_options"])
        if n_trials % len(strata):
            raise ConfigError(f"tuning.n_trials = {n_trials} is not divisible by the {len(strata)} input options")
        N = n_trials // len(strata)
        for s, variant in enumerate(strata):
            rng = np.random.default_rng([cfg["seed"], s])
            u = lhs_unit(N, len(dims), rng)
            for i in range(N):
                settings = {dim: _map_value(space[dim], float(u[i, j])) for j, dim in enumerate(dims)}
                plan.append({"trial": len(plan), "variant": variant, "settings": settings})
    elif tu["sampler"] == "grid":
        for dim in dims:
            if space[dim]["type"] != "choice":
                raise ConfigError(f"tuning.space.{dim}: the grid sampler needs choice dimensions")
        grid = list(itertools.product(*(space[d]["values"] for d in dims)))
        if len(grid) > n_trials:
            raise ConfigError(f"tuning: the grid has {len(grid)} points, more than n_trials = {n_trials}")
        for point in grid:
            plan.append({"trial": len(plan), "variant": "main", "settings": dict(zip(dims, point))})
    else:
        raise ConfigError(f"tuning.sampler {tu['sampler']!r} is not lhs or grid")
    return plan


def tuned_dims(cfg: dict) -> list[str]:
    return list(cfg.get("tuning", {}).get("space", {}))


def resolve_settings(cfg: dict, variant: str, settings: dict) -> dict:
    """Flat fit cfg = the tuned values plus the fixed settings (section 2.14)."""
    model = cfg["model"]
    dims = tuned_dims(cfg)
    missing = [d for d in dims if d not in settings]
    extra = [d for d in settings if d not in dims]
    if missing or extra:
        raise ConfigError(f"resolve_settings: tuned values {sorted(settings)} do not match the space {dims}")
    if model in ("b0", "b1"):
        return {} if model == "b0" else {"knn": copy.deepcopy(cfg["knn"])}
    out = copy.deepcopy(dict(settings))
    wname = settings["fluid_weights"]
    if wname not in cfg["fluid_weight_sets"]:
        raise ConfigError(f"resolve_settings: unknown fluid weight set {wname!r}")
    out["fluid_weights"] = list(cfg["fluid_weight_sets"][wname])
    out["fluid_weights_name"] = wname
    if model in NEURAL:
        if variant not in cfg["network"]["input_options"]:
            raise ConfigError(f"resolve_settings: variant {variant!r} is not an input option")
        out["cell"] = "lstm" if model == "lstm" else settings["cell"]
        out["input_option"] = variant
        tr = cfg["training"]
        for k in NEURAL_FIXED:
            out[k] = copy.deepcopy(tr[k])
        out["x_affine"] = cfg["network"]["x_affine"]
        out["objective_fluids"] = list(cfg["objective"]["fluids"])
    else:
        if variant != "main":
            raise ConfigError(f"resolve_settings: dmd has the variant 'main' only, got {variant!r}")
        out["seed"] = cfg["seed"]
        out["dmd"] = copy.deepcopy(cfg["dmd"])
        out["gp"] = copy.deepcopy(cfg["gp"])
    return out


def tuned_values(cfg: dict, flat: dict) -> dict:
    """Inverse of resolve_settings: the tuned values in a flat fit cfg (for --settings-from)."""
    out = {}
    for d in tuned_dims(cfg):
        key = "fluid_weights_name" if d == "fluid_weights" else d
        if key not in flat:
            raise ConfigError(f"settings: the source run has no value for the tuned setting {d!r}")
        out[d] = flat[key]
    return out
