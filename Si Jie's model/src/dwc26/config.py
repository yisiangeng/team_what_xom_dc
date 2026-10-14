"""Config loading and validation (spec task1 sections 1.2 and 3; spec task2 sections 1.2 and 3)."""

from __future__ import annotations

import copy
import math
import re
from numbers import Real
from pathlib import Path

import yaml

QUANTITY_KEYS = ("oil_rate", "oil_cum", "gas_rate", "gas_cum", "water_rate", "water_cum")
FIELD_REQUIRED = ("date", "oil_rate", "gas_rate", "water_rate")

# Allowed ranges of the fixed tolerances (spec task1 section 3). A value
# outside its range needs a new entry in docs/decisions.md and a spec change.
CHECK_RANGES = {
    "c4_rel_tol": (1.0e-9, 1.0e-4),
    "c6_rel_tol": (1.0e-6, 0.05),
    "c6_floor_frac": (0.0, 1.0e-2),
    "c8_abs_tol": (0.0, 1.0),
    "c10_rel_tol": (0.0, 1.0e-3),   # amendment A1 (D-009)
}
RATE_DEFINITIONS = ("auto", "R1", "R2")
X_BOUND_SOURCES = ("design", "override")
DATE_ROUNDS = ("ms", "us")


class ConfigError(ValueError):
    """A required config key is missing or has an invalid value."""


def _get(d: dict, key: str, where: str):
    if not isinstance(d, dict):
        raise ConfigError(f"{where} must be a mapping")
    if key not in d:
        raise ConfigError(f"missing config key: {where}.{key}" if where else f"missing config key: {key}")
    return d[key]


def _str(d: dict, key: str, where: str) -> str:
    v = _get(d, key, where)
    if not isinstance(v, str) or v == "":
        raise ConfigError(f"{where}.{key} must be a non-empty string")
    return v


def _int(d: dict, key: str, where: str, minimum: int | None = None) -> int:
    v = _get(d, key, where)
    if isinstance(v, bool) or not isinstance(v, int):
        raise ConfigError(f"{where}.{key} must be an integer")
    if minimum is not None and v < minimum:
        raise ConfigError(f"{where}.{key} must be >= {minimum}")
    return v


def _num(d: dict, key: str, where: str) -> float:
    v = _get(d, key, where)
    if isinstance(v, bool) or not isinstance(v, Real):
        raise ConfigError(f"{where}.{key} must be a number")
    return float(v)


def _mapping(d: dict, key: str, where: str) -> dict:
    v = _get(d, key, where)
    if not isinstance(v, dict) or not v:
        raise ConfigError(f"{where}.{key} must be a non-empty mapping")
    return v


def _pattern(d: dict, key: str, where: str) -> None:
    v = _str(d, key, where)
    try:
        rx = re.compile(v)
    except re.error as e:
        raise ConfigError(f"{where}.{key} is not a valid regular expression: {e}") from e
    if rx.groups != 1:
        raise ConfigError(f"{where}.{key} must have exactly one capture group (the case number)")


def validate_config(cfg: dict) -> None:
    """Raises ConfigError for a missing or invalid key. Returns None."""
    if not isinstance(cfg, dict):
        raise ConfigError("the config must be a mapping")
    _int(cfg, "seed", "")

    paths = _mapping(cfg, "paths", "")
    for k in ("file01", "file02", "db", "arrays", "report", "logs"):
        _str(paths, k, "paths")

    f01 = _mapping(cfg, "file01", "")
    _str(f01, "sheet", "file01")
    _str(f01, "label_column", "file01")
    _pattern(f01, "label_pattern", "file01")
    cols = _mapping(f01, "columns", "file01")
    if not all(isinstance(k, str) and isinstance(v, str) and v for k, v in cols.items()):
        raise ConfigError("file01.columns must map header strings to internal names")
    if len(set(cols.values())) != len(cols):
        raise ConfigError("file01.columns: internal names must be unique")

    f02 = _mapping(cfg, "file02", "")
    rows = [_int(f02, k, "file02", minimum=1) for k in ("names_row", "units_row", "first_data_row")]
    if not rows[0] < rows[2] or not rows[1] < rows[2] or rows[0] == rows[1]:
        raise ConfigError("file02: names_row and units_row must differ and precede first_data_row")
    _str(f02, "date_column", "file02")
    _pattern(f02, "case_pattern", "file02")
    _str(f02, "history_column", "file02")
    sheets = _mapping(f02, "sheets", "file02")
    if set(sheets) != set(QUANTITY_KEYS):
        raise ConfigError(f"file02.sheets must have exactly the keys {list(QUANTITY_KEYS)}")
    for q, spec in sheets.items():
        _str(spec, "name", f"file02.sheets.{q}")
        _str(spec, "unit", f"file02.sheets.{q}")
    field = _mapping(f02, "field_sheet", "file02")
    _str(field, "name", "file02.field_sheet")
    fcols = _mapping(field, "columns", "file02.field_sheet")
    missing = [n for n in FIELD_REQUIRED if n not in fcols.values()]
    if missing:
        raise ConfigError(f"file02.field_sheet.columns must map to {list(FIELD_REQUIRED)}; missing {missing}")
    if _str(f02, "date_round", "file02") not in DATE_ROUNDS:
        raise ConfigError(f"file02.date_round must be one of {list(DATE_ROUNDS)}")

    _mapping(cfg, "split", "")  # ranges are validated by dwc26.data.split.assign_split

    xb = _get(cfg, "x_bounds", "")
    if _str(xb, "source", "x_bounds") not in X_BOUND_SOURCES:
        raise ConfigError(f"x_bounds.source must be one of {list(X_BOUND_SOURCES)}")
    _get(xb, "override", "x_bounds")   # checked against the data by fit_x_scaler
    _get(xb, "citation", "x_bounds")

    rates = _get(cfg, "rates", "")
    if _str(rates, "definition", "rates") not in RATE_DEFINITIONS:
        raise ConfigError(f"rates.definition must be one of {list(RATE_DEFINITIONS)}")

    checks = _mapping(cfg, "checks", "")
    for k, (lo, hi) in CHECK_RANGES.items():
        v = _num(checks, k, "checks")
        if not lo <= v <= hi:
            raise ConfigError(f"checks.{k} = {v} is outside the allowed range [{lo}, {hi}] (spec section 3)")


def load_config(path: Path) -> dict:
    """Reads a YAML config with yaml.safe_load and validates it."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigError(f"cannot read config {path}: {e}") from e
    try:
        cfg = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in {path}: {e}") from e
    validate_config(cfg)
    return cfg


# --- Task 2: model configs (spec task2 sections 1.2 and 3) ------------------------

MODEL_NAMES = ("rnn", "lstm", "dmd", "b0", "b1")    # same order as dwc26.models.registry.MODEL_NAMES
NEURAL = ("rnn", "lstm")
CONFIG_FLUIDS = ("oil", "gas", "water")
_COMMON_KEYS = ("model", "seed", "paths", "data", "objective", "report")
MODEL_KEYS = {
    "rnn": _COMMON_KEYS + ("fluid_weight_sets", "network", "training", "tuning", "speed"),
    "lstm": _COMMON_KEYS + ("fluid_weight_sets", "network", "training", "tuning", "speed"),
    "dmd": _COMMON_KEYS + ("fluid_weight_sets", "dmd", "gp", "tuning", "speed"),
    "b0": _COMMON_KEYS,
    "b1": _COMMON_KEYS + ("knn",),
}
SEED_MAX = 2**32 - 1          # numpy SeedSequence and scikit-learn random_state need 0 <= seed < 2**32
WEIGHT_SUM_TOL = 1e-9
KNN_FIXED = {"n_neighbors": 3, "weights": "distance", "algorithm": "brute"}
INPUT_OPTIONS = ("A", "B", "C")
NEURAL_DIMS = ("num_layers", "hidden_size", "lr", "batch_size", "fluid_weights")
DMD_DIMS = ("rank", "kernel", "fluid_weights", "gp_target")     # A2 (D-019): rank replaces energy
DMD_BASES = ("real", "exact")
DMD_RANK_RANGE = (1, 12)
KERNELS = ("matern25", "matern15", "rbf")
GP_TARGETS = ("coef", "proj")
SPEC3 = "(spec task2 section 3)"


def _exact_keys(d: dict, keys, where: str) -> None:
    name = where or "the config"
    if not isinstance(d, dict):
        raise ConfigError(f"{name} must be a mapping")
    missing = [k for k in keys if k not in d]
    if missing:
        raise ConfigError(f"missing config key(s) in {name}: {missing}")
    extra = [k for k in d if k not in keys]
    if extra:
        raise ConfigError(f"unknown config key(s) in {name}: {extra}")


def _bool(d: dict, key: str, where: str) -> bool:
    v = _get(d, key, where)
    if not isinstance(v, bool):
        raise ConfigError(f"{where}.{key} must be true or false")
    return v


def _int_range(d: dict, key: str, where: str, lo: int, hi: int | None) -> int:
    v = _int(d, key, where)
    if v < lo or (hi is not None and v > hi):
        raise ConfigError(f"{where}.{key} = {v} is outside the allowed range [{lo}, {hi}] {SPEC3}")
    return v


def _num_range(d: dict, key: str, where: str, lo: float | None, hi: float | None, *,
               lo_open: bool = False, hi_open: bool = False) -> float:
    v = _num(d, key, where)
    if not math.isfinite(v):
        raise ConfigError(f"{where}.{key} must be finite")
    below = lo is not None and (v <= lo if lo_open else v < lo)
    above = hi is not None and (v >= hi if hi_open else v > hi)
    if below or above:
        lb, rb = ("(" if lo_open else "["), (")" if hi_open else "]")
        raise ConfigError(f"{where}.{key} = {v} is outside the allowed range {lb}{lo}, {hi}{rb} {SPEC3}")
    return v


def _is_num(x) -> bool:
    return isinstance(x, Real) and not isinstance(x, bool) and math.isfinite(x)


def _bounds(d: dict, key: str, where: str) -> tuple[float, float]:
    v = _get(d, key, where)
    if not isinstance(v, list) or len(v) != 2 or not all(_is_num(x) for x in v):
        raise ConfigError(f"{where}.{key} must be [low, high]")
    lo, hi = float(v[0]), float(v[1])
    if not 0 < lo < hi:
        raise ConfigError(f"{where}.{key} must satisfy 0 < low < high, got {v}")
    return lo, hi


def _choice_values(spec: dict, where: str) -> list:
    vals = _get(spec, "values", where)
    if not isinstance(vals, list) or not vals:
        raise ConfigError(f"{where}.values must be a non-empty list")
    if len({repr(v) for v in vals}) != len(vals):
        raise ConfigError(f"{where}.values has duplicates")
    return vals


def _dim(space: dict, dim: str, where: str, allowed_types=("choice",)) -> dict:
    spec = _get(space, dim, where)
    if not isinstance(spec, dict):
        raise ConfigError(f"{where}.{dim} must be a mapping")
    t = _str(spec, "type", f"{where}.{dim}")
    if t not in allowed_types:
        raise ConfigError(f"{where}.{dim}.type must be one of {list(allowed_types)}, got {t!r}")
    if t == "choice":
        _exact_keys(spec, ("type", "values"), f"{where}.{dim}")
        _choice_values(spec, f"{where}.{dim}")
    else:
        _exact_keys(spec, ("type", "low", "high"), f"{where}.{dim}")
    return spec


def _ints_in(vals: list, where: str, lo: int, hi: int | None) -> None:
    for v in vals:
        if isinstance(v, bool) or not isinstance(v, int) or v < lo or (hi is not None and v > hi):
            raise ConfigError(f"{where}: value {v!r} must be an integer in [{lo}, {hi}] {SPEC3}")


def _subset(vals: list, allowed, where: str) -> None:
    bad = [v for v in vals if v not in allowed]
    if bad:
        raise ConfigError(f"{where}: values {bad} not in {list(allowed)} {SPEC3}")


def _seed(v, where: str) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= SEED_MAX:
        raise ConfigError(f"{where} must be an integer in [0, 2**32 - 1], got {v!r}")
    return v


def _validate_weight_sets(cfg: dict) -> dict:
    sets = _mapping(cfg, "fluid_weight_sets", "")
    out = {}
    for name, w in sets.items():
        where = f"fluid_weight_sets.{name}"
        if not isinstance(w, list) or len(w) != len(CONFIG_FLUIDS) or not all(_is_num(x) for x in w):
            raise ConfigError(f"{where} must be a list of {len(CONFIG_FLUIDS)} numbers [oil, gas, water]")
        w = [float(x) for x in w]
        if min(w) < 0:
            raise ConfigError(f"{where}: weights must be >= 0")
        if not (w[0] > 0 and w[2] > 0):
            raise ConfigError(f"{where}: oil and water weights must be > 0")
        s = math.fsum(w)
        if abs(s - 1.0) > WEIGHT_SUM_TOL:
            raise ConfigError(f"{where} sums to {s}, not 1 (tolerance {WEIGHT_SUM_TOL})")
        out[str(name)] = [x / s for x in w]
    return out


def _validate_common(cfg: dict, model: str) -> None:
    _seed(_get(cfg, "seed", ""), "seed")
    paths = _get(cfg, "paths", "")
    _exact_keys(paths, ("db", "models", "logs"), "paths")
    for k in ("db", "models", "logs"):
        _str(paths, k, "paths")
    data = _get(cfg, "data", "")
    _exact_keys(data, ("t_cut",), "data")
    if data["t_cut"] is not None:
        _int_range(data, "t_cut", "data", 3, None)
    obj = _get(cfg, "objective", "")
    _exact_keys(obj, ("fluids",), "objective")
    fl = obj["fluids"]
    if not isinstance(fl, list) or not fl or len(set(map(str, fl))) != len(fl):
        raise ConfigError("objective.fluids must be a non-empty list without duplicates")
    _subset(fl, CONFIG_FLUIDS, "objective.fluids")
    rep = _get(cfg, "report", "")
    _exact_keys(rep, ("plateau_frac", "breakthrough_frac", "forecast_steps", "coverage_z"), "report")
    _num_range(rep, "plateau_frac", "report", 0.5, 1.0)
    _num_range(rep, "breakthrough_frac", "report", 0.0, 0.5)
    _int_range(rep, "forecast_steps", "report", 1, 400)
    _num_range(rep, "coverage_z", "report", 0.0, None, lo_open=True)
    if "speed" in MODEL_KEYS[model]:
        sp = _get(cfg, "speed", "")
        _exact_keys(sp, ("n_batch_rows", "n_calls", "call_rows", "mcmc_calls", "limit_s"), "speed")
        for k in ("n_batch_rows", "n_calls", "call_rows", "mcmc_calls"):
            _int_range(sp, k, "speed", 1, None)
        _num_range(sp, "limit_s", "speed", 0.0, None, lo_open=True)


def _validate_neural(cfg: dict, model: str, weight_names) -> None:
    net = _get(cfg, "network", "")
    _exact_keys(net, ("input_options", "x_affine"), "network")
    opts = net["input_options"]
    if not isinstance(opts, list) or not opts or len(set(map(str, opts))) != len(opts):
        raise ConfigError("network.input_options must be a non-empty list without duplicates")
    _subset(opts, INPUT_OPTIONS, "network.input_options")
    _bool(net, "x_affine", "network")

    tr = _get(cfg, "training", "")
    _exact_keys(tr, ("seeds", "tune_seeds", "max_epochs", "patience", "min_delta", "grad_clip_norm", "adam",
                     "dtype", "torch_threads", "n_jobs", "predict_batch"), "training")
    seeds = tr["seeds"]
    if not isinstance(seeds, list) or not 2 <= len(seeds) <= 10:
        raise ConfigError("training.seeds must be a list of 2 to 10 distinct integers")
    for s in seeds:
        _seed(s, "training.seeds[]")
    if len(set(seeds)) != len(seeds):
        raise ConfigError("training.seeds must be distinct")
    _int_range(tr, "tune_seeds", "training", 1, len(seeds))
    _int_range(tr, "max_epochs", "training", 1, 5000)
    _int_range(tr, "patience", "training", 1, 500)
    _num_range(tr, "min_delta", "training", 0.0, 0.01)
    if _get(tr, "grad_clip_norm", "training") is not None:
        _num_range(tr, "grad_clip_norm", "training", 0.1, 10.0)
    adam = _get(tr, "adam", "training")
    _exact_keys(adam, ("beta1", "beta2", "eps", "weight_decay"), "training.adam")
    _num_range(adam, "beta1", "training.adam", 0.0, 1.0, lo_open=True, hi_open=True)
    _num_range(adam, "beta2", "training.adam", 0.0, 1.0, lo_open=True, hi_open=True)
    _num_range(adam, "eps", "training.adam", 0.0, None, lo_open=True)
    _num_range(adam, "weight_decay", "training.adam", 0.0, 1.0e-2)
    if _str(tr, "dtype", "training") not in ("float32", "float64"):
        raise ConfigError("training.dtype must be float32 or float64")
    _int_range(tr, "torch_threads", "training", 1, 16)
    _int_range(tr, "n_jobs", "training", 1, 16)
    _int_range(tr, "predict_batch", "training", 256, 65536)

    tu = _get(cfg, "tuning", "")
    _exact_keys(tu, ("n_trials", "sampler", "space"), "tuning")
    n_trials = _int_range(tu, "n_trials", "tuning", 1, 30)
    if n_trials % len(opts):
        raise ConfigError(f"tuning.n_trials = {n_trials} is not divisible by the {len(opts)} input options")
    if tu["sampler"] != "lhs":
        raise ConfigError(f"tuning.sampler must be 'lhs' for {model}")
    space = _get(tu, "space", "tuning")
    if model == "lstm" and isinstance(space, dict) and "cell" in space:
        raise ConfigError("tuning.space.cell is not allowed for lstm (the cell is always nn.LSTM)")
    dims = NEURAL_DIMS + (("cell",) if model == "rnn" else ())
    _exact_keys(space, dims, "tuning.space")
    w = "tuning.space"
    _ints_in(_choice_values(_dim(space, "num_layers", w), f"{w}.num_layers"), f"{w}.num_layers", 1, 2)
    _ints_in(_choice_values(_dim(space, "hidden_size", w), f"{w}.hidden_size"), f"{w}.hidden_size", 16, 64)
    lr = _dim(space, "lr", w, ("loguniform", "choice"))
    if lr["type"] == "loguniform":
        lo = _num_range(lr, "low", f"{w}.lr", 1.0e-5, 1.0e-1)
        hi = _num_range(lr, "high", f"{w}.lr", 1.0e-5, 1.0e-1)
        if not lo < hi:
            raise ConfigError(f"{w}.lr: low must be < high")
    else:
        for v in lr["values"]:
            if not _is_num(v) or not 1.0e-5 <= v <= 1.0e-1:
                raise ConfigError(f"{w}.lr: value {v!r} outside [1e-5, 1e-1]")
    _ints_in(_choice_values(_dim(space, "batch_size", w), f"{w}.batch_size"), f"{w}.batch_size", 1, None)
    _subset(_choice_values(_dim(space, "fluid_weights", w), f"{w}.fluid_weights"), weight_names,
            f"{w}.fluid_weights")
    if model == "rnn":
        _subset(_choice_values(_dim(space, "cell", w), f"{w}.cell"), ("rnn", "gru"), f"{w}.cell")


def _validate_dmd(cfg: dict, weight_names) -> None:
    d = _get(cfg, "dmd", "")
    _exact_keys(d, ("basis", "real_kappa_max", "real_kappa_sep", "energy_report", "unstable", "stab_tol",
                    "imag_tol", "sigma_rel_tol", "dup_tol", "r_max"), "dmd")
    if d["basis"] not in DMD_BASES:
        raise ConfigError(f"dmd.basis must be one of {list(DMD_BASES)}, got {d['basis']!r}")
    _num_range(d, "real_kappa_max", "dmd", 0.5, 10.0)
    _num_range(d, "real_kappa_sep", "dmd", 1.0e-4, 0.05)
    _num_range(d, "energy_report", "dmd", 0.9, 1.0)
    if d["unstable"] not in ("clip", "keep"):
        raise ConfigError("dmd.unstable must be clip or keep")
    _num_range(d, "stab_tol", "dmd", 0.0, 1.0e-6)
    _num_range(d, "imag_tol", "dmd", 0.0, 1.0e-6)
    _num_range(d, "sigma_rel_tol", "dmd", 1.0e-14, 1.0e-6)
    _num_range(d, "dup_tol", "dmd", 0.0, 1.0e-4)
    if d["r_max"] is not None:
        _int_range(d, "r_max", "dmd", 1, None)
    g = _get(cfg, "gp", "")
    _exact_keys(g, ("alpha", "normalize_y", "n_restarts_optimizer", "constant_bounds", "length_scale_bounds",
                    "noise_init", "noise_bounds", "std_includes_basis_residual"), "gp")
    _num_range(g, "alpha", "gp", 1.0e-12, 1.0e-2)
    _bool(g, "normalize_y", "gp")
    _int_range(g, "n_restarts_optimizer", "gp", 0, 20)
    _bounds(g, "constant_bounds", "gp")
    _bounds(g, "length_scale_bounds", "gp")
    lo, hi = _bounds(g, "noise_bounds", "gp")
    _num_range(g, "noise_init", "gp", lo, hi)
    _bool(g, "std_includes_basis_residual", "gp")
    tu = _get(cfg, "tuning", "")
    _exact_keys(tu, ("n_trials", "sampler", "space"), "tuning")
    n_trials = _int_range(tu, "n_trials", "tuning", 1, 30)
    if tu["sampler"] != "grid":
        raise ConfigError("tuning.sampler must be 'grid' for dmd")
    space = _get(tu, "space", "tuning")
    _exact_keys(space, DMD_DIMS, "tuning.space")
    w = "tuning.space"
    _ints_in(_choice_values(_dim(space, "rank", w), f"{w}.rank"), f"{w}.rank", *DMD_RANK_RANGE)
    _subset(_choice_values(_dim(space, "kernel", w), f"{w}.kernel"), KERNELS, f"{w}.kernel")
    _subset(_choice_values(_dim(space, "fluid_weights", w), f"{w}.fluid_weights"), weight_names,
            f"{w}.fluid_weights")
    _subset(_choice_values(_dim(space, "gp_target", w), f"{w}.gp_target"), GP_TARGETS, f"{w}.gp_target")
    size = math.prod(len(space[k]["values"]) for k in DMD_DIMS)
    if size > n_trials:
        raise ConfigError(f"tuning: the grid has {size} points, more than n_trials = {n_trials}")


def validate_model_config(cfg: dict) -> dict:
    """Validates a model config (spec task2 section 3). Returns a copy in which
    each fluid weight set is normalised to sum 1. Raises ConfigError."""
    if not isinstance(cfg, dict):
        raise ConfigError("the config must be a mapping")
    model = _get(cfg, "model", "")
    if model not in MODEL_NAMES:
        raise ConfigError(f"model must be one of {list(MODEL_NAMES)}, got {model!r}")
    _exact_keys(cfg, MODEL_KEYS[model], "")
    cfg = copy.deepcopy(cfg)
    _validate_common(cfg, model)
    if model in NEURAL or model == "dmd":
        cfg["fluid_weight_sets"] = _validate_weight_sets(cfg)
        names = tuple(cfg["fluid_weight_sets"])
        if model in NEURAL:
            _validate_neural(cfg, model, names)
        else:
            _validate_dmd(cfg, names)
    if model == "b1":
        knn = _get(cfg, "knn", "")
        if knn != KNN_FIXED or any(isinstance(v, bool) for v in knn.values()):
            raise ConfigError(f"knn must be exactly {KNN_FIXED} (PRD section 6)")
    return cfg


def load_model_config(path: Path) -> dict:
    """Reads configs/models/<model>.yaml with yaml.safe_load and validates it."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigError(f"cannot read config {path}: {e}") from e
    try:
        cfg = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in {path}: {e}") from e
    return validate_model_config(cfg)


# --- Task 3: evaluation config (spec task3 sections 1.2 and 3) ----------------------

PROXY_NAMES = ("rnn", "lstm", "dmd")              # = dwc26.models.registry.PROXIES
BENCHMARK_NAMES = ("b0", "b1")
STRATEGIES = ("S3", "S4")
P2_MODES = ("reject", "report")
SPEC3_T3 = "(spec task3 section 3.1)"
EVAL_KEYS = ("seed", "paths", "candidates", "strategy", "runtime", "consistency_rel_tol", "metrics", "events",
             "mismatch", "continuation", "plausibility", "backtest", "selection", "cost", "freeze", "report")
EVAL_BLOCK_KEYS = {
    "paths": ("db", "models", "reports", "figures", "logs"),
    "candidates": ("runs", "require_clean"),
    "runtime": ("torch_threads",),
    "metrics": ("objective_fluids", "coverage_z", "coverage_target"),
    "events": ("plateau_frac", "breakthrough_frac"),
    "mismatch": ("window", "fluids", "alpha", "beta", "local_n"),
    "continuation": ("end_date",),
    "plausibility": ("p2_mode", "p2_ratio", "p2_case_share", "p3_hist_quarters", "p3_fc_quarters", "p3_ratio_min",
                     "p3_min_decline", "gor_rel_tol", "gor_oil_frac"),
    "backtest": ("cutoff_date", "models", "model_configs", "v6_horizons", "reuse"),
    "selection": ("band", "secondary_only"),
    "cost": ("n_rows", "repeats"),
    "freeze": ("tag", "repo_dir", "decisions_path", "frozen_paths", "min_override_chars"),
    "report": ("plots", "dpi"),
}
# The config blocks that the test scoring must share with the frozen chain (guard G7, spec task3 section 1.10)
G7_BLOCKS = ("metrics", "events", "mismatch", "continuation", "plausibility", "backtest", "selection")


def _rng3(d: dict, key: str, where: str, lo, hi, *, lo_open: bool = False) -> float:
    """Number in [lo, hi] (or (lo, hi] with lo_open); the message names the Task 3 spec."""
    v = _num(d, key, where)
    if not math.isfinite(v) or (v <= lo if lo_open else v < lo) or (hi is not None and v > hi):
        raise ConfigError(f"{where}.{key} = {v} is outside the allowed range "
                          f"{'(' if lo_open else '['}{lo}, {hi}] {SPEC3_T3}")
    return v


def _int3(d: dict, key: str, where: str, lo: int, hi: int | None = None) -> int:
    v = _int(d, key, where)
    if v < lo or (hi is not None and v > hi):
        raise ConfigError(f"{where}.{key} = {v} is outside the allowed range [{lo}, {hi}] {SPEC3_T3}")
    return v


def iso_date(v, where: str) -> str:
    """An ISO date 'YYYY-MM-DD' (a YAML date or a string) -> the string. ConfigError otherwise."""
    import datetime as _dt

    if isinstance(v, _dt.datetime):
        raise ConfigError(f"{where} must be a date 'YYYY-MM-DD', not a date-time")
    if isinstance(v, _dt.date):
        return v.isoformat()
    if not isinstance(v, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
        raise ConfigError(f"{where} must be an ISO date 'YYYY-MM-DD', got {v!r}")
    try:
        _dt.date.fromisoformat(v)
    except ValueError as e:
        raise ConfigError(f"{where}: {v!r} is not a valid date: {e}") from e
    return v


def _int_pair(d: dict, key: str, where: str) -> tuple[int, int]:
    v = _get(d, key, where)
    if not isinstance(v, list) or len(v) != 2 or any(isinstance(x, bool) or not isinstance(x, int) for x in v):
        raise ConfigError(f"{where}.{key} must be a list of two integers [low, high]")
    return int(v[0]), int(v[1])


def _str_list(d: dict, key: str, where: str, *, allow_empty: bool = False) -> list[str]:
    v = _get(d, key, where)
    if not isinstance(v, list) or (not v and not allow_empty) or not all(isinstance(x, str) and x for x in v):
        raise ConfigError(f"{where}.{key} must be a {'' if allow_empty else 'non-empty '}list of strings")
    if len(set(v)) != len(v):
        raise ConfigError(f"{where}.{key} has duplicates")
    return v


def _rel_path(v, where: str) -> str:
    if (not isinstance(v, str) or not v or Path(v).is_absolute() or v.startswith(("/", "\\"))
            or ".." in Path(v).parts):
        raise ConfigError(f"{where} must be a relative path inside the repository, got {v!r}")
    return v


def quarter_months(d0: str, d1: str) -> int | None:
    """Months from d0 to d1 if both ISO dates share the day of month and the month difference is
    a multiple of 3; else None."""
    import datetime as _dt

    a, b = _dt.date.fromisoformat(d0), _dt.date.fromisoformat(d1)
    months = (b.year - a.year) * 12 + (b.month - a.month)
    if a.day != b.day or months % 3:
        return None
    return months


def validate_eval_config(cfg: dict) -> dict:
    """Validates the Task 3 config against spec task3 section 3.1. Returns a copy in which every
    date is an ISO string. The checks that need the data (n of a split, T, k0, H, the last history
    date) run at run time. Raises ConfigError."""
    if not isinstance(cfg, dict):
        raise ConfigError("the config must be a mapping")
    _exact_keys(cfg, EVAL_KEYS, "")
    cfg = copy.deepcopy(cfg)
    for block, keys in EVAL_BLOCK_KEYS.items():
        _exact_keys(cfg[block], keys, block)
    _seed(cfg["seed"], "seed")
    for k in EVAL_BLOCK_KEYS["paths"]:
        _str(cfg["paths"], k, "paths")

    cand = cfg["candidates"]
    runs = cand["runs"]
    if not isinstance(runs, dict) or not runs:
        raise ConfigError("candidates.runs must be a non-empty mapping model -> 'latest' or a run ID")
    bad = [m for m in runs if m not in MODEL_NAMES]
    if bad:
        raise ConfigError(f"candidates.runs: unknown model(s) {bad}; allowed {list(MODEL_NAMES)}")
    for m in BENCHMARK_NAMES:
        if m not in runs:
            raise ConfigError(f"candidates.runs must list {m} (V6 benchmark)")
    proxies = [m for m in runs if m in PROXY_NAMES]
    if not proxies:
        raise ConfigError("candidates.runs must list at least one proxy model (rnn, lstm or dmd)")
    for m, v in runs.items():
        if not isinstance(v, str) or not v.strip() or v != v.strip():
            raise ConfigError(f"candidates.runs.{m} must be 'latest' or a run ID, got {v!r}")
    _bool(cand, "require_clean", "candidates")

    strat = cfg["strategy"]
    if not isinstance(strat, dict):
        raise ConfigError("strategy must be a mapping proxy model -> S3 or S4")
    bad = [m for m in strat if m not in PROXY_NAMES]
    if bad:
        raise ConfigError(f"strategy: {bad} are not proxy models {list(PROXY_NAMES)} {SPEC3_T3}")
    if set(strat) != set(proxies):
        raise ConfigError(f"strategy must have exactly the proxy models of candidates.runs {proxies}, "
                          f"got {list(strat)}")
    for m, s in strat.items():
        if s not in STRATEGIES:
            raise ConfigError(f"strategy.{m} must be one of {list(STRATEGIES)}, got {s!r}")

    _int3(cfg["runtime"], "torch_threads", "runtime", 1, 16)
    _rng3(cfg, "consistency_rel_tol", "", 0.0, 1.0e-4)

    met = cfg["metrics"]
    if met["objective_fluids"] != ["oil", "water"]:
        raise ConfigError(f"metrics.objective_fluids must be exactly [oil, water] (the selection table has one "
                          f"column for each), got {met['objective_fluids']!r} {SPEC3_T3}")
    _rng3(met, "coverage_z", "metrics", 0.0, None, lo_open=True)
    ct = _get(met, "coverage_target", "metrics")
    if (not isinstance(ct, list) or len(ct) != 2 or not all(_is_num(x) for x in ct)
            or not 0 < ct[0] < ct[1] <= 1):
        raise ConfigError(f"metrics.coverage_target must be [lo, hi] with 0 < lo < hi <= 1, got {ct!r}")

    _rng3(cfg["events"], "plateau_frac", "events", 0.5, 1.0)
    _rng3(cfg["events"], "breakthrough_frac", "events", 0.0, 0.5)

    mm = cfg["mismatch"]
    win = _get(mm, "window", "mismatch")
    if not isinstance(win, list) or len(win) != 2:
        raise ConfigError("mismatch.window must be [start, end]")
    mm["window"] = [iso_date(win[0], "mismatch.window[0]"), iso_date(win[1], "mismatch.window[1]")]
    if mm["window"][0] > mm["window"][1]:
        raise ConfigError(f"mismatch.window: start {mm['window'][0]} is after end {mm['window'][1]}")
    fl = _str_list(mm, "fluids", "mismatch")
    _subset(fl, CONFIG_FLUIDS, "mismatch.fluids")
    _rng3(mm, "alpha", "mismatch", 0.01, 0.5)
    _rng3(mm, "beta", "mismatch", 0.001, 0.2)
    _int3(mm, "local_n", "mismatch", 1)

    cont = cfg["continuation"]
    cont["end_date"] = iso_date(cont["end_date"], "continuation.end_date")
    # The match window ends on the last history date (K runs through it), so the end date must be a
    # whole number of quarters after the window end. The run re-checks it against the data.
    q = quarter_months(mm["window"][1], cont["end_date"])
    if q is None or q <= 0:
        raise ConfigError(f"continuation.end_date {cont['end_date']} is not a whole number of quarters after the "
                          f"end of the match window {mm['window'][1]} (the last history date) {SPEC3_T3}")
    n_fc_cfg = q // 3

    pl = cfg["plausibility"]
    if pl["p2_mode"] not in P2_MODES:
        raise ConfigError(f"plausibility.p2_mode must be one of {list(P2_MODES)}, got {pl['p2_mode']!r}")
    _rng3(pl, "p2_ratio", "plausibility", 1.0, 1.5)
    _rng3(pl, "p2_case_share", "plausibility", 0.0, 1.0)
    _int3(pl, "p3_hist_quarters", "plausibility", 1)
    a, b = _int_pair(pl, "p3_fc_quarters", "plausibility")
    if not 1 <= a < b <= n_fc_cfg:
        raise ConfigError(f"plausibility.p3_fc_quarters = [{a}, {b}] must satisfy 1 <= a < b <= n_fc = "
                          f"{n_fc_cfg} {SPEC3_T3}")
    _rng3(pl, "p3_ratio_min", "plausibility", 0.0, 1.0)
    _rng3(pl, "p3_min_decline", "plausibility", 0.0, 0.05)
    _rng3(pl, "gor_rel_tol", "plausibility", 0.0, 0.05)
    _rng3(pl, "gor_oil_frac", "plausibility", 0.0, 0.5)

    bt = cfg["backtest"]
    bt["cutoff_date"] = iso_date(bt["cutoff_date"], "backtest.cutoff_date")
    bm = _str_list(bt, "models", "backtest")
    if set(bm) != set(proxies):
        raise ConfigError(f"backtest.models must be exactly the proxy models of candidates.runs {proxies} "
                          f"(each proxy is backtested, V6), got {bm}")
    mc = bt["model_configs"]
    if not isinstance(mc, dict) or set(mc) != set(bm):
        raise ConfigError(f"backtest.model_configs must have one path for each model of backtest.models {bm}")
    for m, p in mc.items():
        if not isinstance(p, str) or not p:
            raise ConfigError(f"backtest.model_configs.{m} must be a path")
    lo, hi = _int_pair(bt, "v6_horizons", "backtest")
    if not 1 <= lo <= hi:
        raise ConfigError(f"backtest.v6_horizons = [{lo}, {hi}] must satisfy 1 <= lo <= hi {SPEC3_T3}")
    if bt["reuse"] is None:
        bt["reuse"] = {}
    reuse = bt["reuse"]
    if not isinstance(reuse, dict):
        raise ConfigError("backtest.reuse must be a mapping model -> backtest train run ID")
    for m, r in reuse.items():
        if m not in bm:
            raise ConfigError(f"backtest.reuse: {m!r} is not in backtest.models {bm}")
        if not isinstance(r, str) or not r:
            raise ConfigError(f"backtest.reuse.{m} must be a run ID")

    sel = cfg["selection"]
    _rng3(sel, "band", "selection", 0.0, 0.2)
    so = _str_list(sel, "secondary_only", "selection", allow_empty=True)
    _subset(so, PROXY_NAMES, "selection.secondary_only")

    _int3(cfg["cost"], "n_rows", "cost", 1)
    _int3(cfg["cost"], "repeats", "cost", 1, 20)

    fr = cfg["freeze"]
    _str(fr, "tag", "freeze")
    if fr["repo_dir"] is not None and (not isinstance(fr["repo_dir"], str) or not fr["repo_dir"]):
        raise ConfigError("freeze.repo_dir must be null or a path")
    _rel_path(fr["decisions_path"], "freeze.decisions_path")
    for p in _str_list(fr, "frozen_paths", "freeze"):
        _rel_path(p, "freeze.frozen_paths[]")
    _int3(fr, "min_override_chars", "freeze", 10, 200)

    _bool(cfg["report"], "plots", "report")
    _int3(cfg["report"], "dpi", "report", 50, 300)
    return cfg


def load_eval_config(path: Path) -> dict:
    """Reads configs/evaluate.yaml with yaml.safe_load and validates it (spec task3 section 3.1)."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigError(f"cannot read config {path}: {e}") from e
    try:
        cfg = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in {path}: {e}") from e
    return validate_eval_config(cfg)


def check_override_reason(reason: str | None, cfg: dict) -> str | None:
    """None, or the stripped reason if it has at least freeze.min_override_chars characters (D-028)."""
    if reason is None:
        return None
    r = str(reason).strip()
    n = cfg["freeze"]["min_override_chars"]
    if len(r) < n:
        raise ConfigError(f"--override-reason must have at least {n} characters after stripping, got {len(r)}")
    return r
