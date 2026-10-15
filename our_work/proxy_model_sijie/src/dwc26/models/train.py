"""`python -m dwc26 train` (spec task2 sections 1.4, 1.5, 1.6, 2.13, 2.14).

One call is one run: read the train and validation arrays, tune on the
validation objective J, build the final model of each variant (no refit),
then write the model directory, the predictions, the report, the log and one
`runs` row (V8). Test cases are never loaded (V3).
"""

from __future__ import annotations

import contextlib
import csv
import json
import logging
import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from our_work.proxy_model_sijie.src.dwc26.config import ConfigError, load_model_config
from our_work.proxy_model_sijie.src.dwc26.data import db
from our_work.proxy_model_sijie.src.dwc26.data.check import ContractError
from our_work.proxy_model_sijie.src.dwc26.data.load import FLUIDS
from our_work.proxy_model_sijie.src.dwc26.data.prepare import LOG_FORMAT, UTCFormatter
from our_work.proxy_model_sijie.src.dwc26.evaluation.metrics import nrmse, objective_j
from our_work.proxy_model_sijie.src.dwc26.models.base import TrialFailed, onset_index, read_meta
from our_work.proxy_model_sijie.src.dwc26.models.dataset import Dataset, load_dataset
from our_work.proxy_model_sijie.src.dwc26.models.registry import BENCHMARKS, get_class, load_model
from our_work.proxy_model_sijie.src.dwc26.models.train_report import diagnostics, speed_benchmark, write_train_report
from our_work.proxy_model_sijie.src.dwc26.models.tuning import make_plan, resolve_settings, tuned_values

LOG = logging.getLogger("dwc26.train")
NEURAL = ("rnn", "lstm")
TRIAL_ERRORS = (TrialFailed, np.linalg.LinAlgError, ContractError)


@dataclass
class TrialResult:
    trial: int
    variant: str
    settings: dict                 # tuned values only
    status: str                    # 'ok' | 'failed'
    error: str | None
    j_val: float                   # +inf if failed
    nrmse_val: np.ndarray          # (3,), NaN if failed
    extra: dict                    # neural: best_epoch, n_epochs (per seed); DMD: r, r_eff, n_stabilised, cond_B
    fit_s: float
    members: list | None = None    # neural: MemberResult of the trial
    model: object | None = None    # DMD


@dataclass
class TrainOutcome:
    models: dict                   # variant -> final model
    trials: list
    best_trial: dict               # variant -> trial id
    final_members: dict = field(default_factory=dict)   # neural: variant -> all MemberResult (seed order)


# --- torch runtime and parallel member fits ---------------------------------------------

@contextlib.contextmanager
def torch_runtime(threads: int):
    """torch.set_num_threads(threads) and deterministic algorithms, restored on exit."""
    import torch

    prev_threads, prev_det = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.set_num_threads(prev_threads)
        torch.use_deterministic_algorithms(prev_det)


def _worker_init(threads: int) -> None:
    import torch

    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def _member_task(task: tuple):
    """(settings, seed, X, Y, X_val, Y_val, k0) -> ('ok', MemberResult) or ('failed', error text)."""
    from our_work.proxy_model_sijie.src.dwc26.models import seq

    try:
        return "ok", seq.train_member(*task)
    except TRIAL_ERRORS as e:
        return "failed", f"{type(e).__name__}: {e}"


def run_member_tasks(tasks: list[tuple], n_jobs: int, threads: int) -> list:
    """Runs the tasks in order (n_jobs = 1) or in a spawn process pool; results in task order."""
    if n_jobs <= 1 or len(tasks) <= 1:
        return [_member_task(t) for t in tasks]
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(n_jobs, len(tasks)), mp_context=ctx,
                             initializer=_worker_init, initargs=(threads,)) as ex:
        return list(ex.map(_member_task, tasks))


# --- train_model --------------------------------------------------------------------------

def _settings_plan(cfg: dict, settings: dict[str, dict], variants_allowed: list[str]) -> list[dict]:
    unknown = [v for v in settings if v not in variants_allowed]
    if unknown:
        raise ConfigError(f"settings for unknown variant(s) {unknown}; allowed {variants_allowed}")
    order = [v for v in variants_allowed if v in settings]
    return [{"trial": i, "variant": v, "settings": dict(settings[v])} for i, v in enumerate(order)]


def _train_neural(name: str, cfg: dict, data: Dataset, settings: dict | None) -> TrainOutcome:
    cls = get_class(name)
    tr = cfg["training"]
    seeds = list(tr["seeds"])
    k0, T_fit, p = onset_index(data.Y_fit), data.T_fit, data.p
    if settings is None:
        plan, n_tune = make_plan(cfg), tr["tune_seeds"]
    else:
        plan, n_tune = _settings_plan(cfg, settings, list(cfg["network"]["input_options"])), len(seeds)
    resolved = {t["trial"]: resolve_settings(cfg, t["variant"], t["settings"]) for t in plan}
    fidx = tuple(FLUIDS.index(f) for f in cfg["objective"]["fluids"])
    args = (data.X_fit, data.Y_fit, data.X_val, data.Y_val, k0)
    tasks = [(resolved[t["trial"]], s, *args) for t in plan for s in seeds[:n_tune]]
    LOG.info("%s: %d trials x %d seed(s) = %d member fits, n_jobs %d", name, len(plan), n_tune, len(tasks),
             tr["n_jobs"])
    t0 = time.perf_counter()
    with torch_runtime(tr["torch_threads"]):
        results = run_member_tasks(tasks, tr["n_jobs"], tr["torch_threads"])
    LOG.info("trial fits done in %.1f s", time.perf_counter() - t0)
    trials = []
    pos = 0
    for t in plan:
        res = results[pos:pos + n_tune]
        pos += n_tune
        failed = [r for st, r in res if st != "ok"]
        if failed:
            trials.append(TrialResult(t["trial"], t["variant"], t["settings"], "failed", failed[0], math.inf,
                                      np.full(3, np.nan), {}, 0.0, None, None))
            LOG.warning("trial %d (%s) failed: %s", t["trial"], t["variant"], failed[0])
            continue
        members = [r for _, r in res]
        ens = cls.from_members(t["variant"], resolved[t["trial"]], members, k0, T_fit, p)
        P = ens.predict(data.X_val)
        j = objective_j(data.Y_val, P, fidx)
        extra = {"best_epoch": [m.best_epoch for m in members], "n_epochs": [m.n_epochs for m in members]}
        trials.append(TrialResult(t["trial"], t["variant"], t["settings"], "ok", None, j, nrmse(data.Y_val, P),
                                  extra, float(sum(m.fit_s for m in members)), members, None))
        LOG.info("trial %d (%s): J_val %.6f, best epochs %s", t["trial"], t["variant"], j, extra["best_epoch"])
    best = _select(trials, [t["variant"] for t in plan])
    # final ensembles: the members of the best trial plus the remaining seeds (no refit)
    extra_tasks, owners = [], []
    for v, b in best.items():
        for s in seeds[n_tune:]:
            extra_tasks.append((resolved[b], s, *args))
            owners.append(v)
    with torch_runtime(tr["torch_threads"]):
        extra_res = run_member_tasks(extra_tasks, tr["n_jobs"], tr["torch_threads"])
    models, final_members = {}, {}
    for v, b in best.items():
        more = [r for o, r in zip(owners, extra_res) if o == v]
        bad = [r for st, r in more if st != "ok"]
        if bad:
            raise ContractError(f"variant {v}: a final ensemble member failed: {bad[0]}")
        members = list(trials[b].members) + [r for _, r in more]
        final_members[v] = members
        models[v] = cls.from_members(v, resolved[b], members, k0, T_fit, p)
    return TrainOutcome(models=models, trials=trials, best_trial=best, final_members=final_members)


def _select(trials: list[TrialResult], variants: list[str]) -> dict[str, int]:
    best = {}
    for v in dict.fromkeys(variants):
        ok = [t for t in trials if t.variant == v and t.status == "ok"]
        if not ok:
            raise ContractError(f"every trial of variant {v} failed; see trials.csv and the log")
        best[v] = min(ok, key=lambda t: (t.j_val, t.trial)).trial
    return best


def _train_dmd(cfg: dict, data: Dataset, settings: dict | None) -> TrainOutcome:
    """DMD grid with the plausibility gate P1 (section 2.14, D-020): the lowest J among the
    grid points whose validation continuation clips no value; else the lowest J, flagged."""
    cls = get_class("dmd")
    plan = make_plan(cfg) if settings is None else _settings_plan(cfg, settings, ["main"])
    fidx = tuple(FLUIDS.index(f) for f in cfg["objective"]["fluids"])
    steps = cfg["report"]["forecast_steps"]
    trials = []
    for t in plan:
        s = resolve_settings(cfg, "main", t["settings"])
        t0 = time.perf_counter()
        try:
            m = cls().fit(data.X_fit, data.Y_fit, s)
            P = m.predict(data.X_val)
            j = objective_j(data.Y_val, P, fidx)
            m.forecast(data.X_val, steps)                                   # P1: validation X only (V1)
            n_clip_cont = m.last_n_clipped
            d = m.diagnostics
            extra = {"r": d["r"], "r_eff": d["r_eff"], "n_stabilised": d["n_stabilised"], "cond_B": d["cond_B"],
                     "basis": d["basis"], "n_clip_cont": int(n_clip_cont), "plausible": bool(n_clip_cont == 0)}
            trials.append(TrialResult(t["trial"], "main", t["settings"], "ok", None, j, nrmse(data.Y_val, P), extra,
                                      time.perf_counter() - t0, None, m))
            vp = d.get("varpro") or {}
            LOG.info("trial %d %s: J_val %.6f, basis %s, r %d, r_eff %d, n_stabilised %d, cond(B) %.3g, "
                     "n_clip_cont %d, varpro start %s status %s, %.1f s", t["trial"], t["settings"], j, d["basis"],
                     d["r"], d["r_eff"], d["n_stabilised"], d["cond_B"], n_clip_cont, vp.get("best_start"),
                     vp.get("status"), time.perf_counter() - t0)
        except TRIAL_ERRORS as e:
            err = f"{type(e).__name__}: {e}"
            trials.append(TrialResult(t["trial"], "main", t["settings"], "failed", err, math.inf,
                                      np.full(3, np.nan), {}, time.perf_counter() - t0, None, None))
            LOG.warning("trial %d failed: %s", t["trial"], err)
    ok = [t for t in trials if t.status == "ok"]
    if not ok:
        raise ContractError("every trial of variant main failed; see trials.csv and the log")
    plausible = [t for t in ok if t.extra["plausible"]]
    pool = plausible or ok
    best = min(pool, key=lambda t: (t.j_val, t.trial))
    best.model.diagnostics["plausible_selected"] = bool(plausible)
    best.model.diagnostics["n_plausible_trials"] = len(plausible)
    best.model.diagnostics["n_clip_cont"] = best.extra["n_clip_cont"]
    if plausible:
        LOG.info("P1: %d of %d ok grid points clip nothing in the validation continuation; selected trial %d",
                 len(plausible), len(ok), best.trial)
    else:
        LOG.warning("P1: no plausible grid point; selected the lowest J (trial %d); Task 3 rejects this proxy "
                    "under P1", best.trial)
    return TrainOutcome(models={"main": best.model}, trials=trials, best_trial={"main": best.trial})


def train_model(name: str, cfg: dict, data: Dataset, *, settings: dict[str, dict] | None = None) -> TrainOutcome:
    """settings=None: tune (section 2.14). settings={variant: tuned values}: no tuning,
    one fit for each variant (Task 5 backtest)."""
    if name != cfg["model"]:
        raise ConfigError(f"train_model: name {name!r} differs from cfg['model'] {cfg['model']!r}")
    if name in NEURAL:
        return _train_neural(name, cfg, data, settings)
    if name == "dmd":
        return _train_dmd(cfg, data, settings)
    if name in BENCHMARKS:
        if settings is not None:
            _settings_plan(cfg, settings, ["main"])
        m = get_class(name)().fit(data.X_fit, data.Y_fit, resolve_settings(cfg, "main", {}))
        return TrainOutcome(models={"main": m}, trials=[], best_trial={})
    raise ConfigError(f"unknown model {name!r}")


# --- run_train -------------------------------------------------------------------------------

def _open_log(log_dir: Path, run_id: str) -> tuple[Path, logging.Handler]:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{run_id}.log"
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(UTCFormatter(LOG_FORMAT))
    handler.setLevel(logging.INFO)
    LOG.addHandler(handler)
    LOG.setLevel(logging.INFO)
    return path, handler


def _settings_from(con, cfg: dict, name: str, src: str, models_dir: Path) -> dict[str, dict]:
    """Section 2.13: the tuned values of each variant of an earlier ok run of the same model."""
    row = con.execute("SELECT model, status, outputs FROM runs WHERE run_id = ?", (src,)).fetchone()
    if row is None:
        raise ConfigError(f"--settings-from {src}: no such run")
    if row[0] != name:
        raise ConfigError(f"--settings-from {src}: run of model {row[0]!r}, not {name!r}")
    if row[1] != "ok":
        raise ConfigError(f"--settings-from {src}: run status {row[1]!r}, not 'ok'")
    variants = list(json.loads(row[2] or "{}").get("variants", {}))
    if not variants:
        raise ConfigError(f"--settings-from {src}: the run lists no variants")
    out = {}
    for v in variants:
        meta = read_meta(models_dir / src / v)
        if meta.get("model") != name:
            raise ConfigError(f"--settings-from {src}/{v}: meta.json is for model {meta.get('model')!r}")
        out[v] = tuned_values(cfg, meta["settings"])
    return out


def _c7_range(con, prepare_run_id: str) -> tuple | None:
    row = con.execute("SELECT detail FROM data_checks WHERE run_id = ? AND check_id = 'C7'",
                      (prepare_run_id,)).fetchone()
    if row is None:
        return None
    try:
        d = json.loads(row[0])
        lo, hi = d["gor_min"]["value"], d["gor_max"]["value"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    return (float(lo), float(hi)) if lo is not None and hi is not None else None


def _benchmarks(con, data: Dataset) -> dict:
    """Latest ok b0 and b1 runs with the same arrays_sha256 and t_cut (section 1.7, item 5)."""
    out = {}
    for b in ("b0", "b1"):
        rows = con.execute("SELECT run_id, outputs FROM runs WHERE task = 'task2' AND command = 'train' "
                           "AND model = ? AND status = 'ok' ORDER BY start DESC", (b,)).fetchall()
        for run_id, outputs in rows:
            o = json.loads(outputs or "{}")
            if o.get("arrays_sha256") == data.arrays_sha256 and o.get("t_cut") == data.t_cut:
                v = o.get("variants", {}).get("main", {})
                if "rmse_val" in v:
                    out[b] = {"run_id": run_id, "rmse_val": v["rmse_val"]}
                    break
    return out


def _write_trials_csv(path: Path, cfg: dict, name: str, trials: list[TrialResult]) -> None:
    dims = list(cfg.get("tuning", {}).get("space", {}))
    if name == "dmd":
        extra = ["r", "r_eff", "n_stabilised", "cond_B", "basis", "n_clip_cont", "plausible"]
    elif name in NEURAL:
        extra = ["best_epoch", "n_epochs"]
    else:
        extra = []
    cols = ["trial", "variant", "status", "error", *dims, "j_val", "nrmse_val_oil", "nrmse_val_gas",
            "nrmse_val_water", "fit_s", *extra]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for t in trials:
            ex = []
            for k in extra:
                v = t.extra.get(k)
                if isinstance(v, list):
                    ex.append(";".join(str(x) for x in v))
                elif isinstance(v, str):
                    ex.append(v)
                else:
                    ex.append("" if v is None else repr(v))
            w.writerow([t.trial, t.variant, t.status, t.error or "", *[t.settings.get(d) for d in dims],
                        repr(float(t.j_val)), *[repr(float(x)) for x in t.nrmse_val], repr(float(t.fit_s)), *ex])


def _write_history(path: Path, member) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "train_loss", "j_val"])
        for e, loss, j in member.history:
            w.writerow([int(e), repr(float(loss)), repr(float(j))])


def _seed_text(name: str, cfg: dict) -> str:
    if name in NEURAL:
        return f"members {cfg['training']['seeds']} (plan seed {cfg['seed']})"
    if name == "dmd":
        return f"GP optimiser random_state {cfg['seed']} (no ensemble)"
    return f"n/a (no randomness; seed {cfg['seed']} logged)"


def run_train(config_path: Path, model: str, *, t_cut: int | None = None,
              settings_from: str | None = None) -> str:
    """Runs one training run and returns its run_id (spec section 1.4)."""
    config_path = Path(config_path)
    cfg = load_model_config(config_path)                                     # 1. config
    if model != cfg["model"]:
        raise ConfigError(f"--model {model!r} differs from model {cfg['model']!r} in {config_path}")
    if t_cut is not None:
        if isinstance(t_cut, bool) or not isinstance(t_cut, int) or t_cut < 3:
            raise ConfigError(f"--t-cut must be an integer >= 3, got {t_cut!r}")
        cfg["data"]["t_cut"] = t_cut
    if model in NEURAL:
        import torch

        torch.set_num_threads(cfg["training"]["torch_threads"])
        torch.use_deterministic_algorithms(True)
    paths = {k: Path(v) for k, v in cfg["paths"].items()}
    con = db.connect(paths["db"])                                            # 2. run row and log
    try:
        db.init_schema(con)
        run_id = db.start_run(con, task="task2", command="train", cfg=cfg, config_path=config_path,
                              seed=cfg["seed"], model=model)
        log_path, handler = _open_log(paths["logs"], run_id)
        try:
            return _run(con, run_id, cfg, model, config_path, paths, log_path, settings_from)
        except BaseException as e:
            LOG.exception("run %s failed: %r", run_id, e)
            con.rollback()
            db.finish_run(con, run_id, "failed", {"error": str(e), "error_type": type(e).__name__,
                                                  "files": {"log": log_path.as_posix()}})
            raise
        finally:
            LOG.removeHandler(handler)
            handler.close()
    finally:
        con.close()


def _run(con, run_id: str, cfg: dict, name: str, config_path: Path, paths: dict, log_path: Path,
         settings_from: str | None) -> str:
    t_start = time.perf_counter()
    row = con.execute("SELECT git_commit, git_dirty, start, config_sha256 FROM runs WHERE run_id = ?",
                      (run_id,)).fetchone()
    LOG.info("run %s: train %s, config %s (sha256 %s)", run_id, name, config_path.as_posix(), row[3])
    LOG.info("config: %s", json.dumps(cfg, sort_keys=True))
    LOG.info("git commit %s, dirty %s; settings_from %s", row[0], row[1], settings_from)
    t_cut = cfg["data"]["t_cut"]
    data = load_dataset(con, t_cut)                                          # 3. data
    k0 = onset_index(data.Y_fit)
    LOG.info("data: n_fit %d, n_val %d, T_fit %d (T %d, t_cut %s), k0 %d, prepare run %s, arrays_sha256 %s",
             len(data.case_fit), len(data.case_val), data.T_fit, data.T_full, t_cut, k0, data.prepare_run_id,
             data.arrays_sha256)
    models_dir = paths["models"]
    settings = None                                                          # 4. settings
    if settings_from is not None:
        settings = _settings_from(con, cfg, name, settings_from, models_dir)
        LOG.info("settings from %s: %s", settings_from, json.dumps(settings, sort_keys=True))
    outcome = train_model(name, cfg, data, settings=settings)                # 5. trials and final models
    is_proxy = name not in BENCHMARKS
    c7 = _c7_range(con, data.prepare_run_id)
    s = data.y_factor
    variants, diags = {}, {}
    for v, m in outcome.models.items():                                      # 6. diagnostics
        d = diagnostics(m, data, cfg, c7_range=c7)
        sp = speed_benchmark(m, cfg) if is_proxy else None
        diags[v] = d
        b = outcome.best_trial.get(v)
        vd = {
            "proxy_id": f"{run_id}/{v}", "best_trial": b, "settings": m.settings,
            "j_val_trial": outcome.trials[b].j_val if b is not None else None, "j_val": d["j_val"],
            "rmse_train": d["rmse_train"], "rmse_val": d["rmse_val"], "nrmse_train": d["nrmse_train"],
            "nrmse_val": d["nrmse_val"], "n_clipped": d["n_clipped"], "coverage_val": d["coverage_val"],
            "speed": sp,
        }
        if name == "dmd":
            dd = m.diagnostics
            vd["dmd"] = {k: dd[k] for k in ("q", "r", "r_eff", "n_stabilised", "n_merged", "max_abs_lam_raw",
                                            "cond_B", "basis", "rank", "r_99", "lam_used", "plausible_selected")}
            vd["dmd"]["varpro_cost"] = (dd.get("varpro") or {}).get("cost")
        variants[v] = vd
        LOG.info("%s: J_val %.6f (trial %s), RMSE val oil %.2f water %.2f, clipped %s, speed %s", vd["proxy_id"],
                 d["j_val"], vd["j_val_trial"], d["rmse_val"]["oil"], d["rmse_val"]["water"], d["n_clipped"],
                 None if sp is None else {k: sp[k] for k in ("per_call_s", "mcmc_extrapolated_s", "pass")})
    run_dir = models_dir / run_id                                            # 7. save, reload, compare
    steps = cfg["report"]["forecast_steps"]
    for v, m in outcome.models.items():
        m.set_context(run_id=run_id, proxy_id=f"{run_id}/{v}", t_cut=t_cut, settings_from=settings_from,
                      x_names=list(data.x_names), fluids=list(data.fluids), y_factor=data.y_factor,
                      x_lower=data.x_lower, x_upper=data.x_upper, prepare_run_id=data.prepare_run_id,
                      arrays_sha256=data.arrays_sha256)
        m.save(run_dir / v)
        loaded = load_model(f"{run_id}/{v}", models_dir, con=con, proxies_only=False)
        checks = [("predict", lambda mm: mm.predict(data.X_val)), ("predict_std", lambda mm: mm.predict_std(data.X_val))]
        if is_proxy:
            checks.append(("rollout", lambda mm: mm.rollout(data.X_val, data.T_fit + steps)))
        for what, fn in checks:
            a, b = fn(m), fn(loaded)
            same = (a is None and b is None) or (a is not None and b is not None and np.array_equal(a, b))
            if not same:
                raise ContractError(f"{run_id}/{v}: {what} of the reloaded model differs from the fitted model")
        variants[v]["files"] = read_meta(run_dir / v)["files"]
        if hasattr(m, "members_info"):
            variants[v]["members"] = m.members_info
        if name == "dmd":
            variants[v]["dmd_full"] = m.diagnostics
        variants[v]["plateau"] = diags[v]["plateau"]
        variants[v]["continuation"] = diags[v]["continuation"]
        variants[v]["gor"] = diags[v]["gor"]
        variants[v]["std_over_rmse_val"] = diags[v]["std_over_rmse_val"]
    run_dir.mkdir(parents=True, exist_ok=True)                               # 8. files
    files = {"model_dir": run_dir.as_posix(), "report": (run_dir / "report.md").as_posix(),
             "log": log_path.as_posix(), "trials": (run_dir / "trials.csv").as_posix()}
    logged = dict(cfg, cli={"t_cut": t_cut, "settings_from": settings_from})
    (run_dir / "config.yaml").write_text(yaml.safe_dump(logged, sort_keys=False), encoding="utf-8")
    _write_trials_csv(run_dir / "trials.csv", cfg, name, outcome.trials)
    if name in NEURAL:
        hdir = run_dir / "histories"
        hdir.mkdir(exist_ok=True)
        written = set()
        for t in outcome.trials:
            for mem in t.members or []:
                _write_history(hdir / f"trial_{t.trial}_seed_{mem.seed}.csv", mem)
                written.add((t.trial, mem.seed))
        for v, mems in outcome.final_members.items():
            b = outcome.best_trial[v]
            for mem in mems:
                if (b, mem.seed) not in written:
                    _write_history(hdir / f"trial_{b}_seed_{mem.seed}.csv", mem)
    tr = cfg.get("training", {})
    header = {
        "run_id": run_id, "model": name, "git_commit": row[0], "git_dirty": row[1], "start": row[2],
        "config_path": config_path.as_posix(), "config_sha256": row[3], "prepare_run_id": data.prepare_run_id,
        "arrays_sha256": data.arrays_sha256, "n_fit": len(data.case_fit), "n_val": len(data.case_val),
        "T_fit": data.T_fit, "k0": k0, "t_cut": t_cut, "settings_from": settings_from,
        "seeds": _seed_text(name, cfg), "tune_seeds": tr.get("tune_seeds", "n/a"),
        "torch_threads": tr.get("torch_threads", "n/a"), "n_jobs": tr.get("n_jobs", "n/a"),
        "duration_s": time.perf_counter() - t_start,
        "n_val_pre_onset_positive": next(iter(diags.values()))["n_val_pre_onset_positive"],
        "dt_min": float(np.diff(data.t_days).min()), "dt_max": float(np.diff(data.t_days).max()),
    }
    plan_counts = {}
    for t in outcome.trials:
        plan_counts[t.variant] = plan_counts.get(t.variant, 0) + 1
    space = cfg.get("tuning", {}).get("space", {}) if settings_from is None else {}
    write_train_report(run_dir / "report.md", {
        "header": header, "space": space, "plan_counts": plan_counts or {v: 0 for v in outcome.models},
        "trials": outcome.trials, "variants": variants, "benchmarks": _benchmarks(con, data),
        "coverage_z": cfg["report"]["coverage_z"]})
    n_rows = 0                                                               # 9. database, one transaction
    case_id = np.concatenate([data.case_fit, data.case_val])
    split = np.array(["train"] * len(data.case_fit) + ["val"] * len(data.case_val))
    for v, d in diags.items():
        P = np.concatenate([d["P_fit"], d["P_val"]]) * s
        S = None if d["S_fit"] is None else np.concatenate([d["S_fit"], d["S_val"]]) * s
        n_rows += db.write_predictions(con, run_id, v, case_id, split, data.dates, P, S)
    out_variants = {v: {k: vd[k] for k in ("proxy_id", "best_trial", "settings", "j_val_trial", "j_val",
                                           "rmse_train", "rmse_val", "nrmse_train", "nrmse_val", "n_clipped",
                                           "coverage_val", "speed", "files") + (("dmd",) if name == "dmd" else ())}
                    for v, vd in variants.items()}
    outputs = {
        "files": files, "prepare_run_id": data.prepare_run_id, "arrays_sha256": data.arrays_sha256,
        "t_cut": t_cut, "T_fit": data.T_fit, "k0": k0, "n_fit": len(data.case_fit), "n_val": len(data.case_val),
        "settings_from": settings_from, "n_trials": len(outcome.trials),
        "n_failed_trials": sum(t.status != "ok" for t in outcome.trials),
        "torch_threads": tr.get("torch_threads"), "n_jobs": tr.get("n_jobs"),
        "variants": out_variants,
        "best_variant_by_val": min(variants, key=lambda v: (variants[v]["j_val"], v)),
        "n_prediction_rows": n_rows,
    }
    db.finish_run(con, run_id, "ok", outputs)
    dur = con.execute("SELECT duration_s FROM runs WHERE run_id = ?", (run_id,)).fetchone()[0]
    LOG.info("predictions: %d rows; best variant by J_val: %s", n_rows, outputs["best_variant_by_val"])
    LOG.info("run %s ok in %.1f s", run_id, dur)
    return run_id
