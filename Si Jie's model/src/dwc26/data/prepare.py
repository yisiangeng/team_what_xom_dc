"""`python -m dwc26 prepare`: Task 1 pipeline (spec task1 section 1.4).

Steps: load -> C1 -> grid -> C2, C3, C4, C8 -> C5, C6 (rate definition) ->
split, align, V9 -> scale -> C7, C9, C10 -> DB (one transaction), npz,
report, run log. A failed stop check writes the checks that ran, the report
(FAILED) and the run row (failed), leaves the Task 1 tables and the npz
untouched, and raises DataCheckError.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np

from dwc26.config import load_config
from dwc26.data import db
from dwc26.data.align import align_all, history_grid
from dwc26.data.check import (
    ContractError,
    DataCheckError,
    assert_array,
    check_c1,
    check_c2,
    check_c3,
    check_c4,
    check_c5,
    check_c6,
    check_c7,
    check_c8,
    check_c9,
    check_c10,
    write_report,
)
from dwc26.data.load import FLUIDS, load_parameters, load_production
from dwc26.data.scale import fit_x_scaler, fit_y_scaler
from dwc26.data.split import assign_split

LOG = logging.getLogger("dwc26.prepare")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def arrays_sha256(arrays: dict[str, np.ndarray]) -> str:
    """sha256 of the bytes of X, Y, hist, dates (int64), t_days, case_id,
    x_lower, x_upper, y_factor, in this order (spec 1.6)."""
    h = hashlib.sha256()
    for k in ("X", "Y", "hist", "dates", "t_days", "case_id", "x_lower", "x_upper", "y_factor"):
        a = arrays[k].astype("int64") if k == "dates" else arrays[k]
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _npz_contract(arrays: dict, n: int, T: int, p: int) -> None:
    """V9 before np.savez: all keys and shapes of spec 1.5; float arrays float64."""
    F = len(FLUIDS)
    spec = {
        "X": ((n, p), np.float64), "Y": ((n, T, F), np.float64), "hist": ((T, F), np.float64),
        "dates": ((T,), np.dtype("datetime64[D]")), "t_days": ((T,), np.float64),
        "case_id": ((n,), np.int64), "split": ((n,), np.dtype("<U5")), "x_names": ((p,), None),
        "fluids": ((F,), None), "x_lower": ((p,), np.float64), "x_upper": ((p,), np.float64),
        "y_factor": ((F,), np.float64), "X_phys": ((n, p), np.float64), "Y_phys": ((n, T, F), np.float64),
        "hist_phys": ((T, F), np.float64), "rate_definition": ((), None), "run_id": ((), None),
    }
    if set(arrays) != set(spec):
        raise ContractError(f"npz keys {sorted(arrays)} differ from {sorted(spec)}")
    for k, (shape, dtype) in spec.items():
        assert_array(f"npz.{k}", arrays[k], shape, dtype=dtype, finite=dtype is not None)
        if dtype is None and arrays[k].dtype.kind != "U":
            raise ContractError(f"npz.{k}: expected a str array, got {arrays[k].dtype}")


LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"


class UTCFormatter(logging.Formatter):
    """Timestamps in UTC, ISO 8601 with milliseconds and 'Z' (like run_id and runs.start)."""

    converter = time.gmtime
    default_time_format = "%Y-%m-%dT%H:%M:%S"
    default_msec_format = "%s.%03dZ"


def _open_log(log_dir: Path, run_id: str) -> tuple[Path, logging.Handler]:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{run_id}.log"
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(UTCFormatter(LOG_FORMAT))
    handler.setLevel(logging.INFO)
    LOG.addHandler(handler)
    LOG.setLevel(logging.INFO)
    return path, handler


def run_prepare(config_path: Path) -> str:
    """Runs Task 1 and returns the run_id. Raises DataCheckError on a failed stop check."""
    config_path = Path(config_path)
    cfg = load_config(config_path)
    paths = {k: Path(v) for k, v in cfg["paths"].items()}
    con = db.connect(paths["db"])
    try:
        db.init_schema(con)
        run_id = db.start_run(con, task="task1", command="prepare", cfg=cfg,
                              config_path=config_path, seed=cfg["seed"])
        log_path, handler = _open_log(paths["logs"], run_id)
        try:
            return _run(con, run_id, cfg, config_path, paths, log_path)
        except DataCheckError:
            raise
        except BaseException as e:
            LOG.exception("run %s failed: %r", run_id, e)
            db.finish_run(con, run_id, "failed",
                          {"error": f"{type(e).__name__}: {e}",
                           "files": {"log": log_path.as_posix()}})
            raise
        finally:
            LOG.removeHandler(handler)
            handler.close()
    finally:
        con.close()


def _run(con, run_id: str, cfg: dict, config_path: Path, paths: dict, log_path: Path) -> str:
    row = con.execute("SELECT git_commit, git_dirty, start, config_sha256 FROM runs WHERE run_id = ?",
                      (run_id,)).fetchone()
    inputs = {paths[k].as_posix(): sha256_file(paths[k]) for k in ("file01", "file02")}
    meta = {
        "run_id": run_id, "status": "running", "git_commit": row[0], "git_dirty": row[1], "start": row[2],
        "config_path": config_path.as_posix(), "config_sha256": row[3], "inputs": inputs,
    }
    LOG.info("run %s: prepare, config %s (sha256 %s)", run_id, config_path.as_posix(), row[3])
    LOG.info("config: %s", json.dumps(cfg, sort_keys=True))
    LOG.info("git commit %s, dirty %s", row[0], row[1])
    for p, h in inputs.items():
        LOG.info("input %s sha256 %s", p, h)

    results = []

    def record(res):
        results.append(res)
        LOG.info("%s %s (%s)", res.check_id, res.status, res.action)
        return res

    def stop(res):
        meta.update(status="failed", stopped_at=res.check_id)
        with con:
            db.write_checks(con, run_id, results)
        write_report(paths["report"], meta, results)
        db.finish_run(con, run_id, "failed", {
            "files": {"db": paths["db"].as_posix(), "report": paths["report"].as_posix(),
                      "log": log_path.as_posix()},
            "inputs": inputs,
            "checks": {r.check_id: r.status for r in results},
            "stopped_at": res.check_id,
        })
        LOG.error("stopped at %s: %s", res.check_id, json.dumps(res.detail)[:2000])
        raise DataCheckError(f"data check {res.check_id} failed ({res.action}); see {paths['report'].as_posix()}",
                             results)

    # 2. load
    params, problems01 = load_parameters(paths["file01"], cfg["file01"])
    raw = load_production(paths["file02"], cfg["file02"])
    LOG.info("loaded %d cases from file 01, %d native cells from file 02", len(params), len(raw.native))

    # 3. C1, grid
    if record(check_c1(params, problems01, raw, cfg)).status == "fail":
        stop(results[-1])
    case_ids = params["case_id"].to_numpy(dtype=np.int64)
    assert_array("load_parameters.case_id", case_ids, (None,), dtype=np.int64)
    if len(np.unique(case_ids)) != len(case_ids):
        raise ContractError("load_parameters: case_id not unique")
    names = list(cfg["file01"]["columns"].values())
    X_phys = params[names].to_numpy(dtype=np.float64)
    assert_array("load_parameters.X", X_phys, (len(case_ids), len(names)))
    grid = history_grid(raw)
    n, T = len(case_ids), len(grid)
    meta.update(n=n, T=T, first_date=str(grid[0].astype("datetime64[D]")),
                last_date=str(grid[-1].astype("datetime64[D]")))
    LOG.info("history grid: T = %d, %s to %s", T, meta["first_date"], meta["last_date"])

    # 4. stop checks
    for fn in (lambda: check_c2(raw, grid, case_ids), lambda: check_c3(raw),
               lambda: check_c4(raw, grid, cfg), lambda: check_c8(raw, grid, case_ids, cfg)):
        if record(fn()).status == "fail":
            stop(results[-1])

    # 5. C5 (report), C6 (rate definition)
    record(check_c5(raw, grid, case_ids))
    c6, definition = check_c6(raw, grid, cfg)
    record(c6)
    meta.update(rate_definition=definition, c6_convention=c6.detail["convention"], v4_note=c6.detail["v4_note"])
    if definition is None:
        stop(c6)
    LOG.info("C6: e_next %.3g, e_prev %.3g, convention %s -> %s", c6.detail["e_next"], c6.detail["e_prev"],
             c6.detail["convention"], definition)

    # 6. split, align
    split = assign_split(case_ids, cfg["split"])
    aligned = align_all(raw, grid, case_ids, definition, cfg)
    LOG.info("aligned: cum %s, rate %s, n_interpolated %d", aligned.cum.shape, aligned.rate.shape,
             aligned.n_interpolated)

    # 7. scale
    xs = fit_x_scaler(params, cfg)
    X = xs.transform(X_phys)
    assert_array("X", X, (n, len(names)))
    if (X < 0).any() or (X > 1).any():
        raise ContractError("X outside [0, 1] for the design cases")
    ys = fit_y_scaler(aligned.rate, case_ids, split)
    assert_array("y_factor", ys.factor, (len(FLUIDS),), nonneg=True)
    Y = ys.transform(aligned.rate)
    hist = ys.transform(aligned.hist_rate)
    assert_array("Y", Y, (n, T, len(FLUIDS)), nonneg=True)
    assert_array("hist", hist, (T, len(FLUIDS)), nonneg=True)
    LOG.info("X bounds (%s): lower %s, upper %s", xs.source, xs.lower.tolist(), xs.upper.tolist())
    LOG.info("Y factors %s: %s (fitted on %d train cases)", list(ys.fluids), ys.factor.tolist(), len(ys.fitted_on))

    # 8. report checks
    record(check_c7(aligned))
    record(check_c9(params, xs))
    record(check_c10(aligned, split, ys, cfg))

    # 9. outputs
    arrays = {
        "X": X, "Y": Y, "hist": hist, "dates": aligned.dates, "t_days": aligned.t_days,
        "case_id": case_ids, "split": split, "x_names": db.str_array(xs.names), "fluids": db.str_array(ys.fluids),
        "x_lower": xs.lower, "x_upper": xs.upper, "y_factor": ys.factor, "X_phys": X_phys,
        "Y_phys": aligned.rate, "hist_phys": aligned.hist_rate,
        "rate_definition": np.array(str(definition)), "run_id": np.array(str(run_id)),
    }
    arrays = {k: np.ascontiguousarray(v) if v.ndim else v for k, v in arrays.items()}
    _npz_contract(arrays, n, T, len(names))
    for k in ("X", "Y", "hist", "dates", "case_id", "split"):
        LOG.info("array %s %s %s", k, arrays[k].shape, arrays[k].dtype)

    with con:
        db.write_task1_tables(con, run_id, params, split, raw, aligned, xs, ys)
        db.write_checks(con, run_id, results)
    paths["arrays"].parent.mkdir(parents=True, exist_ok=True)
    with open(paths["arrays"], "wb") as fh:
        np.savez(fh, **arrays)
    meta["status"] = "ok"
    write_report(paths["report"], meta, results)
    outputs = {
        "files": {"db": paths["db"].as_posix(), "arrays": paths["arrays"].as_posix(),
                  "report": paths["report"].as_posix(), "log": log_path.as_posix()},
        "inputs": inputs,
        "arrays_sha256": arrays_sha256(arrays),
        "rate_definition": definition,
        "c6_convention": c6.detail["convention"],
        "checks": {r.check_id: r.status for r in results},
        "n_cases": n,
        "n_dates": T,
    }
    if c6.detail["v4_note"]:
        outputs["v4_note"] = c6.detail["v4_note"]
    db.finish_run(con, run_id, "ok", outputs)
    dur = con.execute("SELECT duration_s FROM runs WHERE run_id = ?", (run_id,)).fetchone()[0]
    LOG.info("outputs: %s", json.dumps(outputs, sort_keys=True))
    LOG.info("run %s ok in %.2f s", run_id, dur)
    return run_id
