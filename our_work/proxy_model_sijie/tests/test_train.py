"""M17, M24, M30 to M36, M45: dwc26.models.train, train_report and the `train` command."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from dwc26.cli import main
from dwc26.config import ConfigError, load_model_config
from dwc26.data.db import connect, read_task1_arrays
from dwc26.models import seq
from dwc26.models.base import ModelBase, TrialFailed
from dwc26.models.dataset import load_dataset
from dwc26.models.train import run_train, train_model
from dwc26.models.train_report import PERIODS, diagnostics, plateau_periods
from our_work.proxy_model_sijie.tests.synth_models import ALL_MODELS, K0_DB, T_DB, TEST_IDS, make_curves, synthetic_task1_db, tiny_cfg, unit_X

N_VARIANTS = {"rnn": 3, "lstm": 3, "dmd": 1, "b0": 1, "b1": 1}
N_TRIALS = {"rnn": 3, "lstm": 3, "dmd": 2, "b0": 0, "b1": 0}
OUTPUT_KEYS = {"files", "prepare_run_id", "arrays_sha256", "t_cut", "T_fit", "k0", "n_fit", "n_val",
               "settings_from", "n_trials", "n_failed_trials", "torch_threads", "n_jobs", "variants",
               "best_variant_by_val"}
VARIANT_KEYS = {"proxy_id", "best_trial", "settings", "j_val_trial", "j_val", "rmse_train", "rmse_val",
                "nrmse_train", "nrmse_val", "n_clipped", "coverage_val", "speed", "files"}
REPORT_SECTIONS = ["## 1. Header", "## 2. Search space and plan", "## 3. Trials", "## 4. Variants",
                   "## 5. Benchmarks", "## 6. Clipping", "## 7. Uncertainty preview",
                   "## 8. Continuation check", "## 9. GOR check", "## 10. Plateau and breakthrough",
                   "## 11. DMD", "## 12. Speed"]


@pytest.fixture
def sdb(tmp_path):
    return synthetic_task1_db(tmp_path)


def _train(tmp_path, model, *extra, **overrides) -> tuple[int, str]:
    cfg = tiny_cfg(tmp_path, model, **overrides)
    code = main(["train", "--model", model, "--config", str(cfg), *extra])
    return code, cfg


def _last_run(db_path, model=None):
    con = sqlite3.connect(db_path)
    q = "SELECT run_id, status, outputs, task, command, model, duration_s, packages FROM runs WHERE task = 'task2'"
    rows = con.execute(q + (" AND model = ?" if model else "") + " ORDER BY start", (model,) if model else ()).fetchall()
    con.close()
    return rows[-1]


def _preds(db_path, run_id):
    con = sqlite3.connect(db_path)
    rows = con.execute("SELECT variant, case_id, split, date_idx, oil, gas, water, oil_std, gas_std, water_std "
                       "FROM predictions WHERE run_id = ? ORDER BY variant, case_id, date_idx", (run_id,)).fetchall()
    con.close()
    return rows


# --- M30: end to end -----------------------------------------------------------------------

@pytest.mark.parametrize("model", ALL_MODELS)
def test_m30_end_to_end(sdb, tmp_path, model, capsys):
    code, _ = _train(tmp_path, model)
    assert code == 0
    run_id = capsys.readouterr().out.strip().splitlines()[-1]
    rid, status, outputs, task, command, mdl, dur, packages = _last_run(sdb["db"])
    assert (rid, status, task, command, mdl) == (run_id, "ok", "task2", "train", model)
    assert dur > 0
    pk = json.loads(packages)
    assert pk["torch"] != "not installed" and pk["scikit-learn"] == "1.7.2"
    out = json.loads(outputs)
    assert OUTPUT_KEYS <= set(out)
    assert out["T_fit"] == T_DB and out["k0"] == K0_DB and out["n_fit"] == 18 and out["n_val"] == 6
    assert len(out["variants"]) == N_VARIANTS[model]
    for v, vd in out["variants"].items():
        assert VARIANT_KEYS <= set(vd), set(VARIANT_KEYS) - set(vd)
        assert vd["proxy_id"] == f"{run_id}/{v}"
        assert set(vd["n_clipped"]) == {"train", "val", "forecast_val"}
        if model in ("b0", "b1"):
            assert vd["coverage_val"] is None and vd["speed"] is None
        else:
            assert set(vd["coverage_val"]) == {"oil", "gas", "water"}
            assert {"predict_n_batch_rows_s", "per_call_s", "mcmc_extrapolated_s", "pass"} <= set(vd["speed"])
        if model == "dmd":
            assert {"q", "r", "r_eff", "n_stabilised", "n_merged", "max_abs_lam_raw", "cond_B", "basis", "rank",
                    "r_99", "lam_used", "varpro_cost", "plausible_selected"} <= set(vd["dmd"])
            assert vd["dmd"]["basis"] == "real" and all(im == 0 and 0 < re <= 1 for re, im in vd["dmd"]["lam_used"])
    run_dir = tmp_path / "models" / run_id
    for f in ("config.yaml", "trials.csv", "report.md"):
        assert (run_dir / f).is_file(), f
    for v in out["variants"]:
        assert (run_dir / v / "meta.json").is_file()
    assert (tmp_path / "logs" / f"{run_id}.log").is_file()
    if model in ("rnn", "lstm"):
        assert len(list((run_dir / "histories").glob("trial_*_seed_*.csv"))) == 3 + 3
    rows = _preds(sdb["db"], run_id)
    assert len(rows) == (18 + 6) * T_DB * N_VARIANTS[model]
    assert {r[2] for r in rows} == {"train", "val"}
    assert all(min(r[4:7]) >= 0 for r in rows)
    if model in ("b0", "b1"):
        assert all(r[7] is None and r[8] is None and r[9] is None for r in rows)
    else:
        assert all(r[7] is not None and r[9] is not None for r in rows)
    with open(run_dir / "trials.csv", encoding="utf-8") as fh:
        trows = list(csv.DictReader(fh))
    assert len(trows) == N_TRIALS[model]
    if model == "dmd":
        assert all({"basis", "n_clip_cont", "plausible"} <= set(r) for r in trows)
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    for sec in REPORT_SECTIONS:
        assert sec in report, sec


# --- M31: V3 spy -----------------------------------------------------------------------------

@pytest.mark.parametrize("model", ["lstm", "dmd"])
def test_m31_no_test_case_reaches_a_model(sdb, tmp_path, model, monkeypatch):
    seen = []

    def wrap(fn, pos):
        def inner(*a, **kw):
            seen.append(np.asarray(a[pos], dtype=np.float64).copy())
            return fn(*a, **kw)
        return inner

    for meth in ("fit", "predict", "predict_std", "rollout", "forecast"):
        monkeypatch.setattr(ModelBase, meth, wrap(getattr(ModelBase, meth), 1))
    orig_tm = seq.train_member

    def tm(settings, seed, X, Y, Xv, Yv, k0):
        seen.append(np.asarray(X).copy())
        seen.append(np.asarray(Xv).copy())
        return orig_tm(settings, seed, X, Y, Xv, Yv, k0)

    monkeypatch.setattr(seq, "train_member", tm)
    code, _ = _train(tmp_path, model)
    assert code == 0
    con = connect(sdb["db"])
    a = read_task1_arrays(con)
    con.close()
    X_test = a["X"][a["split"] == "test"]
    assert len(X_test) == len(TEST_IDS) and len(seen) > 5
    for X in seen:
        for x in X_test:
            assert not np.isclose(X, x, rtol=0, atol=1e-12).all(axis=1).any()
    run_id = _last_run(sdb["db"])[0]
    assert not {r[1] for r in _preds(sdb["db"], run_id)} & set(TEST_IDS)


# --- M32: determinism -------------------------------------------------------------------------

def _trials_rows(path: Path):
    with open(path, encoding="utf-8") as fh:
        return [{k: v for k, v in r.items() if k != "fit_s"} for r in csv.DictReader(fh)]


@pytest.mark.parametrize("model", ["lstm", "dmd", "b1"])
def test_m32_run_determinism(sdb, tmp_path, model):
    cfg = tiny_cfg(tmp_path, model)
    r1 = run_train(cfg, model)
    r2 = run_train(cfg, model)
    assert r1 != r2
    p1 = [r[:2] + r[3:] for r in _preds(sdb["db"], r1)]
    p2 = [r[:2] + r[3:] for r in _preds(sdb["db"], r2)]
    assert p1 == p2 and len(p1) > 0
    assert _trials_rows(tmp_path / "models" / r1 / "trials.csv") == _trials_rows(tmp_path / "models" / r2 / "trials.csv")


# --- M33: --settings-from and --t-cut ------------------------------------------------------------

@pytest.mark.parametrize("model", ["lstm", "dmd"])
def test_m33_settings_from_and_t_cut(sdb, tmp_path, model):
    cfg = tiny_cfg(tmp_path, model)
    r1 = run_train(cfg, model)
    r2 = run_train(cfg, model, t_cut=9, settings_from=r1)
    con = sqlite3.connect(sdb["db"])
    o1 = json.loads(con.execute("SELECT outputs FROM runs WHERE run_id = ?", (r1,)).fetchone()[0])
    o2 = json.loads(con.execute("SELECT outputs FROM runs WHERE run_id = ?", (r2,)).fetchone()[0])
    assert o2["T_fit"] == 9 and o2["t_cut"] == 9 and o2["settings_from"] == r1
    assert set(o2["variants"]) == set(o1["variants"])
    cfgd = load_model_config(cfg)
    dims = list(cfgd["tuning"]["space"])
    for v in o1["variants"]:
        s1, s2 = o1["variants"][v]["settings"], o2["variants"][v]["settings"]
        for d in dims:
            key = "fluid_weights_name" if d == "fluid_weights" else d
            assert s1[key] == s2[key], (v, d)
        meta = json.loads((tmp_path / "models" / r2 / v / "meta.json").read_text(encoding="utf-8"))
        assert meta["t_cut"] == 9 and meta["settings_from"] == r1 and meta["T_fit"] == 9
    with open(tmp_path / "models" / r2 / "trials.csv", encoding="utf-8") as fh:
        assert len(list(csv.DictReader(fh))) == len(o1["variants"])
    assert max(r[0] for r in con.execute("SELECT date_idx FROM predictions WHERE run_id = ?", (r2,))) == 8
    con.close()
    other = "b0" if model != "b0" else "b1"
    with pytest.raises(ConfigError):
        run_train(tiny_cfg(tmp_path, other), other, settings_from=r1)


# --- M34: failure path -----------------------------------------------------------------------------

def test_m34_failure_path(sdb, tmp_path, capsys):
    con = sqlite3.connect(sdb["db"])
    con.execute("UPDATE production_grid SET oil_rate = oil_rate * 1.01 WHERE case_id = 3 AND date_idx = 5")
    con.commit()
    con.close()
    code, _ = _train(tmp_path, "b0")
    assert code == 1
    assert "arrays_sha256" in capsys.readouterr().err
    rid, status, outputs = _last_run(sdb["db"])[:3]
    assert status == "failed" and "arrays_sha256" in json.loads(outputs)["error"]
    assert _preds(sdb["db"], rid) == []


def test_m35_model_mismatch(sdb, tmp_path, capsys):
    cfg = tiny_cfg(tmp_path, "b0")
    assert main(["train", "--model", "b1", "--config", str(cfg)]) == 1
    with pytest.raises(ConfigError):
        run_train(cfg, "b1")


# --- M36: trial failures ------------------------------------------------------------------------------

def test_m36_trial_failure_recorded(sdb, tmp_path, monkeypatch):
    calls = {"n": 0}
    orig = seq.train_member

    def flaky(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TrialFailed("forced failure")
        return orig(*a, **kw)

    monkeypatch.setattr(seq, "train_member", flaky)
    cfg = tiny_cfg(tmp_path, "lstm", tuning__n_trials=6)
    run_id = run_train(cfg, "lstm")
    with open(tmp_path / "models" / run_id / "trials.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["status"] == "failed" and rows[0]["j_val"] == "inf" and "forced failure" in rows[0]["error"]
    assert all(r["status"] == "ok" for r in rows[1:])
    out = json.loads(_last_run(sdb["db"])[2])
    assert out["n_failed_trials"] == 1 and out["variants"]["A"]["best_trial"] == 1


def test_m36_all_trials_of_a_variant_fail(sdb, tmp_path, monkeypatch):
    orig = seq.train_member

    def fail_b(settings, *a, **kw):
        if settings["input_option"] == "B":
            raise TrialFailed("forced failure of option B")
        return orig(settings, *a, **kw)

    monkeypatch.setattr(seq, "train_member", fail_b)
    code, _ = _train(tmp_path, "lstm")
    assert code == 1
    rid, status, outputs = _last_run(sdb["db"])[:3]
    assert status == "failed" and "variant B" in json.loads(outputs)["error"]
    assert _preds(sdb["db"], rid) == []


# --- M17: parallel determinism ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", ["lstm", "rnn"])
def test_m17_parallel_determinism(sdb, tmp_path, model):
    con = connect(sdb["db"])
    data = load_dataset(con)
    con.close()
    cfg = load_model_config(tiny_cfg(tmp_path, model))
    cfg["training"]["n_jobs"] = 1
    a = train_model(model, cfg, data)
    cfg["training"]["n_jobs"] = 2
    b = train_model(model, cfg, data)
    assert [t.j_val for t in a.trials] == [t.j_val for t in b.trials]
    for v in a.models:
        assert np.array_equal(a.models[v].predict(data.X_val), b.models[v].predict(data.X_val))
        assert np.array_equal(a.models[v].predict_std(data.X_val), b.models[v].predict_std(data.X_val))


# --- M24: plateau report ------------------------------------------------------------------------------------

def test_m24_plateau_periods_by_hand():
    X = unit_X(9, 51)
    T, k0 = 30, 3
    Y = make_curves(X, T, k0) * np.array([15000.0, 5450.0, 9000.0])
    k_end, k_bt = plateau_periods(Y, k0, 0.98, 0.01)
    for i in range(len(X)):
        oil, wat = Y[i, :, 0], Y[i, :, 2]
        ke = next((k for k in range(k0, T) if oil[k] < 0.98 * oil.max()), T)
        kb = next((k for k in range(T) if wat[k] > 0.01 * wat.max()), T)
        assert k_end[i] == ke and k_bt[i] == kb
        assert k0 < ke < T and k0 < kb < T


@pytest.mark.parametrize("model", ["dmd", "b0"])
def test_m24_diagnostics_periods(sdb, tmp_path, model):
    con = connect(sdb["db"])
    data = load_dataset(con)
    con.close()
    cfg = load_model_config(tiny_cfg(tmp_path, model))
    out = train_model(model, cfg, data)
    d = diagnostics(out.models["main"], data, cfg)
    sets = {"train", "val"} | ({"ls_reconstruction_train"} if model == "dmd" else set())
    assert set(d["plateau"]) == sets
    for s in sets:
        errs = d["plateau"][s]["errors"]
        assert set(errs) == set(PERIODS)
        for per in PERIODS:
            assert {"oil", "gas", "water"} <= set(errs[per])
            if errs[per]["n"]:
                assert all(errs[per][f] >= 0 for f in ("oil", "gas", "water"))


# --- M45: DMD plausibility gate P1 (D-020) ---------------------------------------------------------

def _patch_continuation(monkeypatch, clip_targets):
    """Trial 0 (coef) keeps its fit; trial 1 (proj) gets a worse history fit. A target in
    clip_targets goes negative after T_fit, the others stay positive."""
    from dwc26.models.dmd import DMDProxy

    orig = DMDProxy._raw_rollout

    def raw(self, X, n_total):
        out = orig(self, X, n_total)
        h = self.T_fit - self.k0
        if self.settings["gp_target"] == "proj":
            out = out + 0.05
        if self.settings["gp_target"] in clip_targets:
            out[:, h:] = -1.0
        else:
            out[:, h:] = np.abs(out[:, h:]) + 0.01
        return out

    monkeypatch.setattr(DMDProxy, "_raw_rollout", raw)


def test_m45_gate_selects_plausible(sdb, tmp_path, monkeypatch):
    _patch_continuation(monkeypatch, {"coef"})
    run_id = run_train(tiny_cfg(tmp_path, "dmd"), "dmd")
    with open(tmp_path / "models" / run_id / "trials.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["gp_target"] for r in rows] == ["coef", "proj"]
    assert float(rows[0]["j_val"]) < float(rows[1]["j_val"])
    assert rows[0]["plausible"] == "False" and int(rows[0]["n_clip_cont"]) > 0
    assert rows[1]["plausible"] == "True" and int(rows[1]["n_clip_cont"]) == 0
    assert all(r["basis"] == "real" for r in rows)
    out = json.loads(_last_run(sdb["db"])[2])
    v = out["variants"]["main"]
    assert v["best_trial"] == 1 and v["dmd"]["plausible_selected"] is True and v["n_clipped"]["forecast_val"] == 0


def test_m45_gate_fallback(sdb, tmp_path, monkeypatch):
    _patch_continuation(monkeypatch, {"coef", "proj"})
    run_id = run_train(tiny_cfg(tmp_path, "dmd"), "dmd")
    rid, status, outputs = _last_run(sdb["db"])[:3]
    assert rid == run_id and status == "ok"
    v = json.loads(outputs)["variants"]["main"]
    assert v["best_trial"] == 0 and v["dmd"]["plausible_selected"] is False
    report = (tmp_path / "models" / run_id / "report.md").read_text(encoding="utf-8")
    assert "no plausible grid point; Task 3 rejects this proxy under P1" in report
