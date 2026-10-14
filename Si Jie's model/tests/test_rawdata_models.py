"""M40, M41: Task 2 on the real database (marker rawdata; skipped without data/raw/)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from dwc26.config import load_model_config
from dwc26.models.dataset import load_dataset
from dwc26.models.registry import get_class
from dwc26.models.train_report import PERIODS, diagnostics
from dwc26.models.tuning import resolve_settings

pytestmark = pytest.mark.rawdata
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def real_data(real_prepared):
    con = sqlite3.connect(real_prepared["cfg"]["paths"]["db"])
    data = load_dataset(con)
    out = json.loads(con.execute("SELECT outputs FROM runs WHERE run_id = ?",
                                 (real_prepared["run_id"],)).fetchone()[0])
    split = dict(con.execute("SELECT case_id, split FROM parameters").fetchall())
    con.close()
    return data, out, split


def test_m40_load_dataset(real_data):
    data, out, split = real_data
    assert data.X_fit.shape == (70, 4) and data.Y_fit.shape == (70, 41, 3)
    assert data.X_val.shape == (15, 4) and data.Y_val.shape == (15, 41, 3)
    from dwc26.models.base import onset_index

    assert onset_index(data.Y_fit) == 4
    assert all(split[int(c)] == "train" for c in data.case_fit)
    assert all(split[int(c)] == "val" for c in data.case_val)
    test_ids = {c for c, s in split.items() if s == "test"}
    assert len(test_ids) == 15 and not test_ids & (set(data.case_fit.tolist()) | set(data.case_val.tolist()))
    assert data.arrays_sha256 == out["arrays_sha256"]


@pytest.mark.parametrize("name", ["dmd", "b0", "b1"])
def test_m41_untuned_fits(real_data, name, capsys):
    data, _, _ = real_data
    cfg = load_model_config(REPO / "configs" / "models" / f"{name}.yaml")
    tuned = ({"rank": 4, "kernel": "matern25", "fluid_weights": "d012", "gp_target": "proj"}   # A2
             if name == "dmd" else {})
    m = get_class(name)().fit(data.X_fit, data.Y_fit, resolve_settings(cfg, "main", tuned))
    P = m.predict(data.X_val)
    assert P.shape == (15, 41, 3) and np.isfinite(P).all() and (P >= 0).all()
    d = diagnostics(m, data, cfg)
    assert np.isfinite(d["j_val"])
    assert set(d["plateau"]["val"]["errors"]) == set(PERIODS)
    if name == "dmd":
        assert m.diagnostics["basis"] == "real" and m.diagnostics["r_eff"] == 4
        assert (m.reps.imag == 0).all() and (m.reps.real > 0).all() and (m.reps.real <= 1).all()
        assert isinstance(m.diagnostics["r_99"], int) and m.diagnostics["r_99"] >= 1
        assert "ls_reconstruction_train" in d["plateau"]
        F = m.forecast(data.X_val, 80)
        assert F.shape == (15, 80, 3) and np.isfinite(F).all() and (F >= 0).all()
        with capsys.disabled():
            print(f"\nM41 dmd (real, rank 4, proj): lambda {np.round(m.reps.real, 6).tolist()}, "
                  f"r_99 {m.diagnostics['r_99']}, J_val {d['j_val']:.5f}, continuation clip count {m.last_n_clipped}")
