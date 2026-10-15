"""E30, E31: Task 3 on the real database (marker rawdata; validation path only, spec task3 section 4.3).

No raw-data test calls load_split(..., 'test') or `evaluate --split test` before the freeze.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from dwc26.config import load_eval_config
from dwc26.evaluation.splits import load_split, train_reference
from dwc26.forecast.backtest import check_window, cutoff_index
from dwc26.forecast.dates import quarters_between
from dwc26.hm.objective import build_mismatch_spec, mismatch

pytestmark = pytest.mark.rawdata


@pytest.fixture(scope="module")
def real_splits(real_prepared):
    from pathlib import Path

    cfg = load_eval_config(Path(__file__).resolve().parents[1] / "configs" / "evaluate.yaml")
    con = sqlite3.connect(real_prepared["cfg"]["paths"]["db"])
    train, val = load_split(con, "train"), load_split(con, "val")
    con.close()
    return cfg, train, val


def test_e30_real_splits_and_windows(real_splits):
    cfg, train, val = real_splits
    assert train.Y_phys.shape == (70, 41, 3) and val.Y_phys.shape == (15, 41, 3)
    assert train.X.shape == (70, 4) and val.X.shape == (15, 4)
    assert val.case_id.tolist() == list(range(71, 86))
    ref = train_reference(train)
    assert ref.k0 == 4
    spec = build_mismatch_spec(val.hist_phys, val.dates, cfg["mismatch"])
    assert spec.k_idx.size == 37
    assert str(val.dates[spec.k_idx[0]]) == "1999-01-01" and str(val.dates[spec.k_idx[-1]]) == "2008-01-01"
    assert quarters_between(val.dates[-1], np.datetime64(cfg["continuation"]["end_date"])) == 80
    t_cut = cutoff_index(val.dates, cfg["backtest"]["cutoff_date"])
    assert t_cut == 29 and check_window(t_cut, val.T, ref.k0, cfg["backtest"]["v6_horizons"]) == 12


def test_e31_mismatch_on_the_real_history(real_splits):
    cfg, train, val = real_splits
    spec = build_mismatch_spec(val.hist_phys, val.dates, cfg["mismatch"])
    assert mismatch(val.hist_phys[None], spec)[0] == 0.0
    O_sim = mismatch(val.Y_phys, spec)
    assert O_sim.shape == (15,) and np.isfinite(O_sim).all() and (O_sim > 0).all()
    assert (spec.sigma_obs > 0).all()
    water = list(cfg["mismatch"]["fluids"]).index("water")
    d_w = spec.d[:, water]
    floor = cfg["mismatch"]["beta"] * d_w.max()
    pre = d_w * cfg["mismatch"]["alpha"] < floor                     # dates where the floor applies
    assert pre.any() and np.allclose(spec.sigma_obs[pre, water], floor)
    first_bt = int(np.argmax(d_w > 0)) if (d_w > 0).any() else len(d_w)
    assert pre[:max(first_bt, 1)].all()                              # every date before breakthrough
