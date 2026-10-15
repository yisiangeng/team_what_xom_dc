"""E27: dwc26.evaluation.plots (matplotlib, Agg backend; spec task3 section 1.8)."""

from __future__ import annotations

import time

import numpy as np

from dwc26.cli import main
from dwc26.evaluation.plots import plot_backtest, plot_curves, plot_parity
from our_work.proxy_model_sijie.tests.synth_eval import set_key, tiny_eval_cfg_dict, write_eval_cfg

PNG = b"\x89PNG\r\n\x1a\n"


def _png(path) -> bool:
    data = path.read_bytes()
    return len(data) > 100 and data[:8] == PNG


def test_e27_plot_functions(tmp_path):
    rng = np.random.default_rng(0)
    n, T, n_fc = 6, 13, 8
    dates = np.arange(np.datetime64("2000-01"), np.datetime64("2000-01") + 3 * T, np.timedelta64(3, "M")).astype(
        "datetime64[D]")
    fc = np.arange(np.datetime64("2003-04"), np.datetime64("2003-04") + 3 * n_fc, np.timedelta64(3, "M")).astype(
        "datetime64[D]")
    Y = rng.random((n, T, 3)) * 1000
    curves = {"lstm/A": rng.random((n, T + n_fc, 3)) * 1000, "dmd/main": rng.random((n, T + n_fc, 3)) * 1000}
    plot_backtest(tmp_path / "warm.png", {"x": np.ones((4, 3))}, np.ones((4, 3)), "warm-up", 50)   # one-time import
    t0 = time.perf_counter()
    plot_curves(tmp_path / "c.png", dates, fc, np.arange(19, 25), Y, Y[0], curves, 0, "curves", 60)
    t1 = time.perf_counter()
    plot_parity(tmp_path / "p.png", Y, {"b0/main": Y * 0.9, "lstm/A": Y * 1.1}, 2, "parity", 60)
    t2 = time.perf_counter()
    plot_backtest(tmp_path / "b.png", {"lstm/A": rng.random((4, 3))}, rng.random((4, 3)), "backtest", 60)
    t3 = time.perf_counter()
    for name in ("c.png", "p.png", "b.png"):
        assert _png(tmp_path / name)
    assert max(t1 - t0, t2 - t1, t3 - t2) < 2.0


def test_e27_figures_of_the_commands(tiny_task2_db, tmp_path):
    cfg = set_key(tiny_eval_cfg_dict(tiny_task2_db, tmp_path), "report.plots", True)
    path = write_eval_cfg(cfg, tmp_path / "evaluate.yaml")
    t0 = time.perf_counter()
    assert main(["evaluate", "--split", "val", "--config", str(path)]) == 0
    fig = tmp_path / "figures"
    val = ["curves_val_oil.png", "curves_val_gas.png", "curves_val_water.png", "parity_val.png"]
    assert sorted(p.name for p in fig.glob("*.png")) == sorted(val)
    assert all(_png(fig / f) for f in val)
    assert main(["backtest", "--config", str(path)]) == 0
    assert _png(fig / "backtest_val.png")
    assert len(list(fig.glob("*.png"))) == 5
    assert time.perf_counter() - t0 < 60
