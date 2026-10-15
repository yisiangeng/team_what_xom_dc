"""Follow-up to notebook3_proxy_variants_test.ipynb (15 Oct 2026).

The notebook's rule picked the MLP + GP average, which the single test then rejected (oil
0.30 % against 0.07 %). This script checks the runner-up, the tuned MLP (V4, which also passed
the rule with 20 seeds), with the honest out-of-fold error over cases 1-85 (5 folds, models
trained exactly like the final model, fixed epochs) and, openly as a second use of the test
cases, on cases 86-100. It also checks whether a refit on cases 1-85 helps.

Run from the repo root:  python improve_proxy/followup_runner_up.py
Writes improve_proxy/results/followup_runner_up.csv
"""
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "improve_proxy")]
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

import proxy_lib as pl
import proxy_variants as pv

torch.set_num_threads(4)
d = pl.load_data("cumulative")
cases = d["cases"]
sp = cases["split"].to_numpy()
th = cases[pl.PARAMS].to_numpy(float)
gr = d["grid"][pl.GRID_FEATURES].to_numpy(float)
Y, yh = d["Y"], d["y_hist"]
gm = pl.GridMapper(d).transform(th)
TR, VA, TE = (np.where(sp == s)[0] for s in ("train", "validation", "test"))
ALL85 = np.r_[TR, VA]
va_fold = {c: k % 5 for k, c in enumerate(VA)}
fold = np.array([cases.loc[i, "cv_fold"] if i in TR else va_fold.get(i, -1) for i in range(100)])
FOLDS85 = [(ALL85[fold[ALL85] != f], ALL85[fold[ALL85] == f]) for f in range(5)]

V4 = pv.VConfig(name="V4", target="ow_window", lr=3e-3, weight_decay=1e-5)
VARIANTS = [("V0 baseline (Elton)", pv.VConfig(), 743),
            ("V1 oil+water, producing dates", pv.VConfig(target="ow_window"), 1052),
            ("V4 tuned, 5 seeds", V4, 1145),
            ("V4 tuned, 20 seeds (adopted)", pv.VConfig(**{**V4.__dict__, "n_seeds": 20}), 1145)]


def oof85(cfg, epochs):
    out = np.full(Y.shape, np.nan)
    for itr, iva in FOLDS85:
        out[iva] = pv.fit_mlp(cfg, th, gr, Y, yh, itr, epochs).predict(th[iva], gm[iva])
    return out


rows, t0 = [], time.time()
for name, cfg, ep in VARIANTS:
    e85 = pv.hm_errors(oof85(cfg, ep), Y, yh, ALL85)
    te70 = pv.hm_errors(pv.fit_mlp(cfg, th, gr, Y, yh, TR, ep).predict(th, gm), Y, yh, TE)
    te85 = pv.hm_errors(pv.fit_mlp(cfg, th, gr, Y, yh, ALL85, ep).predict(th, gm), Y, yh, TE)
    rows.append({"variant": name, "epochs": ep, "seeds": cfg.n_seeds,
                 **{f"oof85_{k}": v for k, v in e85.items() if "rms" in k or k == "J_%"},
                 **{f"test_fit70_{k}": v for k, v in te70.items() if "rms" in k or k == "J_%"},
                 **{f"test_fit85_{k}": v for k, v in te85.items() if "rms" in k or k == "J_%"}})
    print(f"{name:<32} done ({time.time() - t0:.0f} s)")
out = pd.DataFrame(rows).set_index("variant")
(ROOT / "improve_proxy" / "results").mkdir(exist_ok=True, parents=True)
out.to_csv(ROOT / "improve_proxy" / "results" / "followup_runner_up.csv")
print(out.round(4).to_string())
