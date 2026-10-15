"""Is the adopted proxy (V4 x 20 seeds) significantly better than the baseline (V0, Elton)?

Per-case paired comparison on the same cases:
  - out-of-fold over cases 1-85 (5 folds, as in followup_runner_up.py), and
  - test cases 86-100 (models fitted on 1-70, the shipped setting).
Per case: RMS error over the 37 window dates, % of final observed cumulative, oil and water.
Tests: Wilcoxon signed-rank on the per-case differences, sign count, and a case-bootstrap
95 % interval for the ratio of pooled RMS (new / baseline).

Run from the repo root:  python improve_proxy/significance_test.py
Writes improve_proxy/results/significance_per_case.csv and significance_summary.csv
"""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "improve_proxy")]
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

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
VARIANTS = {"V0": (pv.VConfig(), 743),
            "V4x20": (pv.VConfig(**{**V4.__dict__, "n_seeds": 20}), 1145)}
K = pv.window_index(yh)
FIN = yh[-1]


def per_case(pred, idx):
    R = (pred[idx][:, K] - Y[idx][:, K]) / FIN
    return {ph: np.sqrt((R[:, :, j] ** 2).mean(axis=1)) * 100 for j, ph in [(pv.OIL, "oil"), (pv.WATER, "water")]}


def oof85(cfg, ep):
    out = np.full(Y.shape, np.nan)
    for itr, iva in FOLDS85:
        out[iva] = pv.fit_mlp(cfg, th, gr, Y, yh, itr, ep).predict(th[iva], gm[iva])
    return out


preds = {}
for name, (cfg, ep) in VARIANTS.items():
    preds[name, "oof85"] = oof85(cfg, ep)
    preds[name, "test"] = pv.fit_mlp(cfg, th, gr, Y, yh, TR, ep).predict(th, gm)
    print(name, "done", flush=True)

rng = np.random.default_rng(0)
rows, summ = [], []
for split, idx in [("oof85", ALL85), ("test", TE)]:
    a, b = per_case(preds["V0", split], idx), per_case(preds["V4x20", split], idx)
    for ph in ("oil", "water"):
        for i, c in enumerate(idx):
            rows.append({"split": split, "phase": ph, "case": int(cases.loc[c, "case"]) if "case" in cases else c + 1,
                         "V0_rms_%": a[ph][i], "V4x20_rms_%": b[ph][i]})
        diff = b[ph] - a[ph]
        pooled = lambda e, s: np.sqrt((e[s] ** 2).mean())
        boots = [pooled(b[ph], s) / pooled(a[ph], s)
                 for s in (rng.integers(0, len(idx), len(idx)) for _ in range(5000))]
        summ.append({"split": split, "phase": ph, "n_cases": len(idx),
                     "V0_pooled_%": pooled(a[ph], slice(None)), "V4x20_pooled_%": pooled(b[ph], slice(None)),
                     "ratio": pooled(b[ph], slice(None)) / pooled(a[ph], slice(None)),
                     "ratio_ci_lo": np.percentile(boots, 2.5), "ratio_ci_hi": np.percentile(boots, 97.5),
                     "cases_better": int((diff < 0).sum()),
                     "wilcoxon_p": wilcoxon(b[ph], a[ph]).pvalue})
pd.DataFrame(rows).to_csv(ROOT / "improve_proxy" / "results" / "significance_per_case.csv", index=False)
out = pd.DataFrame(summ)
out.to_csv(ROOT / "improve_proxy" / "results" / "significance_summary.csv", index=False)
print(out.round(4).to_string())
