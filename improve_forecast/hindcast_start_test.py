"""Hindcast: should the forecast start from each matched case's own state or from the observed state?

For each truth (the real field, and each of the 100 simulated cases as a pseudo-field) and each
cut-off (2005, 2006, 2007):
  1. history-match blind: rank the simulated cases (truth excluded) by the oil + water cumulative
     misfit on the window dates up to the cut-off only; keep the best 10;
  2. forecast each of the 10 from the cut-off to 2008-01-01 with the final method (crm_wor, n=12,
     WOR last), with each start option;
  3. compare the mean of the 10 with what the truth did (cumulative on 2008-01-01).

Start options:
  main       each case's own cumulative and rates (the current deliverable)
  A1         observed cumulative, case's own rates (shift only)
  A2         observed cumulative and observed rates, held (case gives tau, G, WOR slope)
  fade_T     observed cumulative; observed rates fading to the case's own path over T days
  sim        diagnostic: the matched cases' real simulator continuation, own start (no forecast method)
  sim_shift  diagnostic: the same, put on the observed cumulative
  field_only the final method on the truth's own history (no model)

Error = (forecast - truth) / truth of the volume produced after the cut-off, % (oil, water).
Simulated pseudo-fields come from the same simulator as the matches (in-family), the real field
may not (the 2008 rate bias suggests it does not), so both are reported.

Run from the repo root:  python improve_forecast/hindcast_start_test.py
"""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import forecast_lib as fl
import proxy_lib as pl

OIL, GAS, WATER = fl.OIL, fl.GAS, fl.WATER
OUT = ROOT / "improve_forecast" / "results"
OUT.mkdir(parents=True, exist_ok=True)

D = pl.load_data("cumulative")
Y, YH, T = D["Y"], D["y_hist"], D["t_days"]
INJ = pd.read_csv(ROOT / "data_after_engineering" / "history_observed.csv")["water_inj_rate"].to_numpy(float)
K_END, W0 = len(T) - 1, 4                    # 2008-01-01; window starts 1999-01-01 (index 4)
CUTS, N_FIT, ANCHOR, TOPN = [28, 32, 36], 12, "last", 10
FADES = [180, 365, 730]
QY = fl.quarter_rates(Y, T)


def blind_top(truth, kc, exclude=None):
    K = np.arange(W0, kc + 1)
    ref = truth[kc, [OIL, WATER]]
    R = (Y[:, K][:, :, [OIL, WATER]] - truth[K][:, [OIL, WATER]]) / ref
    s = np.sqrt((R ** 2).mean(axis=(1, 2)))
    if exclude is not None:
        s[exclude] = np.inf
    return np.argsort(s)[:TOPN]


def added(C, kc, tout, **kw):
    return np.array(fl.forecast_case("crm_wor", C, T, kc, tout, N_FIT, INJ, INJ[kc], wor_anchor=ANCHOR, **kw))[:, -1]


def run_truth(truth, kc, exclude=None):
    tout = T[kc + 1:]
    top = blind_top(truth, kc, exclude)
    qt = fl.quarter_rates(truth[None, : kc + 1], T[: kc + 1])[0]
    obs_rates = (qt[kc, OIL], qt[kc, WATER])
    cum_obs = truth[kc, [OIL, WATER]]
    end = {}                                                   # option -> (10, 2) cumulative oil, water on 2008
    own = np.array([added(Y[i], kc, tout) for i in top])
    end["main"] = Y[top, kc][:, [OIL, WATER]] + own
    end["A1"] = cum_obs + own
    end["A2"] = cum_obs + np.array([added(Y[i], kc, tout, anchor_rates=obs_rates) for i in top])
    for f in FADES:
        end[f"fade_{f}"] = cum_obs + np.array([added(Y[i], kc, tout, anchor_rates=obs_rates, anchor_fade_days=f) for i in top])
    end["sim"] = Y[top, K_END][:, [OIL, WATER]]
    end["sim_shift"] = cum_obs + (Y[top, K_END] - Y[top, kc])[:, [OIL, WATER]]
    end["field_only"] = (cum_obs + added(truth, kc, tout))[None]
    produced = truth[K_END, [OIL, WATER]] - cum_obs
    err = {k: (v.mean(axis=0) - truth[K_END, [OIL, WATER]]) / produced * 100 for k, v in end.items()}
    err_cum = {k: (v.mean(axis=0) - truth[K_END, [OIL, WATER]]) / truth[K_END, [OIL, WATER]] * 100 for k, v in end.items()}
    lo = {k: v.min(axis=0) for k, v in end.items()}
    hi = {k: v.max(axis=0) for k, v in end.items()}
    cover = {k: ((lo[k] <= truth[K_END, [OIL, WATER]]) & (truth[K_END, [OIL, WATER]] <= hi[k])) for k in end}
    rate_gap = (QY[top, kc][:, [OIL, WATER]].mean(axis=0) / np.array(obs_rates) - 1) * 100
    return err, err_cum, cover, rate_gap


rows = []
for kc in CUTS:
    for j in range(len(Y) + 1):
        is_field = j == len(Y)
        truth = YH if is_field else Y[j]
        err, err_cum, cover, gap = run_truth(truth, kc, None if is_field else j)
        for opt, e in err.items():
            rows.append({"truth": "field" if is_field else f"case_{j + 1}", "cutoff": str(pd.Timestamp(D["dates"][kc]).date()),
                         "horizon_q": K_END - kc, "option": opt, "err_oil_%": e[0], "err_water_%": e[1],
                         "cum_err_oil_%": err_cum[opt][0], "cum_err_water_%": err_cum[opt][1],
                         "covered_oil": bool(cover[opt][0]), "covered_water": bool(cover[opt][1]),
                         "match_rate_gap_oil_%": gap[0], "match_rate_gap_water_%": gap[1]})
    print("cut-off", kc, "done", flush=True)
R = pd.DataFrame(rows)
R.to_csv(OUT / "hindcast_all.csv", index=False)

# Summaries
R["abs_mean"] = (R["err_oil_%"].abs() + R["err_water_%"].abs()) / 2
pseudo = R[R.truth != "field"]
S = pseudo.groupby("option").agg(med_abs_oil=("err_oil_%", lambda x: np.median(np.abs(x))),
                                 med_abs_water=("err_water_%", lambda x: np.median(np.abs(x))),
                                 bias_oil=("err_oil_%", "median"), bias_water=("err_water_%", "median"),
                                 score=("abs_mean", "mean"))
F = R[R.truth == "field"].pivot_table(index="option", columns="cutoff", values=["err_oil_%", "err_water_%"])
F.columns = [f"field_{a.split('_')[1]}_{b[:4]}" for a, b in F.columns]
F["field_score"] = R[R.truth == "field"].groupby("option")["abs_mean"].mean()
S = S.join(F).sort_values("field_score")
S.to_csv(OUT / "hindcast_summary.csv")

# Paired tests on the pseudo-fields: each option against main (per truth x cut-off, mean |error| of oil and water)
base = pseudo[pseudo.option == "main"].set_index(["truth", "cutoff"])["abs_mean"]
tests = []
for opt in S.index:
    if opt in ("main", "field_only"):
        continue
    o = pseudo[pseudo.option == opt].set_index(["truth", "cutoff"])["abs_mean"].loc[base.index]
    tests.append({"option": opt, "pseudo_mean_abs": o.mean(), "main_mean_abs": base.mean(),
                  "share_better_%": float((o < base).mean() * 100), "wilcoxon_p": wilcoxon(o, base).pvalue})
a1 = pseudo[pseudo.option == "A1"].set_index(["truth", "cutoff"])["abs_mean"].loc[base.index]
a2 = pseudo[pseudo.option == "A2"].set_index(["truth", "cutoff"])["abs_mean"].loc[base.index]
tests.append({"option": "A2 vs A1 (rate anchoring alone)", "pseudo_mean_abs": a2.mean(), "main_mean_abs": a1.mean(),
              "share_better_%": float((a2 < a1).mean() * 100), "wilcoxon_p": wilcoxon(a2, a1).pvalue})
pd.DataFrame(tests).to_csv(OUT / "hindcast_tests_vs_main.csv", index=False)

# The same on the 2008 cumulative (% of the truth's cumulative), the quantity the deliverable reports
R["cum_abs_mean"] = (R["cum_err_oil_%"].abs() + R["cum_err_water_%"].abs()) / 2
CUM = R[R.truth != "field"].groupby("option")["cum_abs_mean"].mean().rename("pseudo_cum_score").to_frame()
CUM = CUM.join(R[R.truth == "field"].pivot_table(index="option", columns="cutoff", values="cum_abs_mean").add_prefix("field_cum_"))
CUM.to_csv(OUT / "hindcast_cumulative_error.csv")

pd.set_option("display.width", 220)
print("\nPseudo-fields (100 simulated cases as truth, 3 cut-offs) and the real field. Errors in % of produced volume:")
print(S.round(2).to_string())
print("\nPaired against main (pseudo-fields):")
print(pd.DataFrame(tests).round(4).to_string(index=False))
print("\nError on the 2008 cumulative, mean of |oil| and |water|, % of the truth's cumulative:")
print(CUM.round(3).to_string())
print("\nRate gap of the 10 blind matches at the cut-off vs the truth (field):")
print(R[(R.truth == "field") & (R.option == "main")][["cutoff", "match_rate_gap_oil_%", "match_rate_gap_water_%"]].round(1).to_string(index=False))
print("pseudo-fields, median |gap|:", pseudo[pseudo.option == "main"][["match_rate_gap_oil_%", "match_rate_gap_water_%"]].abs().median().round(2).to_dict())
print("\nCoverage (truth inside the min-max of the 10), field:")
print(R[R.truth == "field"].groupby("option")[["covered_oil", "covered_water"]].mean().to_string())
