"""History matching for DWC26 (steps 2-3): find parameter sets that reproduce the
observed 1998-2008 field history.

Objective (exactly as fixed in the plan): for cumulative curves ``C_hat(theta)``
predicted on the 41 quarterly dates and the observed history ``C_obs``,

    m(theta) = sqrt( mean over 3 phases x 41 dates of ((C_hat(t) - C_obs(t)) / C_obs_final)^2 )

with equal weight per phase (``C_obs_final`` = observed cumulative at 2008-01-01).

Search protocol:
    1. global: 40,000 scrambled Sobol points in the empirical bounds
       (min/max of the 100 provided cases, widened by 5% of the range);
    2. local polish: 5,000 points drawn uniformly in +-2%-of-range neighbourhoods
       around the best 20 global points, clipped to the same bounds.

Artifacts (``cod/proxy_outputs/``), consumed by ``forecast_pipeline.py --forecast``:
    hm_candidates.csv        best 5,000 sampled sets (misfit, per-phase misfit, params)
    hm_matched_models.csv    the top-20 sets - the "history-matched models" (step 3)
    hm_summary.json          bounds, sample counts, seed, model hash, benchmark vs cases
    hm_misfit_hist.png       misfit distribution vs the 100 provided cases
    hm_match_vs_observed.png top-3 matches against the observed history

Run from ``cod/`` with the venv interpreter (after ``--train-cumulative``):

    python history_match.py [--n-global 40000] [--n-local 5000] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

import our_work.proxy_model_elton.forecast_pipeline as fp
from our_work.proxy_model_elton.proxy_model import OUT, PARAMS, PHASES, SEED

BOUNDS_MARGIN = 0.05     # search bounds = case envelope +- 5% of the range
LOCAL_CENTERS = 20       # polish around the best 20 global points
LOCAL_SPAN = 0.02        # neighbourhood half-width as a fraction of the parameter range
POOL_KEEP = 5_000        # rows kept in hm_candidates.csv
TOP_MATCHED = 20         # rows kept in hm_matched_models.csv (step-3 answer)


def search_bounds(cases):
    """Empirical envelope of all 100 provided cases, widened by 5% of the range."""
    p = cases[PARAMS].to_numpy(dtype=np.float64)
    lo, hi = p.min(axis=0), p.max(axis=0)
    width = np.where(hi > lo, hi - lo, 1.0)
    return lo - BOUNDS_MARGIN * width, hi + BOUNDS_MARGIN * width


def sobol_points(n, lo, hi, seed):
    """n scrambled Sobol points scaled from [0,1]^d onto [lo, hi]."""
    from scipy.stats.qmc import Sobol

    power = int(np.ceil(np.log2(max(n, 2))))
    u = Sobol(d=len(lo), scramble=True, seed=seed).random_base2(power)[:n]
    return lo + u * (hi - lo)


def local_points(centers, n, lo, hi, seed):
    """n uniform points in +-LOCAL_SPAN-of-range boxes around the centers, clipped."""
    rng = np.random.default_rng(seed + 1)
    per_center = int(np.ceil(n / len(centers)))
    u = rng.random((len(centers), per_center, lo.shape[0])) * 2.0 - 1.0
    half = LOCAL_SPAN * (hi - lo)
    pts = np.clip(centers[:, None, :] + u * half[None, None, :], lo, hi)
    return pts.reshape(-1, lo.shape[0])[:n]


def evaluate(ckpt, theta, y_hist):
    """Cumulative misfit of a batch of parameter sets; returns (total (n,), per-phase (n, 3)).

    The prediction is thrown away here to keep the 45k-sample search light; the few
    curves that get plotted are re-predicted later.
    """
    total, per_phase = fp.hm_misfit_parts(fp.predict_cumulative(ckpt, theta), y_hist)
    return total, per_phase


def run_search(n_global=40_000, n_local=5_000, seed=SEED):
    t0 = time.time()
    data = fp.load_cumulative_data()
    ckpt = fp.load_cumulative_model()
    y_hist = data["y_hist"]
    lo, hi = search_bounds(data["cases"])
    print("search bounds:")
    for j, p in enumerate(PARAMS):
        print(f"  {p:<12} [{lo[j]:.4g}, {hi[j]:.4g}]")

    # 1. global Sobol screen
    theta_g = sobol_points(n_global, lo, hi, seed)
    mis_g, per_g = evaluate(ckpt, theta_g, y_hist)
    order_g = np.argsort(mis_g)
    print(f"global: {n_global:,} samples, best m = {mis_g[order_g[0]]:.5f}")

    # 2. local polish around the best global points
    theta_l = local_points(theta_g[order_g[:LOCAL_CENTERS]], n_local, lo, hi, seed)
    mis_l, per_l = evaluate(ckpt, theta_l, y_hist)
    print(f"polish: {n_local:,} samples around top {LOCAL_CENTERS}, "
          f"best m = {mis_l.min():.5f}")

    theta = np.vstack([theta_g, theta_l])
    misfit = np.concatenate([mis_g, mis_l])
    per = np.vstack([per_g, per_l])
    stage = np.array(["global"] * len(theta_g) + ["polish"] * len(theta_l))
    rank = np.argsort(misfit)
    best = int(rank[0])
    print(f"combined best m = {misfit[best]:.5f}")

    # 3. benchmark against the 100 provided cases (same objective, stored curves)
    case_m, _ = fp.case_misfits(data)
    best_case = int(np.argmin(case_m))
    best_case_id = int(data["cases"]["case_id"].iloc[best_case])
    print(f"provided cases: best case {best_case_id} m = {case_m[best_case]:.4f}, "
          f"median m = {np.median(case_m):.4f}")

    # 4. artifacts
    keep = rank[:POOL_KEEP]
    cand = pd.DataFrame({
        "stage": stage[keep],
        "misfit": misfit[keep],
        **{f"misfit_{ph}": per[keep, j] for j, ph in enumerate(PHASES)},
        **{p: theta[keep, j] for j, p in enumerate(PARAMS)},
    })
    cand.to_csv(OUT / "hm_candidates.csv", index=False)

    matched = cand.head(TOP_MATCHED).copy()
    matched.insert(0, "rank", np.arange(1, TOP_MATCHED + 1))
    matched.to_csv(OUT / "hm_matched_models.csv", index=False)

    summary = {
        "objective": "sqrt(mean over 3 phases x 41 dates of ((C_hat - C_obs) / C_obs_final)^2)",
        "model": {"path": str(fp.CUM_MODEL), "sha256": fp.ckpt_summary()["sha256"]},
        "bounds": {
            "source": "min/max of the 100 provided cases, widened by the margin",
            "margin_fraction": BOUNDS_MARGIN,
            "lo": {p: float(lo[j]) for j, p in enumerate(PARAMS)},
            "hi": {p: float(hi[j]) for j, p in enumerate(PARAMS)},
        },
        "search": {
            "sampler": "scipy.stats.qmc.Sobol (scrambled)",
            "n_global": int(n_global), "n_local": int(n_local),
            "local_centers": LOCAL_CENTERS, "local_span_fraction": LOCAL_SPAN,
            "seed": int(seed),
        },
        "result": {
            "best_misfit": float(misfit[best]),
            "best_theta": {p: float(theta[best, j]) for j, p in enumerate(PARAMS)},
            "best_provided_case": {"case_id": best_case_id,
                                   "misfit": float(case_m[best_case])},
            "provided_misfit_median": float(np.median(case_m)),
            "improvement_vs_best_case": float(case_m[best_case] / misfit[best]),
        },
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (OUT / "hm_summary.json").write_text(json.dumps(summary, indent=2))

    print("\ntop-5 matches (proxy search):")
    top = rank[:5]
    print(pd.DataFrame({"rank": np.arange(1, 6), "misfit": misfit[top],
                        **{p: theta[top, j] for j, p in enumerate(PARAMS)}})
          .to_string(index=False, float_format=lambda v: f"{v:.5g}"))
    print("\ntop-5 provided cases (same objective):")
    ord_c = np.argsort(case_m)[:5]
    print(pd.DataFrame({"case_id": data["cases"]["case_id"].to_numpy()[ord_c],
                        "misfit": case_m[ord_c]})
          .to_string(index=False, float_format=lambda v: f"{v:.5g}"))

    # 5. figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xmax = float(np.percentile(misfit, 99.5))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(np.clip(misfit, 0, xmax), bins=60, color="tab:blue", alpha=0.75)
    ax.axvline(misfit[best], color="tab:red", lw=2,
               label=f"best match m={misfit[best]:.4f}")
    ax.axvline(1.25 * misfit[best], color="tab:red", lw=1.3, ls=":",
               label="1.25 x best (well-matching cutoff)")
    ax.axvline(case_m[best_case], color="black", lw=1.6, ls="--",
               label=f"best provided case {best_case_id}  m={case_m[best_case]:.4f}")
    ax.axvline(np.median(case_m), color="grey", lw=1.2, ls="-.",
               label=f"median provided case  m={np.median(case_m):.4f}")
    ax.set_xlabel("misfit m (normalized cumulative RMSE; x clipped at the 99.5th percentile)")
    ax.set_ylabel("samples")
    ax.set_title(f"History-match misfit: {len(misfit):,} proxy samples vs the 100 provided cases")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "hm_misfit_hist.png", dpi=150)
    plt.close(fig)

    pred_top = fp.predict_cumulative(ckpt, theta[rank[:3]])
    dates = pd.DatetimeIndex(data["dates"])
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    colors = ["tab:red", "tab:orange", "tab:green"]
    for j, phase in enumerate(PHASES):
        ax = axes[j]
        ax.plot(dates, y_hist[:, j], color="black", lw=2.4, label="observed history")
        for k, i in enumerate(rank[:3]):
            ax.plot(dates, pred_top[k, :, j], color=colors[k], lw=1.4,
                    label=f"match {k + 1}  m={misfit[i]:.4f}")
        ax.set_title(f"{phase} cumulative")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Best history matches vs observed history "
                 f"(best provided case {best_case_id} m = {case_m[best_case]:.4f})")
    fig.tight_layout()
    fig.savefig(OUT / "hm_match_vs_observed.png", dpi=150)
    plt.close(fig)

    print(f"\nwrote {OUT / 'hm_candidates.csv'}, {OUT / 'hm_matched_models.csv'}, "
          f"{OUT / 'hm_summary.json'}, {OUT / 'hm_misfit_hist.png'}, "
          f"{OUT / 'hm_match_vs_observed.png'} ({summary['runtime_seconds']:.0f} s)")
    return misfit, case_m


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-global", type=int, default=40_000)
    ap.add_argument("--n-local", type=int, default=5_000)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    run_search(n_global=args.n_global, n_local=args.n_local, seed=args.seed)


if __name__ == "__main__":
    main()
