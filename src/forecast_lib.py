"""Production forecast methods for DWC26 (notebook 4).

The simulated cases stop at 2008-01-01, so the 20-year forecast cannot be learned from
simulator runs. Each history-matched case is extended with an engineering forecast
method, chosen by a backtest on the 100 simulated cases and the observed history.

Conventions:
* time ``t`` in days since 1998-01-01; quarterly cumulative curves C (n, T, 3) in PHASES
  order [oil, gas, water]; rates are quarter means q_k = (C_k - C_{k-1}) / (t_k - t_{k-1})
  (the same definition for the cases and the history);
* a method takes one case's curve up to the cutoff and returns cumulative oil and water at
  the requested future times; gas = GOR x oil.

Methods (sources in summary_forecast.md):
* ``elton``   : Elton's cumulative families (Arps-cumulative for oil, Gompertz for water) fitted to
  the whole cumulative curve, shifted to be continuous (baseline).
* ``arps_wor``: modified hyperbolic Arps decline of the oil rate (b in [0, 1], switch to exponential
  at a terminal decline), water from the log(WOR) - cumulative-oil trend.
* ``liq_wor`` : total liquid rate with an exponential trend, split into oil and water with the
  log(WOR) - cumulative-oil trend (fractional-flow style).
* ``crm_wor`` : field-level capacitance-resistance model (tank) for the liquid rate, driven by the
  water injection rate (held constant after 2008), split with the same WOR trend.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

OIL, GAS, WATER = 0, 1, 2
DAYS_PER_YEAR = 365.25


def quarter_rates(C, t):
    """Quarter-mean rates (n, T, P) from cumulatives (n, T, P); the first date gets 0."""
    C = np.asarray(C, float)
    dt = np.diff(t, prepend=t[0])
    dt[0] = 1.0
    q = np.diff(C, axis=1, prepend=C[:, :1] * 0) / dt[None, :, None]
    q[:, 0] = 0.0
    return q


# ------------------------------------------------------------------ building blocks

def _window(k_cut, n_fit):
    return np.arange(k_cut - n_fit + 1, k_cut + 1)


def fit_arps(t, q, b_max=1.0):
    """Fit q(t) = qi / (1 + b Di (t - t0))^(1/b) on rates q at times t (t0 = t[-1]) by least squares
    on log q. Returns (qi, Di [1/day], b)."""
    t0 = t[-1]
    ok = q > 0
    t, q = t[ok], q[ok]

    def model(p, tt):
        qi, Di, b = np.exp(p[0]), np.exp(p[1]), p[2]
        x = tt - t0
        if b < 1e-6:
            return qi * np.exp(-Di * x)
        return qi * np.power(np.maximum(1 + b * Di * x, 1e-12), -1.0 / b)

    d0 = max(np.log(q[0] / q[-1]) / max(t[-1] - t[0], 1.0), 1e-6)
    p0 = [np.log(q[-1]), np.log(d0), 0.3]
    r = least_squares(lambda p: np.log(model(p, t)) - np.log(q), p0,
                      bounds=([np.log(q.min() * 0.2), np.log(1e-7), 0.0], [np.log(q.max() * 5), np.log(0.05), b_max]))
    return float(np.exp(r.x[0])), float(np.exp(r.x[1])), float(r.x[2])


def arps_rate(qi, Di, b, x, d_min_year=0.05):
    """Modified hyperbolic rate at x days after t0: hyperbolic until the instantaneous decline
    falls to d_min, then exponential at d_min (Robertson's remedy for long-range hyperbolic)."""
    d_min = -np.log(1 - d_min_year) / DAYS_PER_YEAR if d_min_year > 0 else 0.0
    x = np.asarray(x, float)
    if b < 1e-6 or Di <= d_min:
        return qi * np.exp(-max(Di, d_min) * x)
    x_sw = (Di / d_min - 1) / (b * Di) if d_min > 0 else np.inf      # where D(x) = Di / (1 + b Di x) = d_min
    hyp = qi * np.power(1 + b * Di * np.minimum(x, x_sw), -1.0 / b)
    return np.where(x <= x_sw, hyp, hyp * np.exp(-d_min * (x - x_sw)))


def fit_wor(Np, wor):
    """log(WOR) = a + c Np (Ershaghi-Omoregie / Wright style). Returns (a, c)."""
    ok = wor > 0
    c, a = np.polyfit(Np[ok], np.log(wor[ok]), 1)
    return float(a), float(max(c, 0.0))


def fit_liquid_trend(t, ql):
    """Exponential trend of the liquid rate: ql = q0 exp(-g (t - t0)), g >= 0 (no growth)."""
    c, a = np.polyfit(t - t[-1], np.log(ql), 1)
    return float(np.exp(a)), float(max(-c, 0.0))


def fit_crm(t, ql, inj, k_fit):
    """Field CRM (tank): ql_k = ql_{k-1} e^{-dt/tau} + (1 - e^{-dt/tau}) * G * inj_k.
    Fitted on the window k_fit by least squares. Returns (tau [days], G)."""
    def sim(p, ks):
        tau, G = np.exp(p[0]), np.exp(p[1])
        out = np.empty(len(ks))
        prev = ql[ks[0] - 1]
        for i, k in enumerate(ks):
            lam = np.exp(-(t[k] - t[k - 1]) / tau)
            prev = prev * lam + (1 - lam) * G * inj[k]
            out[i] = prev
        return out
    r = least_squares(lambda p: (sim(p, k_fit) - ql[k_fit]) / ql[k_fit].mean(), [np.log(180.0), np.log(1.3)],
                      bounds=([np.log(5.0), np.log(0.2)], [np.log(5000.0), np.log(5.0)]))
    return float(np.exp(r.x[0])), float(np.exp(r.x[1]))


# ------------------------------------------------------------------ forecasts

def _integrate(rate_fn, t0, t_out, step=5.0):
    """Cumulative of rate_fn(x) from t0 to each t_out (x = t - t0), trapezoid on a fine grid."""
    x = np.arange(0, t_out[-1] - t0 + step, step)
    r = rate_fn(x)
    cum = np.concatenate([[0], np.cumsum(0.5 * (r[1:] + r[:-1]) * np.diff(x))])
    return np.interp(t_out - t0, x, cum)


def _march_split(ql_fn, a, c, Np0, t0, t_out, step=5.0):
    """Step the liquid rate forward and split it with WOR(Np) = exp(a + c Np).
    a may be a function of x (days after t0), for a WOR offset that fades out."""
    a_fn = a if callable(a) else (lambda x, a=a: a)
    x = np.arange(0, t_out[-1] - t0 + step, step)
    Np = Np0
    no, nw = [0.0], [0.0]
    cum_o, cum_w = 0.0, 0.0
    for i in range(1, len(x)):
        xm = 0.5 * (x[i - 1] + x[i])
        ql = ql_fn(xm)
        wor = np.exp(min(a_fn(xm) + c * (Np + cum_o), 50.0))
        qo = ql / (1 + wor)
        cum_o += qo * step
        cum_w += (ql - qo) * step
        no.append(cum_o); nw.append(cum_w)
    return np.interp(t_out - t0, x, no), np.interp(t_out - t0, x, nw)


def forecast_case(method, C, t, k_cut, t_out, n_fit=12, inj=None, inj_future=None, d_min_year=0.05,
                  anchor_rates=None, wor_anchor="fit", anchor_fade_days=None):
    """Forecast one case.

    C: (T, 3) cumulative curve; uses dates 0..k_cut only. t_out: future times (days).
    anchor_rates: optional (oil, water) rates to start from instead of the case's own fitted
    rates at the cutoff (used for the "observed start" alternative).
    anchor_fade_days: crm_wor only. None = the anchor holds for good (A2). A number = the gap
    between the anchor and the case's own liquid rate and WOR fades out as exp(-x / fade), so
    the forecast starts at the observed rates and returns to the case's own path.
    wor_anchor: "fit" = the least-squares WOR line; "last" = same slope, moved through the WOR of
    the last 2 quarters (removes the lag of a line fitted over the whole window).
    Returns cumulative oil and water ADDED after t[k_cut] at t_out: (dNo, dNw).
    """
    C = np.asarray(C, float)[: k_cut + 1]
    tt = t[: k_cut + 1]
    q = quarter_rates(C[None], tt)[0]
    kf = _window(k_cut, n_fit)
    t0 = tt[k_cut]
    Np0 = C[k_cut, OIL]
    if method == "elton":
        raise ValueError("use forecast_elton for the baseline method")
    qo, qw = q[kf, OIL], q[kf, WATER]
    a, c = fit_wor(C[kf, OIL], qw / np.maximum(qo, 1e-9))
    if wor_anchor == "last":
        k2 = kf[-2:]
        a = float(np.log(q[k2, WATER].sum() / q[k2, OIL].sum()) - c * C[k2, OIL].mean())
    if method == "arps_wor":
        qi, Di, b = fit_arps(tt[kf], qo)
        if anchor_rates is not None:
            qi = anchor_rates[0]
        oil_fn = lambda x: arps_rate(qi, Di, b, x, d_min_year)
        dNo = _integrate(oil_fn, t0, t_out)
        if anchor_rates is not None:                      # shift the WOR line through the anchor WOR
            a = np.log(anchor_rates[1] / anchor_rates[0]) - c * Np0
        x = np.arange(0, t_out[-1] - t0 + 5.0, 5.0)
        qo_x = oil_fn(x)
        No_x = np.concatenate([[0], np.cumsum(0.5 * (qo_x[1:] + qo_x[:-1]) * 5.0)])
        qw_x = qo_x * np.exp(np.minimum(a + c * (Np0 + No_x), 50.0))
        Nw_x = np.concatenate([[0], np.cumsum(0.5 * (qw_x[1:] + qw_x[:-1]) * 5.0)])
        return dNo, np.interp(t_out - t0, x, Nw_x)
    ql = qo + qw
    if method == "liq_wor":
        q0, g = fit_liquid_trend(tt[kf], ql)
        if anchor_rates is not None:
            q0 = sum(anchor_rates)
            a = np.log(anchor_rates[1] / anchor_rates[0]) - c * Np0
        return _march_split(lambda x: q0 * np.exp(-g * x), a, c, Np0, t0, t_out)
    if method == "crm_wor":
        tau, G = fit_crm(tt, quarter_rates(C[None], tt)[0][:, OIL] + quarter_rates(C[None], tt)[0][:, WATER], inj, kf)
        target = G * inj_future
        own_fn = lambda x: target + (ql[-1] - target) * np.exp(-x / tau)
        if anchor_rates is None:
            return _march_split(own_fn, a, c, Np0, t0, t_out)
        a_obs = np.log(anchor_rates[1] / anchor_rates[0]) - c * Np0
        if anchor_fade_days is None:
            q_end = sum(anchor_rates)
            return _march_split(lambda x: target + (q_end - target) * np.exp(-x / tau), a_obs, c, Np0, t0, t_out)
        dq, da, f = sum(anchor_rates) - ql[-1], a_obs - a, float(anchor_fade_days)
        return _march_split(lambda x: own_fn(x) + dq * np.exp(-x / f),
                            lambda x: a + da * np.exp(-x / f), c, Np0, t0, t_out)
    raise ValueError(method)


# ------------------------------------------------------------------ Elton's baseline (ported)

def _family_eval(name, z, tt, c_last, t_last):
    import torch
    import torch.nn.functional as F
    if name == "arps_cum":       # Arps hyperbolic rates -> closed-form cumulative
        q0 = (0.05 + F.softplus(z[:, 0])) * 4.0 * (c_last / t_last)
        Di = (0.05 + F.softplus(z[:, 1])) * 4.0 / t_last
        b = 0.2 + 2.0 * torch.sigmoid(z[:, 2])
        x = 1.0 + b[:, None] * Di[:, None] * tt[None]
        a = (1.0 - 1.0 / b)[:, None]
        den = Di[:, None] * (b[:, None] - 1.0)
        den = torch.where(b[:, None] > 1.0, den.clamp(min=1e-12), den.clamp(max=-1e-12))
        arps = q0[:, None] * (x.pow(a) - 1.0) / den
        lin = q0[:, None] * torch.log1p(Di[:, None] * tt[None]) / Di[:, None]
        return torch.where((torch.abs(b - 1.0) < 1e-3)[:, None], lin, arps)
    if name == "gompertz":
        K = (0.2 + F.softplus(z[:, 0])) * 2.0 * c_last
        bb = (0.05 + F.softplus(z[:, 1])) * 4.0 / t_last
        t0 = z[:, 2] * 0.3 * t_last
        return K[:, None] * torch.exp(-torch.exp(-bb[:, None] * (tt[None] - t0[:, None])))
    raise ValueError(name)


def forecast_elton(C, t, k_cut, t_out, steps=1200, lr=0.05):
    """Elton's method for a batch of curves C (n, T, 3): Arps-cumulative (oil) and Gompertz (water)
    fitted to the whole cumulative curve up to k_cut, shifted to be continuous at the cutoff,
    forced monotone. Returns added oil and water (n, len(t_out)) after t[k_cut]."""
    import torch
    out = []
    for j, fam in [(OIL, "arps_cum"), (WATER, "gompertz")]:
        Cj = torch.as_tensor(np.asarray(C, np.float32)[:, : k_cut + 1, j])
        tt = torch.as_tensor(np.asarray(t[: k_cut + 1], np.float32))
        c_last, t_last = Cj[:, -1].clamp(min=1e-9), float(tt[-1])
        z = torch.zeros((Cj.shape[0], 3), requires_grad=True)
        opt = torch.optim.Adam([z], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.1)
        for _ in range(steps):
            opt.zero_grad()
            loss = (((_family_eval(fam, z, tt, c_last, t_last) - Cj) / c_last[:, None]) ** 2).mean()
            loss.backward(); opt.step(); sched.step()
        with torch.no_grad():
            fut = _family_eval(fam, z, torch.as_tensor(np.asarray(t_out, np.float32)), c_last, t_last).numpy()
            at_cut = _family_eval(fam, z, tt[-1:], c_last, t_last).numpy()[:, 0]
        add = np.maximum.accumulate(fut - at_cut[:, None], axis=1)
        out.append(np.maximum(add, 0.0))
    return out[0], out[1]


def hindcast_start(Y, truth, t, inj, k_cut, k_end, exclude=None, n_fit=12, wor_anchor="last", top_n=10, w0=4,
                   fades=(365,)):
    """Blind start-point hindcast for one truth curve (T, 3) and one cut-off.

    The simulated cases Y (n, T, 3) are ranked by the oil + water cumulative misfit to the truth on
    dates w0..k_cut only (the truth itself excluded); the best top_n are forecast to t[k_end] with
    crm_wor, starting from (main) their own state, (A1) the truth's cumulative, (A2) the truth's
    cumulative and rates, (fade_f) the truth's rates fading to the case's own path over f days.
    Returns {option: (oil, water) error of the mean of the top_n on t[k_end], % of the truth's
    produced volume after the cut-off}.
    """
    K = np.arange(w0, k_cut + 1)
    R = (Y[:, K][:, :, [OIL, WATER]] - truth[K][:, [OIL, WATER]]) / truth[k_cut, [OIL, WATER]]
    s = np.sqrt((R ** 2).mean(axis=(1, 2)))
    if exclude is not None:
        s[exclude] = np.inf
    top = np.argsort(s)[:top_n]
    tout = t[k_cut + 1: k_end + 1]
    qt = quarter_rates(truth[None, : k_cut + 1], t[: k_cut + 1])[0]
    obs = (qt[k_cut, OIL], qt[k_cut, WATER])
    cum_obs = truth[k_cut, [OIL, WATER]]

    def added(C, **kw):
        return np.array(forecast_case("crm_wor", C, t, k_cut, tout, n_fit, inj, inj[k_cut], wor_anchor=wor_anchor, **kw))[:, -1]

    own = np.array([added(Y[i]) for i in top])
    end = {"main": Y[top, k_cut][:, [OIL, WATER]] + own, "A1": cum_obs + own,
           "A2": cum_obs + np.array([added(Y[i], anchor_rates=obs) for i in top])}
    for f in fades:
        end[f"fade_{f}"] = cum_obs + np.array([added(Y[i], anchor_rates=obs, anchor_fade_days=f) for i in top])
    produced = truth[k_end, [OIL, WATER]] - cum_obs
    return {k: (v.mean(axis=0) - truth[k_end, [OIL, WATER]]) / produced * 100 for k, v in end.items()}


def gor_at_cut(C, t, k_cut, n=4):
    """Gas-oil ratio of the last n quarters (rate based)."""
    q = quarter_rates(np.asarray(C, float)[None, : k_cut + 1], t[: k_cut + 1])[0]
    return float(q[k_cut - n + 1: k_cut + 1, GAS].sum() / q[k_cut - n + 1: k_cut + 1, OIL].sum())
