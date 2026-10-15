# Input folder of `notebook3_hist_match.ipynb`

`notebook2_proxy_model.ipynb` writes every file here. Do not edit them by hand: rerun notebook 2 instead. Then run all cells of notebook 3. The results appear in `hist_match_output/`.

The data is confidential (DWC26). Do not share it outside the team.

## Files

| File | Rows | Columns | What it is |
|---|---|---|---|
| `simulated.csv` | 100 cases x 41 dates | `case_id`, `split`, `date`, `gas_cum`, `oil_cum`, `water_cum` | The **simulator** curves of the 100 cases on the quarterly dates 1998-01-01 to 2008-01-01. Cumulative values in real units: STB for oil and water, MSCF for gas. Notebook 3 scores these against the history (method A) and offers them as candidates |
| `predictions.csv` | 100 x 41 | same as `simulated.csv` | The **proxy's** curves of the same 100 cases, computed the way notebook 3 calls the proxy (from the parameters, with the grid mapper). Used to check that the proxy ranks the cases like the simulator |
| `case_params.csv` | 100 | `case_id`, `split`, `fault_trans`, `poro_mult`, `perm_mult`, `aquifer_pv` | The 4 uncertainty parameters of each case, in real units. `split` is train (1-70), validation (71-85) or test (86-100) |
| `cumulative_model.pt` | - | - | The trained cumulative proxy: Elton's model, **improved** (adopted 15 Oct 2026, notebook 2 section 6b): 20 networks trained on oil and water of the 37 producing dates (gas = GOR x oil), window scaler, grid mapper, train bounds. Load it with `proxy_lib.CumulativeProxy`. The previous version (Elton's twin, 5 networks) is in `archive/v1_baseline_2026-10-15/hist_match_input/` |
| `proxy_meta.json` | - | - | The model's sha256 hash, configuration, train range of the parameters, matching window, observed final cumulatives and **`sigma_proxy`** (the proxy's error, see below) |
| `proxy_error_by_date.csv` | 37 | `date`, `oil_cum`, `gas_cum`, `water_cum` | The proxy's RMS error on each matching date, over the validation cases, as a share of the observed cumulative at 2008-01-01 |

## How notebook 3 uses the proxy

```python
import proxy_lib as pl                       # src/proxy_lib.py
proxy = pl.CumulativeProxy("hist_match_input/", window_dates, ("oil_cum", "water_cum"))
curves = proxy(theta)          # theta (m, 4) real units, order fault_trans, poro_mult, perm_mult, aquifer_pv
                               # -> (m, 37, 2) cumulative oil and water [STB] on the window dates
proxy.sigma_proxy              # (2,) proxy error per quantity, share of the final observed value
proxy.full(theta)              # dict gas_cum / oil_cum / water_cum, each (m, 41): all dates
```

The loader stops with an error if `cumulative_model.pt` does not match the hash in `proxy_meta.json`. Rerun notebook 2 in that case.

## Rules

* `sigma_proxy` comes from the validation cases (71-85) only. The test cases are reported in notebook 2, not used.
* The proxy is valid inside the **train range** of the 4 parameters (`train_min`, `train_max` in `proxy_meta.json`). Notebook 3 keeps its search inside it.
* Gas is in the files but is not matched. It is about 0.3633 x oil in every case.
