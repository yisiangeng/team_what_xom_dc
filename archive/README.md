# archive: restore points of the pipeline

Each subfolder is a **frozen copy** of the full pipeline at one moment, taken just before a change that might not have worked. If the change had failed, or if the team wants to go back, the copy can be restored. **Do not edit the files inside the subfolders.** Keep them until the final submission. They are not needed to run the current pipeline.

| Folder | Taken | Before which change | Contents |
|---|---|---|---|
| `v1_baseline_2026-10-15/` | 15 Oct 2026 | **The proxy improvement.** This is version 1, which uses Elton's cumulative proxy as is. It was replaced by the improved proxy (version 2) | Notebooks 2 and 3, `src/proxy_lib.py`, `proxy_output/`, `hist_match_input/`, `hist_match_output/`, the history-matching summary and the notebook builders. There was no forecast yet |
| `v2_pre_bias_fix_2026-10-15/` | 15 Oct 2026 | **The forecast start-point fix.** This is version 2, where each case's forecast starts from its own 2008 values. It was replaced by version 3, which starts from the observed 2008 values | Notebooks 2, 3 and 4, `src/` (proxy and forecast code), all four output folders including the version 2 deliverable, both summaries, the action plan and the builders |

Each subfolder's `README.md` lists the settings and key results of that version (for example its chosen 10, posterior and forecast numbers) and gives a copy-paste **revert command** (PowerShell, run from the repo root).

## The versions in short

| Version | Proxy | History match: the 10 | Forecast start | Oil in 2028 (mean of the 10) |
|---|---|---|---|---|
| v1 (archived) | Elton's cumulative twin, 5 seeds | 5 new sets + cases 6, 76, 11, 40, 73 | - (no forecast yet) | - |
| v2 (archived) | Improved, 20 seeds | Case 6, 5 new sets, cases 97, 76, 11, 40 | Each case's own 2008 values | 61.2 MMSTB |
| **v3 (current, repo root)** | Improved, 20 seeds | Same as v2 | Observed 2008 values | **58.7 MMSTB** |

Why each change was made, and the evidence for it:
- **Proxy:** `summary_hist_mat.md`, section 7.
- **Forecast start:** `summary_forecast.md`, section 5.

## Notes

- **Old file names.** The archived summaries keep the names they had then: `hist_mat_summary.md` and `forecast_summary.md`. The revert commands copy them back under the current names, `summary_hist_mat.md` and `summary_forecast.md`.
- **Old folder paths.** Paths inside the archived notebooks refer to the folder layout of that time. For example, `Elton Model/` is now `our_work/proxy_model_elton/`. The executed outputs are complete, so nothing needs rerunning to read them.
- **Confidential.** The archive contains models and outputs derived from the confidential DWC26 data. Do not share it outside the team.
