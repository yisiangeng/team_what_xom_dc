# improve_forecast: 2008 rate-bias fix tests, 15 Oct 2026

**Restore point:** `archive/v2_pre_bias_fix_2026-10-15/`, with a revert command in its README.

**The question.** In 2008 the history-matched cases produce oil 11-15 % too fast and water about 10 % too slowly. Does starting the forecast from the observed 2008 state, instead of each case's own state, forecast better?

**Outcome.** Yes. Starting from the observed cumulatives and rates (A2) halved the hindcast error on 100 pseudo-fields (p ~ 1e-36) and cut it from 6.5 % to 3.8 % on the real field. It was adopted in `notebook4_forecast.ipynb` (section 4 reruns the test) and written up in `summary_forecast.md`, section 5.

| File | What |
|---|---|
| `hindcast_start_test.py` | Full hindcast. Run from the repo root. Besides the starts in the notebook, it adds:<ul><li>`fade_180`</li><li>the simulator's own continuation (`sim`, `sim_shift`)</li><li>the data-only forecast (`field_only`)</li><li>errors on the 2008 cumulative</li><li>coverage, the A2-vs-A1 test, and the match rate gap at each cut-off</li></ul> |
| `results/hindcast_all.csv` | Every truth × cut-off × start |
| `results/hindcast_summary.csv`, `hindcast_tests_vs_main.csv`, `hindcast_cumulative_error.csv` | Summaries and paired tests |
