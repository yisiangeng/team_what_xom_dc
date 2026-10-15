# improve_proxy: proxy improvement tests, 15 Oct 2026

The outcome is recorded in `summary_hist_mat.md`, section 7. The adopted variant now lives in `src/proxy_lib.py` and notebook 2 (section 6b), so this folder is no longer needed to run the pipeline. It is kept as the evidence behind proxy version 2.

| File | What |
|---|---|
| `notebook3_proxy_variants_test.ipynb` | Part A: 8 proxy variants compared (CV + validation), learning curve, single test. Part B: a copy of notebook 3 run on the chosen proxy. Part C: comparison with the baseline |
| `followup_runner_up.py` | The check that led to adoption: out-of-fold error over cases 1-85 and the test cases for V0, V1, V4 (5 and 20 seeds). Run from the repo root |
| `significance_test.py` | Paired per-case test of baseline vs adopted (Wilcoxon, bootstrap CI): `results/significance_per_case.csv`, `results/significance_summary.csv` |
| `proxy_variants.py` | Variant code (targets, monotone output, GP, averages) on top of `src/proxy_lib.py` |
| `results/` | `variant_comparison.csv`, `v4_random_search.csv`, `learning_curve.csv`, `test_once.csv`, `followup_runner_up.csv`, `hm_comparison.csv`, `figs/` |
| `hist_match_input_test/`, `hist_match_output_test/` | Input and output of Part B. The notebook's rule winner failed the test, so this is the baseline proxy with an out-of-fold proxy error |

Baseline restore point: `archive/v1_baseline_2026-10-15/`.
