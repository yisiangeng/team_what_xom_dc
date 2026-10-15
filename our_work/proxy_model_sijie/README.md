# DWC26 code repository

Proxy of the reservoir simulator trained on cases 1 to 70, history-matched with
MCMC (ten matched sets), and continued to 1 January 2028. Requirements:
[PRD.md](PRD.md). Rules V1 to V13 and the agent handoff: [CLAUDE.md](CLAUDE.md).
Method specs: `docs/specs/`.

The data and this repository are not public (V12). The repository has no remote.

## Setup

Python 3.13 on Windows 11 (PowerShell or Git Bash). Run every command from the
repository root.

```bash
python -m venv .venv
# PowerShell:  .venv\Scripts\Activate.ps1
# Git Bash:    source .venv/Scripts/activate
python -m pip install -r requirements-lock.txt   # exact environment, including torch (CPU)
python -m pip install --no-deps -e .
```

`requirements-lock.txt` is `pip freeze --exclude-editable` of the working venv:
every package, including the transitive ones, at its exact version (V7). It
names the official PyTorch CPU index (`--extra-index-url
https://download.pytorch.org/whl/cpu`), where `torch==2.14.1+cpu` comes from.

`pyproject.toml` pins the direct dependencies: numpy 2.3.5, pandas 2.3.3,
openpyxl 3.1.5, scipy 1.17.1 and pyyaml 6.0.3; pytest 9.1.1 in the extra `dev`;
torch 2.14.1+cpu and scikit-learn 1.7.2 in the extra `models` (Task 2 on);
matplotlib 3.10.8 in the extra `eval` (Task 3 figures). Without the lock file:

```bash
python -m pip install -e ".[dev,models,eval]" --extra-index-url https://download.pytorch.org/whl/cpu
```

Task 1 (`prepare`) does not import torch or scikit-learn.

## Raw data

Copy the two input files into `data/raw/` (git-ignored, never committed):

```text
data/raw/01 Introduction.xlsx
data/raw/02 Field Production.xlsx
```

Grid files 03 to 08 are out of scope.

## Commands

| Step | Command | Status |
| --- | --- | --- |
| Task 1: data preparation | `python -m dwc26 prepare --config configs/data.yaml` | Built |
| Task 2: proxy training | `python -m dwc26 train --model <m> --config configs/models/<m>.yaml` | Built |
| Task 3: validation | `python -m dwc26 evaluate --split val`, `backtest`, `select`, `freeze --model <proxy_id>`, `evaluate --split test` | Built |
| Task 4: history matching | `python -m dwc26 match --proxy <run_id> --config configs/hm.yaml` | Not built |
| Task 5: forecast | `python -m dwc26 forecast --matched <run_id> --config configs/forecast.yaml` | Not built |
| All steps | `python -m dwc26 all` | Not built |

### Task 1: `prepare`

Spec: `docs/specs/task1.md`. Settings: `configs/data.yaml`. Run time: about
15 s on a laptop.

The command reads files 01 and 02, runs the data checks C1 to C10, aligns
every case and `RAMP_History` to the history dates with one rate definition
(R1, selected by C6), scales X and Y, and writes:

| Output | Content |
| --- | --- |
| `data/processed/dwc26.sqlite` | Tables `runs`, `parameters`, `production_native`, `production_grid`, `history`, `scalers`, `data_checks` |
| `data/processed/arrays.npz` | `X` (n, 4), `Y` (n, T, 3), `hist` (T, 3), `dates`, physical arrays, scaler values; load with `np.load(path, allow_pickle=False)` |
| `outputs/reports/data_checks.md` | Header (run ID, git commit, input hashes), check table, one section per check |
| `outputs/logs/<run_id>.log` | Run log |

Exit code 0 on success (the run ID is printed), 1 if a stop check fails. Each
run adds one row to `runs` (V8). Later tasks read the scalers with
`dwc26.data.scale.load_scalers(con)` and the arrays with
`dwc26.data.db.read_task1_arrays(con)` or from `arrays.npz`.

### Task 2: `train`

Spec: `docs/specs/task2.md`. Settings: `configs/models/{b0,b1,dmd,rnn,lstm}.yaml`.
After `prepare`, train the two benchmarks first (the proxy reports compare
against them), then the three proxies:

```bash
python -m dwc26 train --model b0   --config configs/models/b0.yaml
python -m dwc26 train --model b1   --config configs/models/b1.yaml
python -m dwc26 train --model dmd  --config configs/models/dmd.yaml
python -m dwc26 train --model rnn  --config configs/models/rnn.yaml
python -m dwc26 train --model lstm --config configs/models/lstm.yaml
```

Measured run times (laptop, 8 cores, no GPU; `n_jobs: 6`, `torch_threads: 1`):
b0 0.1 s, b1 1 s, dmd 147 s (24 grid points: rank x weights x GP target,
real-eigenvalue basis by variable projection, amendment A2), rnn 460 s,
lstm 136 s (30 trials and 12 extra ensemble members each). The results do not
depend on `n_jobs` (test M17). DMD selects the lowest validation J among the
grid points whose 80-step validation continuation clips no value (gate P1).

Each run prints its run ID and writes:

| Output | Content |
| --- | --- |
| `outputs/models/<run_id>/` | `config.yaml`, `trials.csv`, `histories/` (neural), `report.md`, and one folder for each variant with `meta.json` and the weights (`weights.pt`, `dmd.npz` + `gps.pkl`, `b0.npz`, `b1.npz`) |
| `predictions` table | Train and validation predictions in physical units (no test rows, V3) |
| `runs` table | One row (V8), with the metrics of each variant in `outputs` |
| `outputs/logs/<run_id>.log` | Run log |

A fitted model is named by its proxy ID `<run_id>/<variant>` (variants `A`, `B`,
`C` for rnn and lstm, `main` otherwise). Load it with
`dwc26.models.registry.load_model(proxy_id, "outputs/models", con=con)`.
Benchmarks are refused unless `proxies_only=False`. The Task 5 backtest uses
`--t-cut N --settings-from <run_id>`.

### Task 3: `evaluate`, `backtest`, `select`, `freeze`

Spec: `docs/specs/task3.md`. Settings: `configs/evaluate.yaml` (the default
`--config` of all four commands). The figures need the extra `eval`
(matplotlib 3.10.8):

```bash
python -m pip install -e ".[dev,models,eval]" --extra-index-url https://download.pytorch.org/whl/cpu
```

Run in this order, after the Task 2 runs:

```bash
python -m dwc26 evaluate --split val   # all candidates (latest ok full-window train run of each model), train and val metrics
python -m dwc26 backtest               # retrains rnn, lstm and dmd with --t-cut and --settings-from; validation cases only
python -m dwc26 select                 # proposes the primary and the secondary proxy
```

Measured run times (laptop, 8 cores, no GPU; `n_jobs: 6` and `torch_threads: 1`
in the model configs): `evaluate --split val` 71 s, `backtest` 290 s (rnn 229 s,
lstm 42 s and dmd 16 s of retraining at t_cut 29), `select` under 1 s.

The freeze and the single test scoring run only after the review approves the
choice (spec section 0.1, rule V3): the researcher records both proxy IDs in
`docs/decisions.md`, the main session tags that commit `freeze-proxy`, then

```bash
python -m dwc26 freeze --model <primary proxy ID> --secondary <secondary proxy ID>
python -m dwc26 evaluate --split test   # once; guards G1 to G7 refuse otherwise (exit code 3)
```

`freeze` never creates or moves a tag. A second test scoring needs
`--override-reason` and a researcher decision.

| Output | Content |
| --- | --- |
| `metrics` table | Every metric, keyed by run, proxy ID (or `b2`), split, fluid, metric and index (horizon or date) |
| `selection`, `freeze`, `test_scoring` tables | The proposed choice, the frozen choice and the test scoring, with SQLite triggers for rule V3 |
| `outputs/reports/model_comparison.md` | All models and splits, V6, backtest, plausibility (P1, P2, P3, GOR), events, mismatch, coverage, cost, training flags, sigma_proxy for Task 4 |
| `outputs/reports/selection.md` | Step table, ranking trace, secondary rule, proposal and the freeze commands |
| `outputs/figures/task3/` | Curves of each validation case (oil, gas, water), parity grid, backtest RMSE by horizon; the test figures after the freeze |
| `outputs/logs/<run_id>.log` | Run log |

Exit codes: 0 success, 1 configuration or contract error, 2 usage error,
3 a V3 refusal (the guard ID is printed).

## Tests

```bash
python -m pytest                        # default run: every test except those marked slow
python -m pytest -m "not rawdata"       # synthetic tests only (no raw files needed)
python -m pytest -m slow                # slow tests only (M39: 5,000 predict calls per proxy)
python -m pytest -m "slow or not slow"  # everything
```

Tests marked `rawdata` run the pipeline on the real files and are skipped when
`data/raw/` does not hold them. `pyproject.toml` sets `addopts = "-m 'not slow'"`.

## Layout

```text
configs/data.yaml        Task 1 settings
configs/models/          Task 2 settings, one YAML for each model
configs/evaluate.yaml    Task 3 settings
requirements-lock.txt    exact package versions of the venv (V7)
src/dwc26/cli.py         command-line entry point (python -m dwc26)
src/dwc26/config.py      config loading and validation
src/dwc26/data/          load, check, align, scale, split, db, prepare (Task 1)
src/dwc26/models/        base, registry, dataset, tuning, seq, rnn, lstm, dmd, baselines, train (Task 2)
src/dwc26/evaluation/    metrics, events, plausibility, splits, store, guards, evaluate, selection,
                         freeze, report, plots (Task 3)
src/dwc26/forecast/      dates, backtest (Task 3; Task 5 reuses them)
src/dwc26/hm/            objective: the mismatch O(x) (Task 3; Task 4 reuses it)
tests/                   unit and pipeline tests; tests/synth.py builds synthetic input files,
                         tests/synth_models.py synthetic curves, databases and model configs,
                         tests/synth_eval.py the Task 3 fixtures (tiny Task 2 runs, git repository)
docs/                    specs, decisions, references (researcher)
data/, outputs/          generated, git-ignored
```
