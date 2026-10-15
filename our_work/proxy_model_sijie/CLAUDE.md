# DWC26 Code Repository

Proxy of the reservoir simulator trained on cases 1 to 70, history-matched with MCMC (ten matched sets), and continued to 1 January 2028. Full requirements: [PRD.md](PRD.md). Code due 16 October 2026.

## Agents and handoff

Two subagents divide the work (definitions in `.claude/agents/`):

| Agent | Writes | Does not |
| --- | --- | --- |
| `researcher` | `docs/specs/taskN.md`, `docs/decisions.md`, `docs/references.md`, review findings | Write or edit anything outside `docs/` |
| `data-scientist` | `src/`, `tests/`, `configs/`, run logs | Change a spec without researcher approval |

1. Researcher writes `docs/specs/taskN.md`.
2. Data scientist implements it on branch `taskN`, writes the tests, runs `pytest`, logs the run.
3. Researcher reviews against the spec and V1 to V13. Findings are numbered and labelled **blocker**, **required** or **note**.
4. Any blocker returns the work to step 2. Zero blockers: the main session merges to `main` and the researcher logs the decision in `docs/decisions.md`.

Each spec has four sections: inputs and outputs (shapes, units); equations, each with "source: <article>" or "assumption"; defaults and tuning ranges; acceptance tests.

### Definition of done for a merge

- [ ] The code agrees with the spec.
- [ ] `pytest` passes.
- [ ] Rules V1 to V13 are checked.
- [ ] One command reproduces the result from a clean clone.
- [ ] The researcher approves with zero blockers.

## Validity rules

| ID | Rule |
| --- | --- |
| V1 | Fit each learned transform (Y scalers, DMD basis, model weights) on cases 1 to 70 only. Select hyperparameters on cases 71 to 85 only |
| V2 | Split by case. Never split by row or by date |
| V3 | Use cases 86 to 100 for metrics one time only, after the model choice is frozen and tagged `freeze-proxy` |
| V4 | Apply the same alignment, rate definition and scaling to the cases and to `RAMP_History` |
| V5 | Do not hard-code data values. Bounds, dates and counts come from the files at run time |
| V6 | Each model must beat B0 (mean training curve) and B1 (kNN in X) on validation, and B2 (persistence) in the backtest |
| V7 | Fix all random seeds. Keep all settings in YAML configs. Pin package versions |
| V8 | Log each run: run ID, git commit, config, seed, duration, outputs |
| V9 | Assert shapes and values at each module boundary: no NaN, rates ≥ 0, expected array shapes |
| V10 | Write a unit test for each module. `pytest` must pass before a merge |
| V11 | Test the history-matching code with a parameter-recovery test (Task 4) |
| V12 | Do not publish the data or the repository. Keep raw files out of version control |
| V13 | Each proxy keeps the time component. Step 5 continues the Step 2 model. No other forecast method is used |

## Key facts

- Split: train cases 1 to 70, validation 71 to 85, test 86 to 100.
- Inputs: file 01 (4 parameters: `fault_trans`, `poro_mult`, `perm_mult`, `aquifer_pv`) and file 02 (rates, cumulatives, `RAMP_History`). Raw files live in `data/raw/` and are never committed. Grid files 03 to 08 are out of scope.
- Time grid: 41 quarterly history dates. Match window: 1 Jan 1999 to 1 Jan 2008 (37 dates). Forecast: 80 quarterly dates, 1 Apr 2008 to 1 Jan 2028.
- Array contracts: `X` (n, 4) scaled to [0, 1]; `Y` (n, 41, 3) as [oil, gas, water], scaled; `hist` (41, 3), same scaling; `dates` (41,) `datetime64[D]`.
- Every proxy implements the `Proxy` protocol in PRD §3 (`fit`, `predict`, `predict_std`, `forecast`, `save`, `load`, `extrapolates_time = True`).
- One SQLite database, `data/processed/dwc26.sqlite`, is the store for all tasks. Tables are listed in PRD §5.

## Commands

```bash
python -m dwc26 prepare  --config configs/data.yaml
python -m dwc26 train    --model lstm --config configs/models/lstm.yaml [--t-cut N] [--settings-from RUN_ID]
python -m dwc26 evaluate --split val
python -m dwc26 freeze   --model <proxy_id>
python -m dwc26 evaluate --split test
python -m dwc26 match    --proxy <proxy_id> --config configs/hm.yaml
python -m dwc26 forecast --matched <run_id> --config configs/forecast.yaml
python -m dwc26 all      # all steps from the raw files
python -m pytest
```

A fitted proxy is named by its proxy ID, `<run_id>/<variant>`, for example `train-20261013T120000000000Z/B` (D-014). The variant is the RNN/LSTM input option (`A`, `B`, `C`) or `main`.

## Environment

Windows 11, Python 3.13. Both PowerShell and Git Bash are available. The repository has no remote and must not get one (V12).
