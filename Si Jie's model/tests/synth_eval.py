"""Fixtures for the Task 3 tests (spec task3 section 4.0).

- build_tiny_task2: synthetic_task1_db (n 30, T 13, k0 2; train 1-18, val 19-24, test 25-30)
  and run_train for b0, b1, dmd and lstm with tiny_cfg.
- tiny_eval_cfg_dict / tiny_eval_cfg: a valid Task 3 config for that database.
- make_git_repo: a temporary git repository with docs/decisions.md and a placeholder under
  each frozen path. Tests never touch a tag of the project repository.
- FakeGit: an in-memory GitOps.
- insert_run: a plain `runs` row.
"""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path

import yaml

from synth_models import synthetic_task1_db, tiny_cfg

FROZEN_PATHS = ["src/dwc26/evaluation", "src/dwc26/models", "src/dwc26/forecast/backtest.py",
                "src/dwc26/forecast/dates.py", "src/dwc26/hm/objective.py", "configs/evaluate.yaml", "configs/models"]
TINY_MODELS = ("b0", "b1", "dmd", "lstm")
T_CUT, H_BT, N_FC = 9, 4, 8


def build_tiny_task2(root: Path, models=TINY_MODELS) -> dict:
    """Task 1 database and tiny Task 2 runs in root (paths in the run configs point into root)."""
    from dwc26.models.train import run_train

    root = Path(root)
    s = synthetic_task1_db(root)
    runs, cfgs = {}, {}
    for m in models:
        path = tiny_cfg(root, m)
        cfgs[m] = path
        runs[m] = run_train(path, m)
    return {**s, "root": root, "runs": runs, "model_cfgs": cfgs, "models": root / "models"}


def tiny_eval_cfg_dict(t2: dict, out: Path, repo: Path | None = None) -> dict:
    out = Path(out)
    proxies = [m for m in ("rnn", "lstm", "dmd") if m in t2["runs"]]
    strategy = {"rnn": "S3", "lstm": "S3", "dmd": "S4"}
    return {
        "seed": 0,
        "paths": {"db": str(t2["db"]), "models": str(t2["models"]), "reports": str(out / "reports"),
                  "figures": str(out / "figures"), "logs": str(out / "logs")},
        "candidates": {"runs": dict(t2["runs"]), "require_clean": False},
        "strategy": {m: strategy[m] for m in proxies},
        "runtime": {"torch_threads": 1},
        "consistency_rel_tol": 1.0e-7,
        "metrics": {"objective_fluids": ["oil", "water"], "coverage_z": 1.959963984540054,
                    "coverage_target": [0.90, 0.98]},
        "events": {"plateau_frac": 0.98, "breakthrough_frac": 0.01},
        "mismatch": {"window": ["2000-07-01", "2003-01-01"], "fluids": ["oil", "water"], "alpha": 0.10,
                     "beta": 0.02, "local_n": 2},
        "continuation": {"end_date": "2005-01-01"},
        "plausibility": {"p2_mode": "reject", "p2_ratio": 1.05, "p2_case_share": 0.5, "p3_hist_quarters": 4,
                         "p3_fc_quarters": [3, 6], "p3_ratio_min": 0.5, "p3_min_decline": 0.002,
                         "gor_rel_tol": 0.005, "gor_oil_frac": 0.01},
        "backtest": {"cutoff_date": "2002-01-01", "models": proxies,
                     "model_configs": {m: str(t2["model_cfgs"][m]) for m in proxies}, "v6_horizons": [3, 4],
                     "reuse": {}},
        "selection": {"band": 0.05, "secondary_only": ["dmd"]},
        "cost": {"n_rows": 50, "repeats": 1},
        "freeze": {"tag": "freeze-proxy", "repo_dir": str(repo if repo is not None else out / "no_repo"),
                   "decisions_path": "docs/decisions.md", "frozen_paths": list(FROZEN_PATHS),
                   "min_override_chars": 10},
        "report": {"plots": False, "dpi": 60},
    }


def set_key(cfg: dict, dotted: str, value) -> dict:
    d = cfg
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value
    return cfg


def write_eval_cfg(cfg: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def tiny_eval_cfg(t2: dict, out: Path, repo: Path | None = None, **overrides) -> Path:
    """overrides: dotted keys with '__' for '.', e.g. report__plots=True."""
    cfg = copy.deepcopy(tiny_eval_cfg_dict(t2, out, repo))
    for k, v in overrides.items():
        set_key(cfg, k.replace("__", "."), v)
    return write_eval_cfg(cfg, Path(out) / "evaluate.yaml")


# --- git -----------------------------------------------------------------------------------------

def git_available() -> bool:
    return shutil.which("git") is not None


def git(repo: Path, *args: str) -> str:
    p = subprocess.run(["git", "-c", "user.name=E24", "-c", "user.email=e24@example.invalid", "-c",
                        "commit.gpgsign=false", "-c", "tag.gpgsign=false", *args], cwd=repo,
                       capture_output=True, text=True, encoding="utf-8")
    if p.returncode != 0:
        raise RuntimeError(f"git {args} failed: {p.stderr}")
    return p.stdout.strip()


def make_git_repo(path: Path) -> Path:
    """git init, docs/decisions.md, a placeholder under each frozen path, one commit."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-q")
    (path / "docs").mkdir(exist_ok=True)
    (path / "docs" / "decisions.md").write_text("# Decisions\n\nNo choice yet.\n", encoding="utf-8")
    for p in FROZEN_PATHS:
        target = path / p
        if target.suffix:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f"placeholder {p}\n", encoding="utf-8")
        else:
            target.mkdir(parents=True, exist_ok=True)
            (target / "placeholder.txt").write_text(f"placeholder {p}\n", encoding="utf-8")
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "initial")
    return path


class FakeGit:
    """In-memory GitOps: settable tag, head, ancestry, dirty flag, changed paths and file contents."""

    def __init__(self, tag_commit: str | None = "t1", head: str = "t1", dirty: bool = False):
        self.tag_commit = tag_commit
        self.head_commit = head
        self.dirty = dirty
        self.changed: list[str] = []
        self.files: dict[tuple[str, str], str] = {}
        self.ancestors: set[tuple[str, str]] = set()     # (a, b) pairs with a an ancestor of b

    def head(self) -> str:
        return self.head_commit

    def resolve_tag(self, tag: str) -> str | None:
        return self.tag_commit

    def is_ancestor(self, a: str, b: str) -> bool:
        return a == b or (a, b) in self.ancestors

    def is_dirty(self) -> bool:
        return self.dirty

    def changed_paths(self, a: str, b: str, paths: list[str]) -> list[str]:
        return list(self.changed)

    def show(self, rev: str, path: str) -> str | None:
        return self.files.get((rev, path))


# --- runs rows -----------------------------------------------------------------------------------

_SEQ = [0]


def insert_run(con, run_id: str | None = None, *, task: str = "task3", command: str = "evaluate",
               model: str | None = None, config: dict | None = None, outputs: dict | None = None,
               status: str = "ok", git_commit: str = "c1", git_dirty: int | None = 0,
               start: str | None = None) -> str:
    _SEQ[0] += 1
    run_id = run_id or f"{command}-20990101T0000{_SEQ[0]:02d}000000Z"
    start = start or f"2099-01-01T00:00:{_SEQ[0] % 60:02d}.{_SEQ[0]:06d}Z"
    con.execute("INSERT INTO runs (run_id, task, command, model, config, config_path, config_sha256, seed, git_commit, "
                "git_dirty, start, duration_s, status, outputs, python, packages) VALUES "
                "(?, ?, ?, ?, ?, 'synthetic', '0', 0, ?, ?, ?, 1.0, ?, ?, '3.13', '{}')",
                (run_id, task, command, model, json.dumps(config or {}, sort_keys=True), git_commit, git_dirty, start,
                 status, json.dumps(outputs or {}, sort_keys=True)))
    con.commit()
    return run_id


# --- fake models and split arrays (unit tests) ------------------------------------------------------

class FakeProxy:
    """A model whose raw curves are given: rollout(X, n) returns curves[:, :n] clipped at 0 (one row of
    X for each case, in order). Benchmarks (is_proxy False) do not continue and have no std."""

    def __init__(self, curves, T_fit: int, k0: int = 0, std=None, is_proxy: bool = True, p: int = 4):
        import numpy as np

        self.curves = np.asarray(curves, dtype=np.float64)
        self.T_fit, self.k0, self.p = int(T_fit), int(k0), int(p)
        self.std = None if std is None else np.asarray(std, dtype=np.float64)
        self.extrapolates_time = is_proxy
        self.last_n_clipped = 0

    def _clip(self, a):
        import numpy as np

        return np.where(a < 0, 0.0, a), int((a < 0).sum())

    def rollout(self, X, n_total):
        if X.shape[0] != self.curves.shape[0]:
            raise ValueError("FakeProxy: one X row for each case")
        if n_total > self.T_fit and not self.extrapolates_time:
            raise NotImplementedError("benchmark")
        out, k = self._clip(self.curves[:, :n_total])
        self.last_n_clipped = k
        return out

    def predict(self, X):
        return self.rollout(X, self.T_fit)

    def forecast(self, X, n_steps):
        if not self.extrapolates_time:
            raise NotImplementedError("benchmark")
        out, k = self._clip(self.curves[:, self.T_fit:self.T_fit + n_steps])
        self.last_n_clipped = k
        return out

    def predict_std(self, X):
        return None if self.std is None else self.std.copy()


def split_arrays(Y_phys, *, split="val", y_factor=(1.0, 1.0, 1.0), X=None, case_id=None, dates=None, hist=None):
    """A SplitArrays built directly (no database)."""
    import numpy as np

    from dwc26.evaluation.splits import SplitArrays

    Y_phys = np.asarray(Y_phys, dtype=np.float64)
    n, T = Y_phys.shape[:2]
    dates = (np.arange(np.datetime64("2000-01"), np.datetime64("2000-01") + 3 * T, np.timedelta64(3, "M"))
             .astype("datetime64[D]") if dates is None else dates)
    return SplitArrays(split=split, case_id=np.arange(1, n + 1, dtype=np.int64) if case_id is None else case_id,
                       X=np.full((n, 4), 0.5) if X is None else X, Y_phys=Y_phys, dates=dates,
                       t_days=(dates - dates[0]).astype(np.float64),
                       hist_phys=Y_phys[0].copy() if hist is None else hist,
                       y_factor=np.asarray(y_factor, dtype=np.float64), prepare_run_id="p", arrays_sha256="s")
