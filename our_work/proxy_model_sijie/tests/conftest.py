from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from our_work.proxy_model_sijie.tests.synth import SynthRaw, build_raw, write_config

REPO = Path(__file__).resolve().parents[1]
RAW01 = REPO / "data" / "raw" / "01 Introduction.xlsx"
RAW02 = REPO / "data" / "raw" / "02 Field Production.xlsx"


@pytest.fixture
def synthetic_raw(tmp_path) -> SynthRaw:
    """Synthetic files 01 and 02 with the real layout, and a config whose
    paths all lie in tmp_path (spec task1 section 4)."""
    return build_raw(tmp_path)


@pytest.fixture(scope="session")
def real_prepared(tmp_path_factory):
    """Runs run_prepare once on the real files into a temporary directory."""
    if not (RAW01.is_file() and RAW02.is_file()):
        pytest.skip("raw files 01 and 02 are not in data/raw/")
    from dwc26.data.prepare import run_prepare

    root = tmp_path_factory.mktemp("real")
    cfg = yaml.safe_load((REPO / "configs" / "data.yaml").read_text(encoding="utf-8"))
    cfg = copy.deepcopy(cfg)
    cfg["paths"].update(
        file01=str(RAW01),
        file02=str(RAW02),
        db=str(root / "processed" / "dwc26.sqlite"),
        arrays=str(root / "processed" / "arrays.npz"),
        report=str(root / "outputs" / "reports" / "data_checks.md"),
        logs=str(root / "outputs" / "logs"),
    )
    cfg_path = write_config(cfg, root / "data.yaml")
    run_id = run_prepare(cfg_path)
    return {"run_id": run_id, "cfg": cfg, "root": root}


# --- Task 3 (spec task3 section 4.0) ---------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_task2_db(tmp_path_factory):
    """synthetic_task1_db plus run_train for b0, b1, dmd and lstm with tiny_cfg (module scope)."""
    from our_work.proxy_model_sijie.tests.synth_eval import build_tiny_task2

    return build_tiny_task2(tmp_path_factory.mktemp("t2"))


@pytest.fixture
def tiny_eval_cfg(tiny_task2_db, tmp_path):
    """Writes a valid Task 3 config for tiny_task2_db into tmp_path and returns its path."""
    from our_work.proxy_model_sijie.tests.synth_eval import tiny_eval_cfg as make

    def _make(repo=None, **overrides):
        return make(tiny_task2_db, tmp_path, repo, **overrides)

    return _make


@pytest.fixture
def tmp_git_repo(tmp_path):
    """A temporary git repository (docs/decisions.md and the frozen-path placeholders, one commit)."""
    from our_work.proxy_model_sijie.tests.synth_eval import git_available, make_git_repo

    if not git_available():
        pytest.skip("git is not on PATH")
    return make_git_repo(tmp_path / "repo")


@pytest.fixture
def clean_git(monkeypatch):
    """runs rows record git_dirty 0, so that select and freeze accept the runs of a test (spec E24)."""
    from dwc26.data import db

    commit, _ = db.git_state()
    monkeypatch.setattr(db, "git_state", lambda: (commit, 0))
    return commit
