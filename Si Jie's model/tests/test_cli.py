"""dwc26.cli: subcommands and exit codes (E28 for Task 3)."""

from __future__ import annotations

import sqlite3

import pytest

from dwc26.cli import main


@pytest.mark.parametrize(
    "argv",
    [
        ["match", "--proxy", "x", "--config", "configs/hm.yaml"],
        ["forecast", "--matched", "x", "--config", "configs/forecast.yaml"],
        ["all"],
    ],
)
def test_not_implemented(argv, capsys):
    assert main(argv) == 2
    assert "not implemented" in capsys.readouterr().err


def test_prepare_ok(synthetic_raw, capsys):
    assert main(["prepare", "--config", str(synthetic_raw.config_path())]) == 0
    run_id = capsys.readouterr().out.strip().splitlines()[-1]
    con = sqlite3.connect(synthetic_raw.cfg["paths"]["db"])
    assert con.execute("SELECT status FROM runs WHERE run_id = ?", (run_id,)).fetchone() == ("ok",)
    con.close()


def test_prepare_requires_config():
    with pytest.raises(SystemExit) as ei:
        main(["prepare"])
    assert ei.value.code == 2


def test_train_requires_model_and_config():
    for argv in (["train", "--config", "configs/models/b0.yaml"], ["train", "--model", "b0"]):
        with pytest.raises(SystemExit) as ei:
            main(argv)
        assert ei.value.code == 2


def test_train_config_error_exit_1(tmp_path, capsys):
    assert main(["train", "--model", "b0", "--config", str(tmp_path / "missing.yaml")]) == 1
    assert "train failed" in capsys.readouterr().err


# --- E28: Task 3 commands ------------------------------------------------------------------------

def test_e28_config_defaults_to_evaluate_yaml():
    from pathlib import Path

    from dwc26.cli import _parser

    p = _parser()
    for argv in (["evaluate", "--split", "val"], ["evaluate", "--split", "test"], ["backtest"], ["select"],
                 ["freeze", "--model", "r/A"]):
        assert p.parse_args(argv).config == Path("configs/evaluate.yaml")
    args = p.parse_args(["freeze", "--model", "r/A", "--secondary", "r/main", "--override-reason", "a long reason"])
    assert (args.model, args.secondary, args.override_reason) == ("r/A", "r/main", "a long reason")


def test_e28_override_reason_only_where_allowed(tmp_path, capsys):
    with pytest.raises(SystemExit) as ei:
        main(["evaluate", "--split", "val", "--override-reason", "x", "--config", str(tmp_path / "e.yaml")])
    assert ei.value.code == 2
    with pytest.raises(SystemExit) as ei:
        main(["backtest", "--override-reason", "a long enough reason"])
    assert ei.value.code == 2
    with pytest.raises(SystemExit) as ei:
        main(["evaluate", "--split", "train"])
    assert ei.value.code == 2
    with pytest.raises(SystemExit) as ei:
        main(["freeze"])                                     # --model is required
    assert ei.value.code == 2


def test_e28_short_override_reason_exit_1(tmp_path, capsys):
    from synth_eval import tiny_eval_cfg_dict, write_eval_cfg

    t2 = {"db": str(tmp_path / "x.sqlite"), "models": str(tmp_path / "m"),
          "runs": {"b0": "r0", "b1": "r1", "lstm": "r2", "dmd": "r3"}, "model_cfgs": {"lstm": "l.yaml", "dmd": "d.yaml"}}
    cfg = write_eval_cfg(tiny_eval_cfg_dict(t2, tmp_path), tmp_path / "evaluate.yaml")
    for argv in (["select", "--override-reason", "short"], ["freeze", "--model", "r/A", "--override-reason", "  x  "],
                 ["evaluate", "--split", "test", "--override-reason", "too short"]):
        assert main([*argv, "--config", str(cfg)]) == 1
        assert "override-reason" in capsys.readouterr().err
    assert not (tmp_path / "x.sqlite").exists()              # refused before any run row


def test_e28_config_error_exit_1(tmp_path, capsys):
    assert main(["select", "--config", str(tmp_path / "missing.yaml")]) == 1
    assert "select failed" in capsys.readouterr().err
