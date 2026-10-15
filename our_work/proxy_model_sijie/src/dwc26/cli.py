"""Command-line entry point: python -m dwc26 <command> (CLAUDE.md, Commands).

Exit codes: 0 success (the run ID is printed); 1 ConfigError or ContractError; 2 usage
errors and the commands that are not built yet; 3 a V3 refusal (GuardError, with its
guard ID G1 to G7; spec task3 section 1.4).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

NOT_BUILT = ("match", "forecast", "all")
EVAL_CONFIG = Path("configs/evaluate.yaml")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m dwc26", description="DWC26 pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Task 1: data preparation")
    p.add_argument("--config", required=True, type=Path, help="path to configs/data.yaml")
    t = sub.add_parser("train", help="Task 2: proxy and benchmark training")
    t.add_argument("--model", required=True, help="rnn, lstm, dmd, b0 or b1 (must equal 'model' in the config)")
    t.add_argument("--config", required=True, type=Path, help="path to configs/models/<model>.yaml")
    t.add_argument("--t-cut", dest="t_cut", type=int, default=None,
                   help="fit the first N history dates only (Task 5 backtest)")
    t.add_argument("--settings-from", dest="settings_from", default=None,
                   help="run_id of an earlier ok run of the same model: reuse its tuned settings, no tuning")
    e = sub.add_parser("evaluate", help="Task 3: metrics on the validation split, or once on the test split")
    e.add_argument("--split", required=True, choices=("val", "test"))
    e.add_argument("--config", type=Path, default=EVAL_CONFIG, help="default configs/evaluate.yaml")
    e.add_argument("--override-reason", dest="override_reason", default=None,
                   help="--split test only: a second test scoring (needs a researcher decision)")
    b = sub.add_parser("backtest", help="Task 3: retrain at the cutoff and compare with B2 (validation cases)")
    b.add_argument("--config", type=Path, default=EVAL_CONFIG, help="default configs/evaluate.yaml")
    s = sub.add_parser("select", help="Task 3: propose the primary and the secondary proxy")
    s.add_argument("--config", type=Path, default=EVAL_CONFIG, help="default configs/evaluate.yaml")
    s.add_argument("--override-reason", dest="override_reason", default=None,
                   help="select after a freeze (needs a researcher decision)")
    f = sub.add_parser("freeze", help="Task 3: verify the tag and record the frozen choice")
    f.add_argument("--model", required=True, help="proxy ID <run_id>/<variant> of the primary")
    f.add_argument("--secondary", default=None, help="proxy ID of the secondary (default: the selection's)")
    f.add_argument("--config", type=Path, default=EVAL_CONFIG, help="default configs/evaluate.yaml")
    f.add_argument("--override-reason", dest="override_reason", default=None,
                   help="a second freeze or a choice that differs from the selection")
    for name in NOT_BUILT:
        sub.add_parser(name, help="not implemented yet")
    return parser


def _console_logging() -> None:
    from our_work.proxy_model_sijie.src.dwc26.data.prepare import LOG_FORMAT, UTCFormatter

    console = logging.StreamHandler()
    console.setFormatter(UTCFormatter(LOG_FORMAT))
    logging.basicConfig(level=logging.INFO, handlers=[console])


def _task3(args, parser) -> int:
    from our_work.proxy_model_sijie.src.dwc26.config import ConfigError
    from our_work.proxy_model_sijie.src.dwc26.data.check import ContractError
    from our_work.proxy_model_sijie.src.dwc26.evaluation.guards import GuardError

    if args.command == "evaluate" and args.split == "val" and args.override_reason is not None:
        parser.error("--override-reason is accepted only with --split test, select and freeze")
    _console_logging()
    try:
        if args.command == "evaluate":
            from our_work.proxy_model_sijie.src.dwc26.evaluation.evaluate import run_evaluate

            run_id = run_evaluate(args.config, args.split, override_reason=args.override_reason)
        elif args.command == "backtest":
            from our_work.proxy_model_sijie.src.dwc26.forecast.backtest import run_backtest

            run_id = run_backtest(args.config)
        elif args.command == "select":
            from our_work.proxy_model_sijie.src.dwc26.evaluation.selection import run_select

            run_id = run_select(args.config, override_reason=args.override_reason)
        else:
            from our_work.proxy_model_sijie.src.dwc26.evaluation.freeze import run_freeze

            run_id = run_freeze(args.config, args.model, args.secondary, override_reason=args.override_reason)
    except GuardError as e:
        print(f"{args.command} refused (V3, guard {e.guard}): {e}", file=sys.stderr)
        return 3
    except (ConfigError, ContractError) as e:
        print(f"{args.command} failed: {e}", file=sys.stderr)
        return 1
    print(run_id)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args, extra = parser.parse_known_args(argv)
    if args.command in NOT_BUILT:
        print(f"{args.command}: not implemented", file=sys.stderr)
        return 2
    if extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")

    if args.command == "prepare":
        from our_work.proxy_model_sijie.src.dwc26.data.check import DataCheckError
        from our_work.proxy_model_sijie.src.dwc26.data.prepare import run_prepare

        _console_logging()
        try:
            run_id = run_prepare(args.config)
        except DataCheckError as e:
            print(f"prepare stopped: {e}", file=sys.stderr)
            return 1
        print(run_id)
        return 0

    if args.command == "train":
        from our_work.proxy_model_sijie.src.dwc26.config import ConfigError
        from our_work.proxy_model_sijie.src.dwc26.data.check import ContractError
        from our_work.proxy_model_sijie.src.dwc26.models.train import run_train

        _console_logging()
        try:
            run_id = run_train(args.config, args.model, t_cut=args.t_cut, settings_from=args.settings_from)
        except (ConfigError, ContractError) as e:
            print(f"train failed: {e}", file=sys.stderr)
            return 1
        print(run_id)
        return 0

    if args.command in ("evaluate", "backtest", "select", "freeze"):
        return _task3(args, parser)
    return 2  # unreachable: argparse rejects unknown commands
