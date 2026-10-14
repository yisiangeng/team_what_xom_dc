"""Run lifecycle shared by the Task 3 commands (spec task3 section 1.5).

Every command: loads the config, runs `connect` and `init_schema`, calls
`start_run(task='task3', command=...)` and opens `outputs/logs/<run_id>.log`.
On any exception after `start_run` and before `finish_run(ok)` it rolls back the
open transaction, runs the registered failure callbacks (for example
`finish_test_scoring(..., 'failed')`), calls `finish_run(status='failed')` and
re-raises.

After `finish_run(ok)` (spec task3 section 0.5, R2): the figures and the reports
run through `ctx.post`. A failure there is logged at ERROR and the next step
still runs. The run status and every outputs key stay; the errors are stored as
`outputs.post_finish_error`, and the command then raises PostFinishError (exit 1).
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import json

from dwc26.data import db
from dwc26.data.check import ContractError
from dwc26.data.prepare import LOG_FORMAT, UTCFormatter

LOG = logging.getLogger("dwc26.task3")


class PostFinishError(ContractError):
    """A figure or a report failed after finish_run(ok); the run stays ok (R2). Exit code 1."""


@dataclass
class RunCtx:
    con: object
    run_id: str
    log_path: Path
    cfg: dict
    config_path: Path
    on_fail: list = field(default_factory=list)      # callables run after the rollback of a failure
    finished: bool = False
    post_errors: list = field(default_factory=list)  # failures of the steps after finish_run(ok)

    def finish(self, outputs: dict) -> None:
        """finish_run(status='ok'), which commits the open transaction."""
        outputs = dict(outputs)
        outputs.setdefault("files", {})["log"] = self.log_path.as_posix()
        db.finish_run(self.con, self.run_id, "ok", outputs)
        self.finished = True

    def post(self, name: str, fn, *args, **kw):
        """Runs one step after finish_run(ok) (a figure or a report). A failure is logged at ERROR and
        recorded; it does not stop the next step (R2)."""
        if not self.finished:
            raise RuntimeError(f"post step {name!r} before finish_run")
        try:
            return fn(*args, **kw)
        except Exception as e:  # noqa: BLE001 - recorded, raised at the end of the run
            LOG.exception("run %s: %s failed after finish_run(ok): %r", self.run_id, name, e)
            self.post_errors.append(f"{name}: {type(e).__name__}: {e}")
            return None


def _record_post_error(con, run_id: str, errors: list[str]) -> str:
    """Adds outputs.post_finish_error to the finished run; the status and the other keys stay."""
    text = "; ".join(errors)
    row = con.execute("SELECT outputs FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    outputs = json.loads(row[0] or "{}") if row else {}
    if outputs.get("post_finish_error"):
        text = f"{outputs['post_finish_error']}; {text}"
    outputs["post_finish_error"] = text
    con.execute("UPDATE runs SET outputs = ? WHERE run_id = ?", (json.dumps(outputs, sort_keys=True), run_id))
    con.commit()
    return text


def _open_log(log_dir: Path, run_id: str) -> logging.Handler:
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_dir / f"{run_id}.log", encoding="utf-8")
    handler.setFormatter(UTCFormatter(LOG_FORMAT))
    handler.setLevel(logging.INFO)
    LOG.addHandler(handler)
    LOG.setLevel(logging.INFO)
    return handler


@contextlib.contextmanager
def task3_run(cfg: dict, config_path: Path, command: str):
    """Yields a RunCtx. The caller calls ctx.finish(outputs) inside the block."""
    paths = cfg["paths"]
    con = db.connect(Path(paths["db"]))
    handler = None
    try:
        db.init_schema(con)
        run_id = db.start_run(con, task="task3", command=command, cfg=cfg, config_path=Path(config_path),
                              seed=cfg["seed"], model=None)
        log_dir = Path(paths["logs"])
        handler = _open_log(log_dir, run_id)
        ctx = RunCtx(con=con, run_id=run_id, log_path=log_dir / f"{run_id}.log", cfg=cfg,
                     config_path=Path(config_path))
        row = con.execute("SELECT git_commit, git_dirty, config_sha256 FROM runs WHERE run_id = ?",
                          (run_id,)).fetchone()
        LOG.info("run %s: %s, config %s (sha256 %s), git commit %s, dirty %s", run_id, command,
                 Path(config_path).as_posix(), row[2], row[0], row[1])
        try:
            yield ctx
            if not ctx.finished:
                raise RuntimeError(f"{command}: the run ended without finish_run")
        except BaseException as e:
            if ctx.finished:                                  # R2: after finish_run(ok)
                LOG.exception("run %s: error after finish_run(ok); the status stays ok: %r", run_id, e)
                con.rollback()
                text = _record_post_error(con, run_id, [f"{type(e).__name__}: {e}"])
                if isinstance(e, Exception):
                    raise PostFinishError(f"{command} {run_id}: the run is ok, but a step after it failed: "
                                          f"{text}") from e
                raise
            LOG.exception("run %s failed: %r", run_id, e)
            con.rollback()
            for cb in ctx.on_fail:
                try:
                    cb()
                except Exception as e2:  # noqa: BLE001 - keep the original error
                    LOG.error("failure callback failed: %r", e2)
            db.finish_run(con, run_id, "failed", {"error": str(e), "error_type": type(e).__name__,
                                                  "guard": getattr(e, "guard", None),
                                                  "files": {"log": ctx.log_path.as_posix()}})
            raise
        if ctx.post_errors:
            text = _record_post_error(con, run_id, ctx.post_errors)
            LOG.error("run %s is ok, but a step after finish_run(ok) failed: %s", run_id, text)
            raise PostFinishError(f"{command} {run_id}: the run is ok, but a step after it failed: {text}")
    finally:
        if handler is not None:
            LOG.removeHandler(handler)
            handler.close()
        con.close()
