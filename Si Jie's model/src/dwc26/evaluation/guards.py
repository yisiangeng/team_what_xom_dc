"""V3 guards of the freeze and the single test scoring (spec task3 sections 1.10 and 2.10; D-028).

| ID | Guard |
| G1 | the tag `freeze.tag` exists |
| G2 | HEAD contains the tag commit |
| G3 | the working tree is clean |
| G4 | no file under `freeze.frozen_paths` differs between the tag commit and HEAD (test only) |
| G5 | freeze: the decision text and the lineage; test: an active freeze row with the current tag commit |
| G6 | once only (override with a logged reason) |
| G7 | the test run's metric config blocks equal those of the frozen chain E, B and S |

Every git failure refuses (fail closed). SQLite triggers enforce G6 and the test-row guards
even if this module is bypassed (section 1.6).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from dwc26.config import G7_BLOCKS
from dwc26.data.check import ContractError


class GuardError(ContractError):
    """A V3 refusal. `guard` is the guard ID (G1 to G7)."""

    def __init__(self, guard: str, message: str):
        super().__init__(f"{guard}: {message}")
        self.guard = guard


def package_dir() -> Path:
    """The directory of this module: the start of the search for the repository when
    freeze.repo_dir is null."""
    return Path(__file__).resolve().parent


class GitOps:
    """Read-only git queries of one repository. It never creates, moves or deletes a tag."""

    def __init__(self, repo_dir: Path | None):
        if repo_dir is None:
            # R1: find the top level from the package directory, then run every git command there, because
            # `git diff -- <paths>` and `git ls-tree <rev> -- <path>` read paths relative to the cwd.
            self.cwd = package_dir()
            top = Path(self._top())
            if not top.is_dir():
                raise GuardError("G1", f"git: the top level {top} of the package repository is not a directory")
            self.cwd = top
        else:
            self.cwd = Path(repo_dir)
            if not self.cwd.is_dir():
                raise GuardError("G1", f"git: {self.cwd} is not a directory")
            top = self._top()
            try:
                same = os.path.samefile(top, self.cwd)
            except OSError:
                same = False
            if not same:
                raise GuardError("G1", f"git: {self.cwd} is not the top level of a git repository (top level {top})")

    def _run(self, *args: str, ok=(0,), guard: str = "G1") -> subprocess.CompletedProcess:
        """Runs git; a missing git, a timeout or an exit code outside `ok` refuses (fail closed)."""
        try:
            p = subprocess.run(["git", *args], cwd=self.cwd, capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=60)
        except (OSError, subprocess.SubprocessError) as e:
            raise GuardError(guard, f"git {' '.join(args)} failed: {e}") from e
        if p.returncode not in ok:
            raise GuardError(guard, f"git {' '.join(args)} exited {p.returncode}: {p.stderr.strip()}")
        return p

    def _top(self) -> str:
        return self._run("rev-parse", "--show-toplevel").stdout.strip()

    def head(self) -> str:
        return self._run("rev-parse", "--verify", "HEAD^{commit}").stdout.strip()

    def resolve_tag(self, tag: str) -> str | None:
        p = self._run("rev-parse", "-q", "--verify", f"refs/tags/{tag}^{{commit}}", ok=(0, 1))
        return p.stdout.strip() if p.returncode == 0 and p.stdout.strip() else None

    def is_ancestor(self, a: str, b: str) -> bool:
        return self._run("merge-base", "--is-ancestor", a, b, ok=(0, 1), guard="G2").returncode == 0

    def is_dirty(self) -> bool:
        return bool(self._run("status", "--porcelain", "--untracked-files=no", guard="G3").stdout.strip())

    def changed_paths(self, a: str, b: str, paths: list[str]) -> list[str]:
        out = self._run("diff", "--name-only", a, b, "--", *paths, guard="G4").stdout
        return [line.strip() for line in out.splitlines() if line.strip()]

    def show(self, rev: str, path: str) -> str | None:
        listed = self._run("ls-tree", "--name-only", rev, "--", path, guard="G5").stdout.strip()
        if not listed:
            return None
        return self._run("show", f"{rev}:{path}", guard="G5").stdout


@dataclass(frozen=True)
class TestTicket:
    __test__ = False           # not a pytest class
    scoring_id: int
    run_id: str                # the evaluate-test run
    freeze_id: int
    tag_commit: str


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def check_freeze_tag(git: GitOps, cfg: dict) -> tuple[str, str]:
    """G1 to G3 -> (tag_commit, head)."""
    tag = cfg["freeze"]["tag"]
    tag_commit = git.resolve_tag(tag)
    if tag_commit is None:
        raise GuardError("G1", f"the tag {tag!r} does not exist; the main session tags the frozen commit "
                               "after the researcher records the choice (section 0.1)")
    head = git.head()
    if not git.is_ancestor(tag_commit, head):
        raise GuardError("G2", f"HEAD {head[:12]} does not contain the tag commit {tag_commit[:12]}")
    if git.is_dirty():
        raise GuardError("G3", "the working tree has uncommitted changes to tracked files")
    return tag_commit, head


def active_freeze(con) -> dict | None:
    row = con.execute("SELECT freeze_id, run_id, selection_run, primary_id, secondary_id, tag, tag_commit, "
                      "fallback_secondary, override_reason FROM freeze ORDER BY freeze_id DESC LIMIT 1").fetchone()
    if row is None:
        return None
    keys = ("freeze_id", "run_id", "selection_run", "primary_id", "secondary_id", "tag", "tag_commit",
            "fallback_secondary", "override_reason")
    return dict(zip(keys, row))


def run_record(con, run_id: str) -> dict:
    row = con.execute("SELECT run_id, command, status, config, outputs, git_commit, git_dirty, start, duration_s, "
                      "config_path, config_sha256 FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        raise ContractError(f"no run {run_id} in the runs table")
    keys = ("run_id", "command", "status", "config", "outputs", "git_commit", "git_dirty", "start", "duration_s",
            "config_path", "config_sha256")
    d = dict(zip(keys, row))
    d["config"] = json.loads(d["config"] or "{}")
    d["outputs"] = json.loads(d["outputs"] or "{}")
    return d


def frozen_chain(con, freeze_row: dict) -> dict:
    """{'S': select run, 'E': evaluate run, 'B': backtest run} records of the frozen choice."""
    S = run_record(con, freeze_row["selection_run"])
    E = run_record(con, S["outputs"]["evaluate_run"])
    B = run_record(con, S["outputs"]["backtest_run"])
    return {"S": S, "E": E, "B": B}


def g7_mismatch(cfg: dict, records: dict[str, dict]) -> list[str]:
    """The (run, block) pairs whose config block differs from cfg (json equality)."""
    bad = []
    for name, rec in records.items():
        for block in G7_BLOCKS:
            if json.dumps(rec["config"].get(block), sort_keys=True) != json.dumps(cfg.get(block), sort_keys=True):
                bad.append(f"{name} ({rec['run_id']}).{block}")
    return bad


def claim_test_scoring(con, run_id: str, cfg: dict, git: GitOps, override_reason: str | None) -> TestTicket:
    """G1 to G7, then the claim row (status 'running'), committed before any test row is read."""
    tag_commit, head = check_freeze_tag(git, cfg)                                    # G1 to G3
    changed = git.changed_paths(tag_commit, head, list(cfg["freeze"]["frozen_paths"]))
    if changed:                                                                      # G4
        raise GuardError("G4", f"frozen paths changed since the tag commit {tag_commit[:12]}: {changed}")
    fr = active_freeze(con)                                                          # G5
    if fr is None:
        raise GuardError("G5", "no freeze row: run `freeze` after the tag")
    if fr["tag_commit"] != tag_commit:
        raise GuardError("G5", f"the active freeze row {fr['freeze_id']} has tag commit {fr['tag_commit'][:12]}, "
                               f"the tag now points to {tag_commit[:12]} (the tag was moved)")
    n_prev = con.execute("SELECT COUNT(*) FROM test_scoring").fetchone()[0]          # G6
    if n_prev and override_reason is None:
        raise GuardError("G6", f"the test split was already scored ({n_prev} scoring row(s), a failed attempt "
                               "counts); a second scoring needs --override-reason and a researcher decision")
    bad = g7_mismatch(cfg, frozen_chain(con, fr))                                    # G7
    if bad:
        raise GuardError("G7", f"config blocks differ from the frozen chain: {bad}")
    cur = con.execute(
        "INSERT INTO test_scoring (run_id, freeze_id, tag_commit, head_commit, started, finished, status, "
        "override_reason) VALUES (?, ?, ?, ?, ?, NULL, 'running', ?)",
        (run_id, fr["freeze_id"], tag_commit, head, _now(), override_reason))
    con.commit()
    return TestTicket(scoring_id=int(cur.lastrowid), run_id=run_id, freeze_id=int(fr["freeze_id"]),
                      tag_commit=tag_commit)


def verify_ticket(con, ticket: TestTicket) -> None:
    """The claim row exists, has status 'running' and the same run, freeze and tag commit."""
    if not isinstance(ticket, TestTicket):
        raise GuardError("G6", "the test split needs a test-scoring ticket (claim_test_scoring)")
    row = con.execute("SELECT run_id, freeze_id, tag_commit, status FROM test_scoring WHERE scoring_id = ?",
                      (ticket.scoring_id,)).fetchone()
    if row is None:
        raise GuardError("G6", f"no test_scoring row {ticket.scoring_id}: the ticket is forged")
    if row[3] != "running":
        raise GuardError("G6", f"test_scoring row {ticket.scoring_id} has status {row[3]!r}, not 'running'")
    if (row[0], int(row[1]), row[2]) != (ticket.run_id, ticket.freeze_id, ticket.tag_commit):
        raise GuardError("G6", f"test_scoring row {ticket.scoring_id} does not match the ticket")


def finish_test_scoring(con, ticket: TestTicket, status: str) -> None:
    """Sets the claim to 'ok' or 'failed' and commits (the trigger makes it final)."""
    if status not in ("ok", "failed"):
        raise ContractError(f"finish_test_scoring: status must be 'ok' or 'failed', got {status!r}")
    cur = con.execute("UPDATE test_scoring SET status = ?, finished = ? WHERE scoring_id = ? AND run_id = ?",
                      (status, _now(), ticket.scoring_id, ticket.run_id))
    if cur.rowcount != 1:
        raise ContractError(f"finish_test_scoring: no test_scoring row {ticket.scoring_id} for run {ticket.run_id}")
    con.commit()
