# Task 0: Agents

Status: approved (13 Oct 2026). Source: PRD §4.

## 1. Inputs and outputs

| Output | Content |
| --- | --- |
| `CLAUDE.md` | Rules V1 to V13, handoff, definition of done, commands, key facts. Both agents read it |
| `.claude/agents/researcher.md` | Researcher agent: specs, references, decisions, reviews |
| `.claude/agents/data-scientist.md` | Data-scientist agent: code, tests, configs, runs |
| `docs/specs/`, `docs/decisions.md`, `docs/references.md` | Researcher's files |
| `.gitignore` | Keeps raw data, processed data and outputs out of version control (V12) |

## 2. Roles

| Item | Researcher | Data scientist |
| --- | --- | --- |
| Purpose | Make each method correct | Make each method run, tested and reproducible |
| Writes | `docs/` only | `src/`, `tests/`, `configs/`, run logs |
| Tools | Read, Grep, Glob, WebSearch, WebFetch, Write, Edit | Read, Write, Edit, Bash, Grep, Glob |
| Checks | Spec compliance, V1 to V13, plausibility of results | `pytest`, data checks, run from a clean clone |

The PRD gives the researcher no Write or Edit tool, but it also asks the researcher to write files in `docs/`. Decision D-002 resolves this.

## 3. Handoff

1. Researcher writes `docs/specs/taskN.md`.
2. Data scientist codes on branch `taskN`, tests, and logs the run.
3. Researcher reviews and returns numbered findings (blocker, required, note).
4. Any blocker returns the work to step 2. Zero blockers: merge to `main`, then log the decision.

Spec template: inputs and outputs (shapes, units); equations, each with "source: <article>" or "assumption"; defaults and tuning ranges; acceptance tests.

## 4. Acceptance tests

- [x] Both agent files exist in `.claude/agents/`, with `name`, `description` and `tools` in the front matter.
- [x] `CLAUDE.md` lists V1 to V13 and the commands.
- [x] `.gitignore` excludes `*.xlsx`, `data/raw/`, `data/processed/` and `outputs/`.
- [x] Both agents load in the session as agent types `researcher` and `data-scientist`.
- [x] The researcher writes `docs/specs/task1.md` from PRD §5 (first use of the handoff). Full cycle completed: spec, build, review, merge (D-011).
