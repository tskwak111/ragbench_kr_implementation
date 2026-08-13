# Task 1 Report: Repository Bootstrap, Typed Configuration, and CI

## Implementation

- Added Python 3.12 project metadata and a committed `uv.lock` with Pydantic v2,
  pydantic-settings, pytest, Ruff, and mypy development dependencies.
- Added `Settings`, which loads environment configuration, permits a missing
  `UPSTAGE_API_KEY` for offline use, and rejects live mode without that key.
  Defaults are a `Decimal("135.00")` budget, concurrency `5`, retries `5`, and
  no gold access. It also contains database, cache/data, Upstage endpoint/model,
  and Embed 2 promotion-deadline settings.
- Added frozen `VersionBundle` for code/config/data reproducibility identifiers.
- Added safe `.env.example`, local ignore rules, pre-commit hooks, pytest markers,
  strict mypy/Ruff configuration, and Python 3.12 CI. CI runs a tracked-file
  secret guard before Ruff, mypy, and non-live/non-gold tests.
- `PriceBook.from_yaml` and `configs/prices.yaml` were intentionally deferred to
  Task 3, per the resolved task boundary.

## TDD evidence

1. Wrote `tests/unit/core/test_config.py` before production package code.
2. RED command: `uv --cache-dir /private/tmp/ragbench-task1-uv-cache run --python 3.12 pytest tests/unit/core/test_config.py -v`
   - Expected result observed: `ModuleNotFoundError: No module named 'ragbench'`.
3. Added the minimum package, settings, and version implementation.
4. GREEN command: the same focused pytest command.
   - Result: `4 passed in 0.14s`.

## Verification

| Command | Result |
| --- | --- |
| `.venv/bin/ruff check .` | `All checks passed!` |
| `.venv/bin/mypy src/ragbench` | `Success: no issues found in 4 source files` |
| `.venv/bin/pytest -m 'not live and not gold' -q` | `4 passed in 0.04s` |
| tracked `UPSTAGE_API_KEY` guard | passed; no nonempty tracked assignment |
| `git diff --check` | passed |

No network or paid provider API calls were made by tests. Initial dependency resolution
was sandbox-network blocked; it was rerun with explicit approval to download declared
development dependencies only.

## Files changed

- `.gitignore`
- `.env.example`
- `.pre-commit-config.yaml`
- `.github/workflows/ci.yml`
- `pyproject.toml`
- `uv.lock`
- `src/ragbench/__init__.py`
- `src/ragbench/core/__init__.py`
- `src/ragbench/core/config.py`
- `src/ragbench/core/versions.py`
- `tests/unit/core/test_config.py`

## Self-review

- Confirmed the live API key validator is conditional on the explicit live flag and
  accepts missing credentials for offline commands.
- Confirmed monetary defaults use `Decimal`, tool rules match the requested Ruff and
  mypy settings, and CI excludes both `live` and `gold` markers.
- Corrected the secret guard to avoid false positives from the task-plan prose while
  still rejecting nonempty assignment values, including quoted values.
- Confirmed no credentials appear in new tracked files and no whitespace errors exist.

## Concerns

- None. Exact current provider model IDs must still be verified before live use in
  Task 4, as prescribed by the implementation plan.

## Commit

- `chore: bootstrap typed ragbench project`

## Fix Round 1: CI secret guard whitespace bypass

### Root cause and implementation

The original value expression required the first character after `=` to be
alphanumeric, `_`, or `-`. Therefore, `UPSTAGE_API_KEY= secret` did not match.
The guard now anchors to a dotenv assignment line, permits whitespace around the
key and `=`, and fails when the first non-whitespace value character is not `#`.
Empty and comment-only assignments remain allowed.

### Covering executable behavior check

RED command:

```sh
pattern="UPSTAGE_API_KEY=([[:alnum:]_-][^[:space:]#]*|\"[^\"]+\"|'[^']+')"; if printf '%s\n' 'UPSTAGE_API_KEY= secret' | grep -qE "$pattern"; then echo 'unexpected current guard match'; exit 1; else echo 'RED reproduced: current guard misses whitespace-prefixed nonempty value'; exit 1; fi
```

Output and exit: `RED reproduced: current guard misses whitespace-prefixed nonempty value` (exit 1).

GREEN command:

```sh
pattern='^[[:space:]]*UPSTAGE_API_KEY[[:space:]]*=[[:space:]]*[^[:space:]#]'; printf '%s\n' 'UPSTAGE_API_KEY= secret' '  UPSTAGE_API_KEY = "secret"' | grep -qE "$pattern" && ! printf '%s\n' 'UPSTAGE_API_KEY=' ' UPSTAGE_API_KEY =    ' 'UPSTAGE_API_KEY= # intentionally unset' | grep -qE "$pattern" && ! git grep -nE "$pattern" && echo 'secret guard behavior: whitespace-prefixed values rejected; empty/comment-only assignments allowed'
```

Output and exit: `secret guard behavior: whitespace-prefixed values rejected; empty/comment-only assignments allowed` (exit 0).

### Relevant global checks

- `.venv/bin/ruff check .`
- `.venv/bin/mypy src/ragbench`
- `.venv/bin/pytest -m 'not live and not gold' -q`
- `git diff --check`
