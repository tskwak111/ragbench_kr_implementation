# Final Verification Record

Verification date: 2026-08-14 (Asia/Seoul)

Task 18 base commit: `f09db57`

Scope: locally executable gates only; no provider, gold, or private corpus access

## Verified locally

| Gate | Evidence | Status |
|---|---|---|
| Locked install | `uv --cache-dir /private/tmp/ragbench-task18-uv-cache sync --frozen --all-groups` | PASS, 57 packages checked |
| Ruff lint | `uv ... run ruff check .` | PASS |
| Ruff format | `uv ... run ruff format --check .` | PASS, 155 files formatted |
| Strict type check | `uv ... run mypy src/ragbench` | PASS, 65 source files |
| Offline/non-gold tests | `uv ... run pytest -m 'not live and not gold' -q` | PASS, 458 passed and 4 skipped |
| Offline retrieval fixture | `uv ... run python scripts/run_retrieval_screen.py --config tests/fixtures/mini-screen.yaml` | PASS, 2 questions, BM25, zero provider calls |
| Offline experiment fixture | `uv ... run python scripts/run_experiment.py --config tests/fixtures/mini-experiment.yaml --offline` | PASS, 2 responses, zero provider calls |
| Fixture rerun | repeat the preceding experiment command | PASS, same run ID, `cache_reused=true`, `new_responses=0` |
| Migration compilation | `alembic upgrade head --sql` and `alembic downgrade head:base --sql` | PASS, 399 upgrade and 174 downgrade SQL lines |
| Compose structure | parse `docker-compose.yml` with PyYAML and assert API/DB, both health checks, dependency, and named volumes | PASS |

The four skipped tests are integration/live tests whose external prerequisites are deliberately not
enabled by the normal marker expression. The successful suite made no paid or gold access.

## Environment-blocked or intentionally withheld

| Gate | Observed evidence | Status |
|---|---|---|
| `docker compose up -d db` | `docker` executable is not installed in this environment | BLOCKED, not run |
| Online `alembic upgrade head` | no PostgreSQL is available; connection ends in local `PermissionError` | BLOCKED; offline SQL compilation passed |
| `docker compose up --build -d` and API `curl` | Docker is unavailable | BLOCKED, not claimed |
| `ragbench preflight --offline` | truthfully reported Docker/DB/migration/budget unavailable, price snapshot older than 24 hours, and secret missing; cache writable | EXPECTED NONZERO until operator prerequisites are supplied |
| Live Upstage smoke/paid corpus work | no credential or operator approval | NOT RUN |
| Sealed gold execution | no gold capability or operator approval | NOT RUN |
| Corpus-scale analysis and final research claims | corpus/parse/index/question/gold evidence does not exist | `PENDING_EVIDENCE` |

The empty key copied from `.env.example` is normalized to an unset credential and the `.env` file is
ignored by Git. No credential, provider response, restricted question, or private document was read
or created during verification.

## Release interpretation

The application framework, API contract, documentation, and no-cost fixture are locally verified.
Database/container runtime and every empirical/paid/gold gate remain pending. The optional dashboard
was therefore not built. This record does not authorize a benchmark claim or provider spend.
