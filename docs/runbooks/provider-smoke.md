# Provider smoke runbook

Provider smoke commands are dry-runs by default. They show the exact model and worst-case
projected cost without creating an Upstage HTTP client or making any network request.

```bash
uv run ragbench preflight --offline --json
uv run ragbench prices verify --json
uv run ragbench usage status --json
uv run ragbench smoke solar --json
uv run ragbench smoke parse --json
uv run ragbench smoke embed --json
```

The preflight result contains Docker, PostgreSQL readiness, migration-head, writable-cache,
price-snapshot, budget, and credential-presence checks. It never prints an API key. A missing
Docker executable or unavailable database is a failed check, not a false positive.

## Live operator procedure

Do not enable live tests or `--execute` without explicit operator approval. First update the
local price snapshot and confirm the currently supported provider model IDs in the Upstage
console. The price verifier rejects a snapshot older than 24 hours or an ambiguous promotion.

After approval, set the credential only in the operator environment and explicitly enable live
tests. Each smoke path has a bounded request: a 200-token Korean answer cap, one locally
generated PDF page, or a short Korean embedding input. The CLI requires both `--execute` and
`--approve`; it additionally requires `RUN_LIVE_UPSTAGE_TESTS=1` and `UPSTAGE_API_KEY`.

```bash
export RUN_LIVE_UPSTAGE_TESTS=1
export UPSTAGE_API_KEY='set-in-your-shell-only'
uv run ragbench smoke solar --execute --approve --json
uv run ragbench smoke parse --execute --approve --json
uv run ragbench smoke embed --execute --approve --json
uv run pytest -m live tests/live/test_upstage_smoke.py -q
```

Run each command once. Record the JSON `provider_response_id`, `latency_ms`, model ID,
projected maximum, and the UTC timestamp in the deployment log. The gateway writes raw provider
responses into the response cache. Repeating the exact request is a cache hit and must not send a
second paid request; the opt-in live test verifies that behavior with a fresh in-memory cache.

The production CLI owns and disposes the regular database engine, the dedicated `NullPool`
advisory-lock engine, and the provider HTTP client when an executed smoke command completes.
