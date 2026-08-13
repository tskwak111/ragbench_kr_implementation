# Task 10 report — grounded generation and citations

## Implementation

- Added a deterministic `ContextBuilder` over resolved retrieval evidence. It orders by retrieval
  rank, descending score, and stable source identity; de-duplicates only identical chunk
  provenance; rejects conflicting duplicate IDs; and retains a top-ranked prefix without ever
  splitting a chunk/provenance envelope. Exact token usage is counted with the pinned vendored
  `cl100k_base` tokenizer over the final serialized context, including delimiters and metadata.
- Added explicit untrusted-document JSON envelopes containing citation ID, chunk ID, document ID
  and title, page range, section path, and content. Markup characters inside document data are
  escaped so source text cannot forge the control delimiters.
- Added three packaged, immutable prompt versions: V1 basic, V2 context-only with required
  citations, and V3 context-only with explicit abstention. All require one strict JSON object with
  exactly `answer`, `citations`, and `abstained`; document instructions remain untrusted data.
- Added strict Pydantic response parsing with extra fields and coercions forbidden. The only repair
  is one removal of a complete Markdown JSON fence. Persistent syntax/schema failures surface the
  stable `GENERATION_SCHEMA_ERROR`; semantic fields and citations are never repaired or invented.
- Added citation policy and resolution against only the context records actually included after
  token truncation. Duplicate, raw chunk-ID, unknown, unsupported, and abstention-conflicting
  citations fail. Document/page/section provenance is attached exclusively from server-side
  context evidence.
- Added `RagService.answer`, returning question, answer/abstention, included evidence, enriched
  citations, latency, provider usage, model/prompt/config/experiment identities, cache status,
  response ID, and correlation ID. It validates that the evidence source exactly matches the
  retriever results before generation. It calls only the existing `ProviderGateway.generate`,
  preserving the gateway's cache, singleflight, budget, pricing, and metering path.
- Extended `GenerateResponse` with cache-hit evidence and populated it on both cold and cached
  Upstage generation paths. Added a fail-closed, injectable `ragbench query` CLI service path that
  emits the structured response and preserves schema/citation error codes without leaking model
  output. Production retrieval/evidence wiring remains an application assembly concern; the CLI
  does not invent a direct-provider fallback.

## TDD evidence

The initial focused RED run failed during collection because the four RAG modules did not exist:

```text
ModuleNotFoundError: No module named 'ragbench.rag.context'
ModuleNotFoundError: No module named 'ragbench.rag.citations'
4 errors during collection
```

The gateway cache-evidence contract was separately observed RED with:

```text
AttributeError: 'GenerateResponse' object has no attribute 'cache_hit'
```

The CLI query contracts were observed RED because `CommandServices` had no `query_runner`. Self-
review also produced focused RED failures for mutable provider parameters, conflicting duplicate
provenance, missing evidence-source rows, and missing stable CLI schema error code. Each was fixed
with a minimal behavioral change before the final focused run:

```text
uv --cache-dir /private/tmp/ragbench-task10-uv-cache run pytest tests/unit/rag -q
28 passed in 0.10s
```

## Verification and limitations

Fresh verification before the implementation commit:

```text
uv --cache-dir /private/tmp/ragbench-task10-uv-cache run pytest \
  -m 'not live and not gold' -q
263 passed, 4 skipped in 1.82s

uv --cache-dir /private/tmp/ragbench-task10-uv-cache run ruff check .
All checks passed!

uv --cache-dir /private/tmp/ragbench-task10-uv-cache run mypy src/ragbench
Success: no issues found in 41 source files
```

`git diff --check` and a targeted credential-pattern scan were clean. No HTTP provider call, paid
action, PostgreSQL call, sealed-gold access, private corpus access, or real-corpus quality claim was
performed. Task 10's tiny live check remains intentionally pending explicit budget/operator
approval and complete application dataset wiring.
