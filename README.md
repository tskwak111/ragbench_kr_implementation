# RAGBench-KR

RAGBench-KR is a reproducible experiment framework for measuring Korean long-document
retrieval-augmented generation. It is an experiment platform, not a general-purpose chatbot.

## Key Results

**No benchmark result is published yet.** The software framework, no-cost public fixture, frozen
corpus, and reconstructed Standard parse snapshot are implemented. A 30-page Standard-only QA
found prose and most tables usable for RAG, but chart extraction unsafe for numeric evidence.
Enhanced parsing, paired QA, embedding, benchmark human validation, and sealed-gold execution
remain pending.
Consequently, this repository does not claim that one parser, chunker, retriever, prompt, or Solar
model is better than another.

The small offline fixture is a reproduction smoke test only. Its perfect deterministic scores are
constructed expectations and are not research evidence.

## Problem

Korean long-document RAG quality can change with parsing mode, chunk boundary policy, retriever,
Top-K, and grounding instructions. RAGBench-KR records those decisions as immutable versions and
measures retrieval quality, grounded answer quality, latency, and estimated cost without treating
an LLM judge as the sole ground truth.

The seven preregistered research questions are listed in the
[implementation plan](2026-08-13-ragbench-kr-implementation-plan.md). Claims are restricted to the
eventual tested corpus, question cohort, confidence intervals, and documented counterexamples.

## Architecture

The Python 3.12 package separates deterministic domain logic from external effects:

- ingestion records corpus provenance and dual parse checkpoints;
- normalization and chunking preserve page/section evidence;
- dense, BM25, and RRF hybrid retrieval share immutable snapshot filters;
- grounded generation accepts only server-resolved citations;
- benchmark splits keep ordinary development paths blind to sealed gold;
- experiment runners are resumable, budget-guarded, and content-addressed;
- evaluation and analysis export aggregate, versioned, public-safe artifacts;
- FastAPI is a thin injected transport layer and never exposes raw provider or gold records.

Every provider call must pass through the cached, metered gateway with a correlation ID, bounded
retry policy, price snapshot, and atomic budget reservation. Unit and contract tests cannot make a
paid network request.

## Dataset

The target is 20–30 Korean corporate/public long documents totaling 1,500–2,000 pages, balanced
across text-heavy and table-heavy material. The committed frozen manifest records 20 locally
acquired documents totaling 1,981 pages. A human reviewer approved the source, metadata, rendered
page samples, local evaluation use, and no-redistribution restriction on 2026-08-21. Raw files
remain private and ignored by Git. Every current record is conservatively nonredistributable, and
document bytes are never part of public exports.

The only committed document content is a two-chunk synthetic fixture under `tests/fixtures/`. It
exists to prove that a fresh environment can retrieve, cite, score, and cache a tiny run without a
key. Corpus metadata is committed in `configs/corpus.yaml`; its source PDF bytes are not.

## Methodology

The planned core grid compares two parse modes, six fixed token chunkers plus one heading-aware
chunker, dense/BM25/hybrid retrieval, and Top-K 3/5/10. Retrieval screening precedes generation.
Shortlisting and final selection rules are fixed before outcome inspection. A development split is
used for iteration; a human-reviewed sealed gold split is opened once for the preregistered top
three configurations.

Deterministic metrics, human calibration, paired bootstrap intervals, and failure analysis are
implemented as separate stages. Any gold-affecting bug fix invalidates affected results instead of
silently rewriting them.

## Experiments

Experiment identity binds exact corpus, parse, chunk, embedding, question, prompt, model, metric,
price, and code versions. Paid execution requires a dry-run plan, a matching confirmation hash,
fresh prices, live-mode opt-in, and budget headroom. Large API requests return a queued plan; they
are never executed while holding the HTTP request open.

Current example development configs are plans, not completed runs. The public fixture command is:

```bash
uv run python scripts/run_experiment.py \
  --config tests/fixtures/mini-experiment.yaml \
  --offline
```

## Retrieval

The dense path includes a NumPy cosine reference and versioned pgvector evidence. The sparse path
uses a documented Unicode-normalized, conservative Korean/number tokenizer. Hybrid search
over-fetches both branches and applies deterministic reciprocal-rank fusion. All three consume the
same corpus/parse/chunk/embedding filter identity.

No corpus-scale retrieval comparison has been run. The fixture uses BM25 only and reports zero
provider calls.

## Generation

Prompt versions cover a basic answer, context-only cited answer, and context-only answer with
explicit abstention. Retrieved document text is delimited as untrusted data. Generated JSON is
strictly parsed, citation IDs are validated against the exact included context, and document/page
metadata is attached server-side. Persistent schema errors remain errors rather than invented
answers.

The fixture uses deterministic expected text and is not a model evaluation.

## Error Analysis

The frozen taxonomy includes parser, retrieval, chunk-boundary, table, generation, hallucination,
citation, abstention, and benchmark-defect failures. The publication pipeline expects 50–100
stratified manual failure reviews in source-to-answer order. See
[final-analysis.md](docs/reports/final-analysis.md) for the pending-evidence report structure.

## Cost

Money is represented with `Decimal`; integer tokens/pages and configured price snapshots determine
worst-case reservations. Provider-console billing remains the source of truth and must be
reconciled before claims. The Standard run's console evidence reports 2,081 pages and USD 20.81 in
used credit; this is 100 pages above the 1,981-page successful corpus and remains an explicit
reconciliation discrepancy.

## Demo

Start the local API and pgvector database:

```bash
cp .env.example .env
docker compose up --build -d --wait
curl --fail http://localhost:8000/health
```

The public API exposes:

- `GET /health`
- `POST /documents` and `GET /documents/{id}`
- `POST /search` and `POST /query`
- `POST /experiments`, `GET /experiments`, `GET /experiments/{id}`, and
  `GET /experiments/{id}/metrics`

Domain services are injected. The default Compose image reports `domain_services` as not ready and
unbound operations fail closed with a sanitized `503` envelope; container health confirms process
liveness, not benchmark readiness.
Every response carries `X-Correlation-ID`. There are intentionally no gold/question-preview or raw
provider-response routes.

## Reproduction

The complete fresh-clone, no-key, database/API, and user-corpus instructions are in
[docs/runbooks/reproduction.md](docs/runbooks/reproduction.md). The shortest no-cost verification is:

```bash
uv sync --frozen --all-groups
uv run pytest -m "not live and not gold" -q
uv run python scripts/run_retrieval_screen.py --config tests/fixtures/mini-screen.yaml
uv run python scripts/run_experiment.py --config tests/fixtures/mini-experiment.yaml --offline
```

Do not set `UPSTAGE_API_KEY` for those commands. Repeating the fixture experiment resolves the same
content-addressed artifact and creates no duplicate response file.

## Limitations

- Dual parses, 14 chunk snapshots, embedding indexes, and the synthetic benchmark are pending
  provider execution and downstream review.
- PostgreSQL/pgvector integration and container startup require Docker; environments without it can
  run only the offline fixture and static/unit gates.
- Current provider model IDs and prices must be reverified before every paid batch.
- Korean BM25 deliberately omits morphological analysis to keep the baseline reproducible.
- Judge metrics cannot replace human calibration or sealed-gold human evidence.
- The API is an experiment transport, not a production multi-tenant service.
- The optional dashboard was omitted because the core empirical gates are not green.

## Ethics and Licensing

Use only documents whose acquisition and processing are authorized. Preserve source attribution,
license state, and redistribution restrictions. Do not publish private documents, raw provider
responses, sealed questions, expected answers, credentials, or identifiable reviewer data.

Generated questions can contain factual or representational defects. Human review against the
original PDF is required before gold inclusion. Report benchmark limitations, uncertainty, failed
counterexamples, and potential domain/template bias alongside any future result.

No project-wide software license has been declared yet; absent an explicit license grant, normal
copyright restrictions apply. Vendored tokenizer assets retain their documented upstream license,
and source documents retain their own terms.
