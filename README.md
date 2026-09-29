# qdrant-md-sync

A GitHub Action that keeps a repository's Markdown files synchronized with an existing Qdrant
collection on every push. **Git state in, synchronized Qdrant points out** — no retrieval, ranking,
serving or answer generation.

## Usage

```yaml
# .github/workflows/qdrant-sync.yml
name: qdrant-sync
on:
  push:
    branches: [main]
  workflow_dispatch:
    inputs:
      full-reindex:
        type: boolean
        default: false

# One run per branch at a time; queued runs are not cancelled mid-write.
concurrency:
  group: qdrant-sync-${{ github.ref }}
  cancel-in-progress: false

permissions:
  contents: read

jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: sm-th/indexer@v1
        with:
          qdrant-url: ${{ secrets.QDRANT_URL }}
          qdrant-api-key: ${{ secrets.QDRANT_API_KEY }}
          collection: docs
          embedding-api-key: ${{ secrets.OPENAI_API_KEY }}
          full-reindex: ${{ inputs.full-reindex || false }}
```

The Action is a Docker action using the prebuilt image `ghcr.io/sm-th/indexer:v1`, so it needs a
Linux runner.

Embeddings are computed by any OpenAI-compatible `/embeddings` API, such as OpenAI, Azure OpenAI,
Ollama, vLLM, LiteLLM or TEI. Set `embedding-base-url` and `embedding-model` to use one other than OpenAI.

The collection must already exist, and its vector size must match the model. The default model,
`text-embedding-3-small`, produces 1536-dim vectors:

```bash
curl -X PUT "$QDRANT_URL/collections/docs" -H "api-key: $QDRANT_API_KEY" \
  -H 'content-type: application/json' \
  -d '{"vectors": {"size": 1536, "distance": "Cosine"}}'
```

### Inputs

| Input | Default | |
|---|---|---|
| `qdrant-url` | — (required) | |
| `qdrant-api-key` | | Needs write access to the collection only. |
| `collection` | — (required) | Existing collection. |
| `repository` | `${{ github.repository }}` | Namespace stored on every point. |
| `scope` | `${{ github.ref_name }}` | Branch / source scope stored on every point. |
| `vector-name` | unnamed vector | Named dense vector to write. |
| `embedding-base-url` | `https://api.openai.com/v1` | Any OpenAI-compatible API. |
| `embedding-model` | `text-embedding-3-small` | Output size must match the collection. |
| `embedding-api-key` | | Bearer token; optional for local servers. |
| `embedding-dimensions` | | Optional `dimensions` request parameter. |
| `include` | `**/*.md` | Gitignore-style globs, newline or comma separated (e.g. `**/*.md, **/*.mdx`). |
| `exclude` | | Added to the built-in excludes (`node_modules/`, `vendor/`, `third_party/`, `dist/`, `build/`, `out/`, `target/`, `site/`, `_site/`, `.venv/`, …). `!build/` re-includes. |
| `max-tokens` | `512` | Chunk budget, measured with `tokenizer`. Keep it within the model's input limit. |
| `tokenizer` | `cl100k_base` | tiktoken encoding. |
| `seed` | | Mixed into every hash; change it to force re-chunking and re-embedding. |
| `full-reindex` | `false` | Rebuild every point of this repository/scope, re-embedding everything. |
| `path` | `.` | Repository path inside the workspace. |

Outputs: `mode`, `commit`, `files-added`, `files-modified`, `files-renamed`, `files-deleted`,
`chunks-embedded`, `chunks-reused`, `chunks-deleted`, `state-written`. The run also writes a job summary.

## How it works

Only files committed at `HEAD` are indexed; regular files only, so symlinks and submodules are skipped. File
contents are read from Git objects, not from the working tree.

**Run state.** Each repository/scope has one point with `kind = "sync_state"` in the collection. It
holds the last successfully indexed commit and fingerprints of the configuration. It is written
only after every data write of the run succeeded, so a failed run never claims its revision.

| Situation | Mode |
|---|---|
| state commit is in local history | `incremental`: `git diff state..HEAD` (adds, modifications, renames, deletions) |
| no state, commit missing from history (shallow clone, force push), chunking/model/selection changed | `reconcile`: every selected file is compared with the index; points of files that no longer exist are deleted |
| `full-reindex: true` | `full`: like reconcile but everything is re-chunked and re-embedded |
| `HEAD` already indexed | `noop` |
| a newer descendant commit is already indexed | `skipped`: an older run never overwrites newer state |

**Hashes.** `chunking_version = hash(schema, chunker config, embedding model, vector name, seed)`.

- `file_hash = hash(chunking_version, file bytes)`: a file whose points all carry the current
  `file_hash` is skipped without chunking.
- `chunk_hash = hash(chunking_version, embedded text)`: before embedding, the collection is
  searched for a point with the same `chunk_hash` and its vector is reused. Editing one paragraph of
  a large file embeds only that chunk; renames and moved sections re-embed nothing.
- Point ID = `uuid5(repository, scope, path, chunk_hash, occurrence)`. A chunk whose text did not
  change keeps its point, and only its payload (position, commit) is rewritten. Changing the model or
  chunking changes every ID, and the old points are deleted.

Per file, writes happen in this order: new points are upserted, payloads of kept points are updated,
then stale points are deleted. If a run is interrupted, the file's points are left with mixed
hashes and counts, so the retry redoes that file. Re-running a revision never creates duplicates.

**Chunking** (deterministic, no LLM): ATX headings outside fenced code form a tree. A section that fits
`max-tokens` becomes one chunk. An oversized section is split into its intro and its subsections,
and small neighbours are packed back together. Oversized text is split on paragraph boundaries with
fenced code kept whole where it fits; a single oversized block is split by Chonkie's
`RecursiveChunker`. The embedded text is `Title > Heading > Subheading` followed by a blank line and the chunk text.
YAML front matter is excluded from chunks; its `title:` (else the first H1) becomes the document title.

## Payload

Chunk points (`kind = "chunk"`):

| Field | |
|---|---|
| `repository`, `scope`, `path` | Source namespace. Always filter searches by `kind` and usually by `repository`. |
| `commit` | Revision at which this file's current content was indexed (its content equals `HEAD`'s). |
| `title`, `headings`, `heading`, `anchor` | Document title, heading breadcrumb, deepest heading, GitHub heading anchor. |
| `chunk_index`, `chunk_count`, `start_line`, `end_line` | Position within the file (1-based lines). |
| `text` | Raw chunk text (verbatim span of the file). |
| `source_url` | `https://github.com/<repo>/blob/<commit>/<path>#<anchor>`. |
| `file_hash`, `chunk_hash`, `chunking_version`, `indexer_version` | Sync bookkeeping. |

Searches must exclude the state point, e.g. `{"must": [{"key": "kind", "match": {"value": "chunk"}}]}`.
The Action creates keyword payload indexes on `kind`, `repository`, `scope`, `path` and `chunk_hash`.

## Security

- Trigger on `push` (and `workflow_dispatch`) only. Never run the Action from `pull_request_target`
  or other contexts that expose secrets to fork code.
- Use a Qdrant API key restricted to the target collection.
- Markdown is treated as data: nothing is executed or rendered. Special-token strings are tokenized
  as plain text.
- Errors name the pipeline stage and file. They include only the exception type and first line,
  and credentials are masked in the log.

## CLI

The Action runs a thin CLI that also works locally:

```bash
uv run qdrant-md-sync --path . --qdrant-url http://localhost:6333 --collection docs \
  --repository acme/docs --scope main --embedding-api-key "$OPENAI_API_KEY"
```

Every flag can also be set by environment variable: `QDRANT_URL`, `QDRANT_API_KEY`, and `QMS_<FLAG>`
for the rest, e.g. `QMS_EMBEDDING_BASE_URL`.

## Verifying

### 1. Acceptance tests (no services needed)

```bash
uv sync
uv run pytest -q
```

These tests run against an in-process Qdrant and real Git repositories with a deterministic fake
embedder. They cover the first index, editing, renaming, deleting, re-running a revision, failure
followed by retry, full reindex, and an older run racing a newer one. Each case runs with both an
unnamed and a named vector.

### 2. End to end against a real Qdrant and a real embeddings API

`scripts/e2e.py` creates a throwaway collection and Git repository and drives the CLI through
every acceptance scenario, printing `PASS`/`FAIL` for each check. It deletes the collection at the
end; pass `--keep` to inspect it afterwards.

```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

# Embeddings, option A: OpenAI
export QMS_EMBEDDING_API_KEY=sk-...

# Embeddings, option B: local Ollama, no key needed
docker run -d --name ollama -p 11434:11434 ollama/ollama
docker exec ollama ollama pull nomic-embed-text
export QMS_EMBEDDING_BASE_URL=http://localhost:11434/v1 QMS_EMBEDDING_MODEL=nomic-embed-text

uv run scripts/e2e.py
```

Point it at another Qdrant with `QDRANT_URL` / `QDRANT_API_KEY`. Expected ending:

```text
7. full reindex reproduces the collection state
PASS  mode=full, re-embedded 5 chunks
PASS  same point IDs
PASS  same payloads (commit refreshed to HEAD)
8. search results are traceable
PASS  top hit carries repository, path, headings, commit
      top hit: docs/manual.md > Operator Guide > Troubleshooting @ d09669d81a7f

ALL CHECKS PASSED
```

### 3. The Docker image

```bash
docker build -t qdrant-md-sync:dev .
docker run --rm qdrant-md-sync:dev --help
```

### 4. In GitHub Actions

1. Create a collection in a Qdrant instance that GitHub runners can reach (e.g. Qdrant Cloud) with
   the model's vector size (see [Usage](#usage)).
2. In a test repository, add the secrets `QDRANT_URL`, `QDRANT_API_KEY` and `OPENAI_API_KEY`, and the
   workflow from [Usage](#usage). Pass `embedding-base-url`/`embedding-model` if you use an embeddings API other than OpenAI.
3. Walk through the scenarios and compare each run's job summary:

| Action | Expected job summary |
|---|---|
| Push the workflow (first run) | Mode `reconcile — no previous sync state`; every Markdown file counted as added; `Revision recorded: yes` |
| Edit one paragraph of one file and push | Mode `incremental`; files modified `1`; chunks embedded = only the changed chunks |
| `git mv` a file and push | Files renamed `1`; chunks embedded `0`; vectors reused |
| Delete a file and push | Files deleted `1`; stale chunks deleted > 0 |
| Re-run the last job (*Re-run jobs*) | Mode `noop — commit already indexed` |
| *Run workflow* with `full-reindex` checked | Mode `full`; every chunk re-embedded; point count unchanged |
| Make the embeddings key invalid, push, then fix the key and re-run | The failed run shows an error annotation naming the stage and file, plus `Revision recorded: no`; the re-run succeeds |

4. Check that the points are traceable, and that no point is left under an old path:

```bash
curl -s "$QDRANT_URL/collections/docs/points/scroll" -H "api-key: $QDRANT_API_KEY" \
  -H 'content-type: application/json' \
  -d '{"limit": 1000, "with_payload": ["path", "headings", "commit"],
       "filter": {"must": [{"key": "kind", "match": {"value": "chunk"}},
                           {"key": "repository", "match": {"value": "OWNER/REPO"}}]}}' \
  | jq -r '.result.points[].payload | "\(.path)  \(.headings | join(" > "))  \(.commit[:12])"' | sort
```

## Releasing

Push a `vX.Y.Z` tag. `.github/workflows/release.yml` publishes `ghcr.io/sm-th/indexer:vX.Y.Z`,
`:vX.Y` and `:vX`. After a release, move the `v1` Git tag so that `uses: sm-th/indexer@v1`
resolves to it. The GHCR package must be public so that other repositories can pull the image.
