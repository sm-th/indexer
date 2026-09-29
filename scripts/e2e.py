"""End-to-end check against a real Qdrant and a real OpenAI-compatible embeddings API.

Creates a throwaway collection and Git repository, drives the CLI through the acceptance
scenarios (first index, edit, rename, delete, re-run, failed run + retry, full reindex, search),
prints PASS/FAIL per check and deletes the collection again.

    uv run scripts/e2e.py [--keep]

Configuration comes from the same environment variables the CLI reads:
QDRANT_URL (default http://localhost:6333), QDRANT_API_KEY, QMS_EMBEDDING_BASE_URL,
QMS_EMBEDDING_MODEL, QMS_EMBEDDING_API_KEY, QMS_EMBEDDING_DIMENSIONS.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from qdrant_client import QdrantClient, models

from qdrant_md_sync.embeddings import OpenAIEmbeddings

REPOSITORY, SCOPE = "e2e/demo", "main"
PARAGRAPHS = "\n\n".join(
    f"Paragraph {i}. The service reads its settings from environment variables, validates them on "
    f"startup and refuses to start when a required value is missing. Option {i} controls retries."
    for i in range(20)
)
GUIDE = f"""---
title: Operator Guide
---
# Operator Guide

## Install

Download the release archive and unpack it into /opt/demo. Then run the installer.

## Configure

{PARAGRAPHS}

## Troubleshooting

If the service does not start, check the logs first.
"""

failures = 0


def check(ok: bool, message: str) -> None:
    global failures
    failures += 0 if ok else 1
    print(f"{'PASS' if ok else 'FAIL'}  {message}", flush=True)


class Repo:
    def __init__(self, root: Path):
        self.root = root
        self.git("init", "-q", "-b", "main")

    def git(self, *args: str) -> str:
        cmd = ["git", "-C", str(self.root), "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false",
               "-c", "user.name=e2e", "-c", "user.email=e2e@example.invalid", *args]
        return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()

    def write(self, path: str, text: str) -> None:
        f = self.root / path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)

    def commit(self, message: str) -> str:
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)
        return self.git("rev-parse", "HEAD")


def main() -> int:
    keep = "--keep" in sys.argv[1:]
    os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ.get("QDRANT_API_KEY") or None)
    embedder = OpenAIEmbeddings(
        model=os.environ.get("QMS_EMBEDDING_MODEL") or "text-embedding-3-small",
        api_key=os.environ.get("QMS_EMBEDDING_API_KEY", ""),
        base_url=os.environ.get("QMS_EMBEDDING_BASE_URL") or "https://api.openai.com/v1",
        dimensions=int(os.environ["QMS_EMBEDDING_DIMENSIONS"]) if os.environ.get("QMS_EMBEDDING_DIMENSIONS") else None,
    )
    dim = len(embedder.embed(["dimension probe"])[0])
    collection = f"qms-e2e-{uuid.uuid4().hex[:8]}"
    client.create_collection(collection, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))
    print(f"collection {collection} ({dim} dims) at {os.environ['QDRANT_URL']}\n")

    def sync(*extra: str, env: dict[str, str] | None = None) -> tuple[int, dict]:
        cmd = [sys.executable, "-m", "qdrant_md_sync.cli", "--path", str(repo.root), "--collection", collection,
               "--repository", REPOSITORY, "--scope", SCOPE, *extra]
        proc = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **(env or {})})
        if proc.returncode != 0:
            print(f"      cli exit {proc.returncode}: {proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ''}")
            return proc.returncode, {}
        return 0, json.loads(proc.stdout)

    def chunks() -> dict[str, dict]:
        flt = models.Filter(must=[models.FieldCondition(key="kind", match=models.MatchValue(value="chunk"))])
        records, _ = client.scroll(collection, scroll_filter=flt, limit=10_000, with_payload=True)
        return {str(r.id): r.payload for r in records}

    def paths() -> set[str]:
        return {p["path"] for p in chunks().values()}

    def state_commit() -> str | None:
        flt = models.Filter(must=[models.FieldCondition(key="kind", match=models.MatchValue(value="sync_state"))])
        records, _ = client.scroll(collection, scroll_filter=flt, limit=10, with_payload=True)
        return records[0].payload["commit"] if records else None

    try:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Repo(Path(tmp))
            repo.write("README.md", "# Demo\n\nA tiny demo project.\n")
            repo.write("docs/guide.md", GUIDE)
            repo.write("node_modules/pkg/README.md", "# vendored dependency\n")
            repo.write("notes.txt", "not markdown\n")
            head = repo.commit("initial")

            print("1. first run indexes all selected Markdown files")
            rc, s = sync()
            check(rc == 0 and s["mode"] == "reconcile" and s["state_written"], f"mode={s.get('mode')}, revision recorded")
            check(paths() == {"README.md", "docs/guide.md"}, f"indexed paths {sorted(paths())} (node_modules, *.txt skipped)")
            guide_chunks = [p for p in chunks().values() if p["path"] == "docs/guide.md"]
            check(len(guide_chunks) > 1, f"guide split into {len(guide_chunks)} chunks")
            check(state_commit() == head, "state point holds HEAD")

            print("2. editing one paragraph reprocesses only that file and chunk")
            readme_before = {k: v for k, v in chunks().items() if v["path"] == "README.md"}
            repo.write("docs/guide.md", GUIDE.replace("check the logs first", "check the logs and the config first"))
            head = repo.commit("edit troubleshooting")
            rc, s = sync()
            check(rc == 0 and s["mode"] == "incremental" and s["files_modified"] == 1, f"mode={s.get('mode')}, files_modified={s.get('files_modified')}")
            check(s.get("chunks_embedded") == 1, f"chunks_embedded={s.get('chunks_embedded')} (unchanged chunks keep their vectors)")
            check({k: v for k, v in chunks().items() if v["path"] == "README.md"} == readme_before, "README.md points untouched")

            print("3. renaming a file leaves no points under the old path")
            repo.git("mv", "docs/guide.md", "docs/manual.md")
            head = repo.commit("rename")
            rc, s = sync()
            check(rc == 0 and s["files_renamed"] == 1 and s["chunks_embedded"] == 0, f"files_renamed={s.get('files_renamed')}, chunks_embedded={s.get('chunks_embedded')} (vectors reused)")
            check(paths() == {"README.md", "docs/manual.md"}, f"paths now {sorted(paths())}")

            print("4. deleting a file removes its points")
            repo.git("rm", "-q", "README.md")
            head = repo.commit("delete")
            rc, s = sync()
            check(rc == 0 and s["files_deleted"] == 1 and s["chunks_deleted"] >= 1, f"files_deleted={s.get('files_deleted')}, chunks_deleted={s.get('chunks_deleted')}")
            check(paths() == {"docs/manual.md"}, f"paths now {sorted(paths())}")

            print("5. re-running the same revision creates no duplicates")
            before = chunks()
            rc, s = sync()
            check(rc == 0 and s["mode"] == "noop", f"mode={s.get('mode')}")
            check(chunks() == before, f"{len(before)} points, unchanged")

            print("6. a failed run is not recorded and can be retried")
            recorded = state_commit()
            repo.write("faq.md", "# FAQ\n\nQuestions and answers.\n")
            head = repo.commit("add faq")
            broken = {"QMS_EMBEDDING_BASE_URL": os.environ["QDRANT_URL"].rstrip("/") + "/no-such-embeddings-api"}
            rc, _ = sync(env=broken)
            check(rc == 1, "run with a broken embeddings endpoint fails")
            check(state_commit() == recorded, "state still points at the previous revision")
            rc, s = sync()
            check(rc == 0 and s["files_added"] == 1 and state_commit() == head, "retry succeeds and records HEAD")

            print("7. full reindex reproduces the collection state")
            before = chunks()
            rc, s = sync("--full-reindex")
            after = chunks()
            check(rc == 0 and s["mode"] == "full" and s["chunks_embedded"] == len(before), f"mode={s.get('mode')}, re-embedded {s.get('chunks_embedded')} chunks")
            check(set(after) == set(before), "same point IDs")
            strip = lambda d: {k: {f: v for f, v in p.items() if f not in ("commit", "source_url")} for k, p in d.items()}
            check(strip(after) == strip(before), "same payloads (commit refreshed to HEAD)")

            print("8. search results are traceable")
            query = embedder.embed(["How do I install the service?"])[0]
            hits = client.query_points(
                collection,
                query=query,
                limit=1,
                query_filter=models.Filter(must=[
                    models.FieldCondition(key="kind", match=models.MatchValue(value="chunk")),
                    models.FieldCondition(key="repository", match=models.MatchValue(value=REPOSITORY)),
                ]),
            ).points
            top = hits[0].payload if hits else {}
            check(all(top.get(k) for k in ("repository", "path", "headings", "commit")), "top hit carries repository, path, headings, commit")
            print(f"      top hit: {top.get('path')} > {' > '.join(top.get('headings', []))} @ {str(top.get('commit'))[:12]}")
    finally:
        if keep:
            print(f"\nkept collection {collection}")
        else:
            client.delete_collection(collection)

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else f'{failures} CHECK(S) FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
