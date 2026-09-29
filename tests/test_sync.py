"""Acceptance behaviour of the sync engine against an in-process Qdrant and a real Git repo."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest
from qdrant_client import QdrantClient, models

from qdrant_md_sync.chunking import MarkdownChunker
from qdrant_md_sync.gitrepo import FileSelector, GitRepo
from qdrant_md_sync.sync import KIND_CHUNK, Source, SyncError, Syncer

DIM = 8
LONG_SECTION = "\n\n".join(f"Paragraph {i} " + "lorem ipsum dolor sit amet " * 6 for i in range(6))


class FakeEmbedder:
    fingerprint = '{"model": "fake"}'

    def __init__(self) -> None:
        self.texts: list[str] = []
        self.fail_on: str | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.fail_on and any(self.fail_on in t for t in texts):
            raise RuntimeError("embedding backend unavailable")
        self.texts.extend(texts)
        return [[b / 255 + 0.01 for b in hashlib.sha256(t.encode()).digest()[:DIM]] for t in texts]


class Repo:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True)
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.email", "t@example.com")
        self.git("config", "user.name", "t")

    def git(self, *args: str) -> str:
        return subprocess.run(["git", "-C", str(self.root), *args], check=True, capture_output=True, text=True).stdout

    def write(self, path: str, text: str) -> None:
        f = self.root / path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)

    def commit(self, msg: str = "c") -> str:
        self.git("add", "-A")
        self.git("commit", "-q", "--allow-empty", "-m", msg)
        return self.git("rev-parse", "HEAD").strip()


@pytest.fixture(params=[None, "text"], ids=["unnamed-vector", "named-vector"])
def env(request, tmp_path):
    vector_name = request.param
    client = QdrantClient(":memory:")
    params = models.VectorParams(size=DIM, distance=models.Distance.COSINE)
    client.create_collection("docs", vectors_config={vector_name: params} if vector_name else params)
    repo = Repo(tmp_path / "repo")
    repo.write("README.md", "# Project\n\nIntro text.\n")
    repo.write("docs/guide.md", f"# Guide\n\n## Setup\n\n{LONG_SECTION}\n\n## Usage\n\nRun it.\n")
    repo.write("node_modules/pkg/README.md", "# vendored\n")
    repo.write("notes.txt", "not markdown\n")
    repo.commit("initial")
    return client, repo, vector_name


def make_syncer(client, repo, vector_name, embedder, **kw) -> Syncer:
    return Syncer(
        client=client,
        collection="docs",
        repo=GitRepo(repo.root),
        source=Source("acme/docs", "main"),
        chunker=MarkdownChunker(max_tokens=64),
        embedder=embedder,
        selector=FileSelector(),
        vector_name=vector_name,
        log=lambda _msg: None,
        **kw,
    )


def without_commit(snapshot: dict[str, dict]) -> dict[str, dict]:
    return {k: {f: v for f, v in p.items() if f not in ("commit", "source_url")} for k, p in snapshot.items()}


def chunks(client) -> dict[str, dict]:
    flt = models.Filter(must=[models.FieldCondition(key="kind", match=models.MatchValue(value=KIND_CHUNK))])
    records, _ = client.scroll("docs", scroll_filter=flt, limit=10_000, with_payload=True)
    return {str(r.id): r.payload for r in records}


def paths(client) -> dict[str, int]:
    out: dict[str, int] = {}
    for p in chunks(client).values():
        out[p["path"]] = out.get(p["path"], 0) + 1
    return out


def test_first_run_indexes_selected_markdown_with_traceable_metadata(env):
    client, repo, vn = env
    head = repo.git("rev-parse", "HEAD").strip()
    stats = make_syncer(client, repo, vn, FakeEmbedder()).run()

    assert stats.mode == "reconcile" and stats.state_written
    assert set(paths(client)) == {"README.md", "docs/guide.md"}
    assert paths(client)["docs/guide.md"] > 2  # long section was split
    usage = next(p for p in chunks(client).values() if p["heading"] == "Usage")
    assert usage["repository"] == "acme/docs" and usage["scope"] == "main" and usage["commit"] == head
    assert usage["headings"] == ["Guide", "Usage"] and usage["title"] == "Guide"
    assert usage["source_url"] == f"https://github.com/acme/docs/blob/{head}/docs/guide.md#usage"
    assert "Run it." in usage["text"]


def test_editing_one_file_only_reprocesses_that_file_and_changed_chunks(env):
    client, repo, vn = env
    make_syncer(client, repo, vn, FakeEmbedder()).run()
    before = chunks(client)

    repo.write("docs/guide.md", f"# Guide\n\n## Setup\n\n{LONG_SECTION}\n\n## Usage\n\nRun it twice.\n")
    head = repo.commit("edit usage")
    emb = FakeEmbedder()
    stats = make_syncer(client, repo, vn, emb).run()

    after = chunks(client)
    assert stats.mode == "incremental" and stats.files_modified == 1 and stats.files_added == 0
    assert len(emb.texts) == 1 and "Run it twice." in emb.texts[0]  # untouched chunks are not re-embedded
    readme = {k: v for k, v in after.items() if v["path"] == "README.md"}
    assert readme == {k: v for k, v in before.items() if v["path"] == "README.md"}
    guide = [v for v in after.values() if v["path"] == "docs/guide.md"]
    assert {v["commit"] for v in guide} == {head}
    assert sorted(v["chunk_index"] for v in guide) == list(range(len(guide)))
    assert not any("Run it.\n" in v["text"] or v["text"].endswith("Run it.") for v in guide)


def test_delete_and_rename_leave_no_points_under_old_paths(env):
    client, repo, vn = env
    make_syncer(client, repo, vn, FakeEmbedder()).run()

    repo.git("rm", "-q", "README.md")
    repo.git("mv", "docs/guide.md", "docs/manual.md")
    repo.commit("delete + rename")
    emb = FakeEmbedder()
    stats = make_syncer(client, repo, vn, emb).run()

    assert set(paths(client)) == {"docs/manual.md"}
    assert stats.files_deleted == 1 and stats.files_renamed == 1
    assert emb.texts == []  # renamed content reuses stored vectors


def test_rerun_same_revision_and_lost_state_produce_no_duplicates(env):
    client, repo, vn = env
    make_syncer(client, repo, vn, FakeEmbedder()).run()
    first = chunks(client)

    assert make_syncer(client, repo, vn, FakeEmbedder()).run().mode == "noop"
    client.delete("docs", points_selector=models.PointIdsList(points=[make_syncer(client, repo, vn, FakeEmbedder()).state_id]))
    emb = FakeEmbedder()
    stats = make_syncer(client, repo, vn, emb).run()

    assert stats.mode == "reconcile" and stats.files_unchanged == 2 and emb.texts == []
    assert chunks(client) == first


def test_failed_run_is_not_recorded_and_retry_converges(env):
    client, repo, vn = env
    make_syncer(client, repo, vn, FakeEmbedder()).run()
    state_before = make_syncer(client, repo, vn, FakeEmbedder())._read_state()

    repo.write("a.md", "# A\n\nalpha\n")
    repo.write("b.md", "# B\n\nboom\n")
    repo.commit("two files")
    failing = FakeEmbedder()
    failing.fail_on = "boom"
    with pytest.raises(SyncError) as exc:
        make_syncer(client, repo, vn, failing).run()
    assert exc.value.stage == "embed" and exc.value.path == "b.md"
    assert make_syncer(client, repo, vn, FakeEmbedder())._read_state() == state_before

    make_syncer(client, repo, vn, FakeEmbedder()).run()
    # `commit` records the revision at which a file's current content was indexed, so it depends on
    # history; everything else must equal a from-scratch index.
    retried = without_commit(chunks(client))
    fresh = QdrantClient(":memory:")
    params = models.VectorParams(size=DIM, distance=models.Distance.COSINE)
    fresh.create_collection("docs", vectors_config={vn: params} if vn else params)
    make_syncer(fresh, repo, vn, FakeEmbedder()).run()
    assert retried == without_commit(chunks(fresh))


def test_full_reindex_matches_fresh_index_and_prunes_foreign_leftovers(env):
    client, repo, vn = env
    make_syncer(client, repo, vn, FakeEmbedder()).run()
    expected = chunks(client)
    # Simulate corruption: a leftover point for a file that no longer exists.
    vec = [0.5] * DIM
    client.upsert(
        "docs",
        points=[
            models.PointStruct(
                id="00000000-0000-0000-0000-000000000001",
                vector={vn: vec} if vn else vec,
                payload={"kind": KIND_CHUNK, "repository": "acme/docs", "scope": "main", "path": "gone.md"},
            )
        ],
    )

    emb = FakeEmbedder()
    stats = make_syncer(client, repo, vn, emb, full_reindex=True).run()

    assert stats.mode == "full" and len(emb.texts) == len(expected)
    assert chunks(client) == expected


def test_older_run_does_not_overwrite_newer_state(env):
    client, repo, vn = env
    old = repo.git("rev-parse", "HEAD").strip()
    repo.write("new.md", "# New\n\ntext\n")
    new = repo.commit("newer")
    make_syncer(client, repo, vn, FakeEmbedder()).run()

    repo.git("checkout", "-q", old)
    stats = make_syncer(client, repo, vn, FakeEmbedder()).run()

    assert stats.mode == "skipped" and not stats.state_written
    assert make_syncer(client, repo, vn, FakeEmbedder())._read_state()["commit"] == new
    assert "new.md" in paths(client)
