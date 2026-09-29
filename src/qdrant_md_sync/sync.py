"""Git state in, synchronized Qdrant points out.

Identity and change detection
-----------------------------
* `chunking_version` = hash(schema, chunker config, embedding model, vector name, user seed).
* `file_hash`  = hash(chunking_version, file bytes)   -> skip unchanged files without chunking.
* `chunk_hash` = hash(chunking_version, embedded text) -> reuse vectors, never re-embed known text.
* point id     = uuid5(repository, scope, path, chunk_hash, occurrence) -> a chunk whose text did
  not change keeps its point; only its positional payload is rewritten.

Run state (last indexed commit + config fingerprints) lives in one `kind=sync_state` point in
the same collection, written only after every data write of the run succeeded.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from importlib.metadata import version
from urllib.parse import quote

from qdrant_client import QdrantClient, models

from .chunking import Chunk, Chunker
from .embeddings import Embedder
from .gitrepo import Change, FileSelector, GitRepo

SCHEMA_VERSION = 1
INDEXER_VERSION = version("qdrant-md-sync")
POINT_NAMESPACE = uuid.UUID("5a1f3c2e-8b7d-4e61-9f0a-6c2d8e4b1a73")
KIND_CHUNK = "chunk"
KIND_STATE = "sync_state"
KEYWORD_INDEXES = ("kind", "repository", "scope", "path", "chunk_hash")
_EXISTING_FIELDS = ["path", "file_hash", "chunk_index", "chunk_count"]


class SyncError(RuntimeError):
    """A failure attributed to a pipeline stage and, when known, a source file."""

    def __init__(self, stage: str, message: str, path: str | None = None):
        self.stage = stage
        self.path = path
        where = f" [{path}]" if path else ""
        super().__init__(f"{stage}{where}: {message}")


@dataclass(frozen=True)
class Source:
    repository: str
    scope: str
    server_url: str | None = "https://github.com"

    def url(self, commit: str, path: str, anchor: str | None) -> str | None:
        if not self.server_url or self.repository.count("/") != 1:
            return None
        url = f"{self.server_url.rstrip('/')}/{self.repository}/blob/{commit}/{quote(path)}"
        return f"{url}#{anchor}" if anchor else url


@dataclass
class Stats:
    mode: str = ""
    reason: str = ""
    base_commit: str | None = None
    commit: str = ""
    files_added: int = 0
    files_modified: int = 0
    files_renamed: int = 0
    files_deleted: int = 0
    files_unchanged: int = 0
    chunks_embedded: int = 0
    chunks_reused: int = 0
    chunks_updated: int = 0
    chunks_deleted: int = 0
    duration_seconds: float = 0.0
    state_written: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Plan:
    mode: str
    reason: str
    base: str | None = None
    upserts: dict[str, str] = field(default_factory=dict)  # path -> A | M | R | ? (reconcile)
    deletes: set[str] = field(default_factory=set)
    renamed_from: set[str] = field(default_factory=set)  # subset of deletes counted as renames
    prune_except: set[str] | None = None  # reconcile: delete every path not in this set
    force: bool = False


@dataclass(frozen=True)
class _NewPoint:
    id: str
    chunk_hash: str
    embed_text: str
    payload: dict


def _hash(*parts: str | bytes) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode() if isinstance(part, str) else part)
        h.update(b"\0")
    return h.hexdigest()


def _match(key: str, value: str) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def _match_any(key: str, values: Iterable[str]) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchAny(any=sorted(values)))


def embed_text(title: str | None, chunk: Chunk) -> str:
    context = list(chunk.headings)
    if title and (not context or context[0] != title):
        context.insert(0, title)
    header = " > ".join(context)
    return f"{header}\n\n{chunk.text}" if header else chunk.text


class Syncer:
    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        repo: GitRepo,
        source: Source,
        chunker: Chunker,
        embedder: Embedder,
        selector: FileSelector,
        vector_name: str | None = None,
        seed: str = "",
        full_reindex: bool = False,
        upsert_batch: int = 64,
        lookup_batch: int = 64,
        log: Callable[[str], None] = print,
    ):
        self.client = client
        self.collection = collection
        self.repo = repo
        self.source = source
        self.chunker = chunker
        self.embedder = embedder
        self.selector = selector
        self.vector_name = vector_name or None
        self.full_reindex = full_reindex
        self.upsert_batch = upsert_batch
        self.lookup_batch = lookup_batch
        self.log = log
        self.chunking_version = _hash(
            json.dumps(
                {
                    "schema": SCHEMA_VERSION,
                    "chunker": chunker.fingerprint,
                    "embedding": embedder.fingerprint,
                    "vector": self.vector_name or "",
                    "seed": seed,
                },
                sort_keys=True,
            )
        )[:32]
        self.state_id = str(uuid.uuid5(POINT_NAMESPACE, f"{KIND_STATE}\x1f{source.repository}\x1f{source.scope}"))
        self._dimension = 0

    # ---------------------------------------------------------------- public

    def run(self) -> Stats:
        started = time.monotonic()
        stats = Stats()
        head = self.repo.head()
        stats.commit = head
        self._dimension = self._stage("collection", self._vector_dimension)
        self._stage("collection", self._ensure_indexes)
        state = self._stage("state", self._read_state)
        plan = self._stage("plan", lambda: self._plan(head, state))
        stats.mode, stats.reason, stats.base_commit = plan.mode, plan.reason, plan.base
        self.log(f"mode={plan.mode} ({plan.reason}) base={plan.base or '-'} head={head}")

        if plan.mode not in ("noop", "skipped"):
            # Upsert first: renamed files can then reuse vectors still stored under the old path.
            self._upsert_files(head, plan, stats)
            self._delete_paths(plan, stats)
            self._stage("state", lambda: self._commit_state(head, state, stats))
        stats.duration_seconds = round(time.monotonic() - started, 3)
        return stats

    # ---------------------------------------------------------------- planning

    def _plan(self, head: str, state: dict | None) -> _Plan:
        if self.full_reindex:
            return self._reconcile_plan(head, "full", "full reindex requested", force=True)
        if state is None:
            return self._reconcile_plan(head, "reconcile", "no previous sync state")
        if state.get("chunking_version") != self.chunking_version:
            return self._reconcile_plan(head, "reconcile", "chunking/embedding configuration changed")
        if state.get("selection") != self.selector.fingerprint:
            return self._reconcile_plan(head, "reconcile", "file selection changed")
        base = state.get("commit")
        if base == head:
            return _Plan("noop", "commit already indexed", base=base)
        if not base or not self.repo.has_commit(base):
            return self._reconcile_plan(head, "reconcile", f"indexed commit {base} not in local history")
        if self.repo.is_ancestor(head, base):
            return _Plan("skipped", f"newer commit {base} already indexed", base=base)
        return self._incremental_plan(base, head)

    def _reconcile_plan(self, head: str, mode: str, reason: str, force: bool = False) -> _Plan:
        files = [p for p in self.repo.regular_files(head) if self.selector.matches(p)]
        return _Plan(mode, reason, upserts={p: "?" for p in files}, prune_except=set(files), force=force)

    def _incremental_plan(self, base: str, head: str) -> _Plan:
        plan = _Plan("incremental", "git diff since last indexed commit", base=base)
        changes: list[Change] = self.repo.diff(base, head)
        for ch in changes:
            new_ok = ch.status != "D" and self.selector.matches(ch.path)
            old_path = ch.old_path if ch.status == "R" else ch.path
            old_ok = ch.status != "A" and self.selector.matches(old_path)
            if ch.status == "R":
                if old_ok:
                    plan.deletes.add(old_path)
                if new_ok:
                    plan.upserts[ch.path] = "R" if old_ok else "A"
                    if old_ok:
                        plan.renamed_from.add(old_path)
            elif ch.status == "D":
                if old_ok:
                    plan.deletes.add(ch.path)
            elif new_ok:
                plan.upserts[ch.path] = ch.status
        # Paths that stopped being regular files (e.g. became symlinks) are deletions.
        regular = set(self.repo.regular_files(head, sorted(plan.upserts)))
        for path in [p for p in plan.upserts if p not in regular]:
            del plan.upserts[path]
            plan.deletes.add(path)
        plan.deletes -= set(plan.upserts)
        return plan

    # ---------------------------------------------------------------- deletes

    def _scope_filter(self, *conditions: models.Condition, must_not: list[models.Condition] | None = None):
        return models.Filter(
            must=[
                _match("kind", KIND_CHUNK),
                _match("repository", self.source.repository),
                _match("scope", self.source.scope),
                *conditions,
            ],
            must_not=must_not,
        )

    def _delete_paths(self, plan: _Plan, stats: Stats) -> None:
        if plan.prune_except is not None:
            flt = self._scope_filter(must_not=[_match_any("path", plan.prune_except)] if plan.prune_except else None)
            stale_paths = self._stage("delete", lambda: {r.payload["path"] for r in self._scroll(flt, ["path"])})
            stats.files_deleted += len(stale_paths)
        else:
            if not plan.deletes:
                return
            flt = self._scope_filter(_match_any("path", plan.deletes))
            stats.files_deleted += len(plan.deletes - plan.renamed_from)
        stats.chunks_deleted += self._stage("delete", lambda: self._delete_filter(flt))

    def _delete_filter(self, flt: models.Filter) -> int:
        count = self.client.count(self.collection, count_filter=flt, exact=True).count
        if count:
            self.client.delete(self.collection, points_selector=models.FilterSelector(filter=flt), wait=True)
        return count

    # ---------------------------------------------------------------- upserts

    def _upsert_files(self, head: str, plan: _Plan, stats: Stats) -> None:
        paths = sorted(plan.upserts)
        for i in range(0, len(paths), self.lookup_batch):
            batch = paths[i : i + self.lookup_batch]
            blobs = self._stage("read", lambda: self.repo.read_blobs(head, batch))
            existing = self._stage("lookup", lambda: self._existing(batch))
            for path in batch:
                changed = self._sync_file(head, path, blobs[path], existing.get(path, []), plan.force, stats)
                status = plan.upserts[path]
                if status == "?":
                    status = "M" if existing.get(path) else "A"
                if not changed and status in ("M", "?"):
                    stats.files_unchanged += 1
                elif status == "A":
                    stats.files_added += 1
                elif status == "R":
                    stats.files_renamed += 1
                else:
                    stats.files_modified += 1

    def _existing(self, paths: list[str]) -> dict[str, list[models.Record]]:
        grouped: dict[str, list[models.Record]] = {}
        for rec in self._scroll(self._scope_filter(_match_any("path", paths)), _EXISTING_FIELDS):
            grouped.setdefault(rec.payload["path"], []).append(rec)
        return grouped

    def _sync_file(
        self, head: str, path: str, content: bytes, existing: list[models.Record], force: bool, stats: Stats
    ) -> bool:
        """Bring one file's points in line with `content`. Returns False if nothing had to change."""
        file_hash = _hash(self.chunking_version, content)
        if not force and _consistent(existing, file_hash):
            return False

        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("utf-8", errors="replace")
            self.log(f"warning: {path} is not valid UTF-8; undecodable bytes replaced")
        doc = self._stage("chunk", lambda: self.chunker.chunk(path, text), path)

        existing_ids = {str(r.id) for r in existing}
        occurrences: Counter[str] = Counter()
        new_points: list[_NewPoint] = []
        updates: list[tuple[str, dict]] = []
        keep: set[str] = set()
        for index, chunk in enumerate(doc.chunks):
            text_to_embed = embed_text(doc.title, chunk)
            chunk_hash = _hash(self.chunking_version, text_to_embed)
            occurrence = occurrences[chunk_hash]
            occurrences[chunk_hash] += 1
            point_id = str(
                uuid.uuid5(
                    POINT_NAMESPACE,
                    "\x1f".join((self.source.repository, self.source.scope, path, chunk_hash, str(occurrence))),
                )
            )
            keep.add(point_id)
            payload = self._payload(head, path, doc.title, chunk, index, len(doc.chunks), file_hash, chunk_hash)
            if point_id in existing_ids and not force:
                updates.append((point_id, payload))
            else:
                new_points.append(_NewPoint(point_id, chunk_hash, text_to_embed, payload))

        # Order matters for crash safety: new points first, stale ones last. An interrupted run
        # leaves the file inconsistent (mixed file_hash / chunk_count), so the retry redoes it.
        self._write_new_points(path, new_points, force, stats)
        self._stage("payload", lambda: self._set_payloads(updates), path)
        stats.chunks_updated += len(updates)
        stale = sorted(existing_ids - keep)
        if stale:
            self._stage(
                "delete",
                lambda: self.client.delete(
                    self.collection, points_selector=models.PointIdsList(points=stale), wait=True
                ),
                path,
            )
            stats.chunks_deleted += len(stale)
        return True

    def _payload(
        self,
        head: str,
        path: str,
        title: str | None,
        chunk: Chunk,
        index: int,
        count: int,
        file_hash: str,
        chunk_hash: str,
    ) -> dict:
        return {
            "kind": KIND_CHUNK,
            "repository": self.source.repository,
            "scope": self.source.scope,
            "path": path,
            "commit": head,
            "title": title,
            "headings": list(chunk.headings),
            "heading": chunk.headings[-1] if chunk.headings else None,
            "anchor": chunk.anchor,
            "chunk_index": index,
            "chunk_count": count,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "text": chunk.text,
            "source_url": self.source.url(head, path, chunk.anchor),
            "file_hash": file_hash,
            "chunk_hash": chunk_hash,
            "chunking_version": self.chunking_version,
            "indexer_version": INDEXER_VERSION,
        }

    def _write_new_points(self, path: str, new_points: list[_NewPoint], force: bool, stats: Stats) -> None:
        if not new_points:
            return
        cached = {} if force else self._stage("lookup", lambda: self._cached_vectors(new_points), path)
        to_embed = [p for p in new_points if p.chunk_hash not in cached]
        embedded: dict[str, list[float]] = {}
        if to_embed:
            vectors = self._stage("embed", lambda: self.embedder.embed([p.embed_text for p in to_embed]), path)
            for p, vec in zip(to_embed, vectors, strict=True):
                if len(vec) != self._dimension:
                    raise SyncError("embed", f"vector has {len(vec)} dimensions, collection expects {self._dimension}", path)
                embedded[p.id] = vec
        stats.chunks_embedded += len(to_embed)
        stats.chunks_reused += len(new_points) - len(to_embed)

        points = [
            models.PointStruct(
                id=p.id,
                vector=self._vector(embedded[p.id] if p.id in embedded else cached[p.chunk_hash]),
                payload=p.payload,
            )
            for p in new_points
        ]
        for i in range(0, len(points), self.upsert_batch):
            batch = points[i : i + self.upsert_batch]
            self._stage("upsert", lambda: self.client.upsert(self.collection, points=batch, wait=True), path)

    def _cached_vectors(self, new_points: list[_NewPoint]) -> dict[str, list[float]]:
        """Vectors already stored anywhere in the collection for identical embedded text."""
        wanted = {p.chunk_hash for p in new_points}
        found: dict[str, list[float]] = {}
        hashes = sorted(wanted)
        for i in range(0, len(hashes), 256):
            flt = models.Filter(must=[_match("kind", KIND_CHUNK), _match_any("chunk_hash", hashes[i : i + 256])])
            for rec in self._scroll(flt, ["chunk_hash"], with_vectors=[self.vector_name] if self.vector_name else True):
                vec = rec.vector.get(self.vector_name) if isinstance(rec.vector, dict) else rec.vector
                if isinstance(vec, list) and len(vec) == self._dimension:
                    found.setdefault(rec.payload["chunk_hash"], vec)
        return found

    def _set_payloads(self, updates: list[tuple[str, dict]]) -> None:
        ops = [
            models.SetPayloadOperation(set_payload=models.SetPayload(payload=payload, points=[pid]))
            for pid, payload in updates
        ]
        for i in range(0, len(ops), self.upsert_batch):
            self.client.batch_update_points(self.collection, update_operations=ops[i : i + self.upsert_batch], wait=True)

    # ---------------------------------------------------------------- state

    def _read_state(self) -> dict | None:
        recs = self.client.retrieve(self.collection, ids=[self.state_id], with_payload=True, with_vectors=False)
        return dict(recs[0].payload) if recs else None

    def _commit_state(self, head: str, before: dict | None, stats: Stats) -> None:
        current = self._read_state()
        current_commit = current.get("commit") if current else None
        before_commit = before.get("commit") if before else None
        if (
            current_commit
            and current_commit != before_commit
            and self.repo.has_commit(current_commit)
            and self.repo.is_ancestor(head, current_commit)
        ):
            self.log(f"not recording {head}: a concurrent run already recorded newer commit {current_commit}")
            return
        payload = {
            "kind": KIND_STATE,
            "repository": self.source.repository,
            "scope": self.source.scope,
            "commit": head,
            "chunking_version": self.chunking_version,
            "selection": self.selector.fingerprint,
            "indexer_version": INDEXER_VERSION,
        }
        unit = [1.0] + [0.0] * (self._dimension - 1)
        self.client.upsert(
            self.collection,
            points=[models.PointStruct(id=self.state_id, vector=self._vector(unit), payload=payload)],
            wait=True,
        )
        stats.state_written = True

    # ---------------------------------------------------------------- helpers

    def _vector(self, vec: list[float]):
        return {self.vector_name: vec} if self.vector_name else vec

    def _vector_dimension(self) -> int:
        if not self.client.collection_exists(self.collection):
            raise SyncError("collection", f"collection '{self.collection}' does not exist; create it before syncing")
        vectors = self.client.get_collection(self.collection).config.params.vectors
        if self.vector_name:
            if not isinstance(vectors, dict) or self.vector_name not in vectors:
                names = sorted(vectors) if isinstance(vectors, dict) else ["<unnamed>"]
                raise SyncError("collection", f"dense vector '{self.vector_name}' not found; collection has {names}")
            return vectors[self.vector_name].size
        if isinstance(vectors, dict):
            raise SyncError("collection", f"collection uses named vectors {sorted(vectors)}; set vector-name")
        if vectors is None:
            raise SyncError("collection", "collection has no dense vector configured")
        return vectors.size

    def _ensure_indexes(self) -> None:
        for field_name in KEYWORD_INDEXES:
            try:
                self.client.create_payload_index(
                    self.collection, field_name=field_name, field_schema=models.PayloadSchemaType.KEYWORD, wait=True
                )
            except Exception as exc:  # optimisation only; sync stays correct without indexes
                self.log(f"warning: could not create payload index on '{field_name}': {type(exc).__name__}")

    def _scroll(self, flt: models.Filter, fields: list[str], with_vectors: bool | list[str] = False):
        offset = None
        while True:
            records, offset = self.client.scroll(
                self.collection,
                scroll_filter=flt,
                limit=256,
                offset=offset,
                with_payload=fields,
                with_vectors=with_vectors,
            )
            yield from records
            if offset is None:
                return

    def _stage(self, stage: str, fn: Callable, path: str | None = None):
        try:
            return fn()
        except SyncError:
            raise
        except Exception as exc:
            raise SyncError(stage, _describe(exc), path) from exc


def _consistent(existing: list[models.Record], file_hash: str) -> bool:
    if not existing:
        return False
    n = len(existing)
    return all(r.payload.get("file_hash") == file_hash and r.payload.get("chunk_count") == n for r in existing) and sorted(
        r.payload.get("chunk_index") for r in existing
    ) == list(range(n))


def _describe(exc: Exception) -> str:
    # Client exceptions can embed request URLs/headers; keep the type and the first line only.
    first_line = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    return f"{type(exc).__name__}: {first_line[:300]}" if first_line else type(exc).__name__
