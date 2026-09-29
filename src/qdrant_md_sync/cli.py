"""Command-line entry point; the GitHub Action passes its inputs as environment variables."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from qdrant_client import QdrantClient

from .chunking import MarkdownChunker
from .embeddings import OpenAIEmbeddings
from .gitrepo import DEFAULT_EXCLUDE, DEFAULT_INCLUDE, FileSelector, GitError, GitRepo
from .sync import Source, Stats, SyncError, Syncer

DEFAULT_MODEL = "text-embedding-3-small"


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, "")
    return value if value.strip() else default


def _globs(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(g.strip() for line in value.splitlines() for g in line.split(",") if g.strip())


def _bool(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qdrant-md-sync",
        description="Synchronize a Git repository's Markdown files into a Qdrant collection.",
    )
    p.add_argument("--path", default=_env("QMS_PATH", "."), help="checked-out repository (default: .)")
    p.add_argument("--qdrant-url", default=_env("QDRANT_URL"), help="env: QDRANT_URL")
    p.add_argument("--qdrant-api-key", default=_env("QDRANT_API_KEY"), help="env: QDRANT_API_KEY")
    p.add_argument("--collection", default=_env("QMS_COLLECTION"), help="existing collection (env: QMS_COLLECTION)")
    p.add_argument("--vector-name", default=_env("QMS_VECTOR_NAME"), help="named dense vector; empty = unnamed")
    p.add_argument("--repository", default=_env("QMS_REPOSITORY", _env("GITHUB_REPOSITORY")), help="owner/name")
    p.add_argument("--scope", default=_env("QMS_SCOPE", _env("GITHUB_REF_NAME")), help="branch or source scope")
    p.add_argument("--server-url", default=_env("QMS_SERVER_URL", _env("GITHUB_SERVER_URL", "https://github.com")))
    p.add_argument("--include", default=_env("QMS_INCLUDE"), help="newline/comma separated gitignore-style globs")
    p.add_argument("--exclude", default=_env("QMS_EXCLUDE"), help="extra excludes, added to the defaults")
    p.add_argument("--embedding-model", default=_env("QMS_EMBEDDING_MODEL", DEFAULT_MODEL))
    p.add_argument("--embedding-api-key", default=_env("QMS_EMBEDDING_API_KEY"), help="optional for local servers")
    p.add_argument("--embedding-base-url", default=_env("QMS_EMBEDDING_BASE_URL", "https://api.openai.com/v1"))
    p.add_argument("--embedding-dimensions", type=int, default=_env("QMS_EMBEDDING_DIMENSIONS"))
    p.add_argument("--max-tokens", type=int, default=int(_env("QMS_MAX_TOKENS", "512")))
    p.add_argument("--tokenizer", default=_env("QMS_TOKENIZER", "cl100k_base"), help="tiktoken encoding")
    p.add_argument("--seed", default=_env("QMS_SEED", ""), help="change to force re-chunking and re-embedding")
    p.add_argument("--full-reindex", action="store_true", default=_bool(_env("QMS_FULL_REINDEX")))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    in_actions = _bool(os.environ.get("GITHUB_ACTIONS"))
    if in_actions:
        for secret in (args.qdrant_api_key, args.embedding_api_key):
            if secret:
                print(f"::add-mask::{secret}", flush=True)
    missing = [n for n in ("qdrant_url", "collection", "repository", "scope") if not getattr(args, n)]
    if missing:
        print(f"error: missing required settings: {', '.join('--' + m.replace('_', '-') for m in missing)}", file=sys.stderr)
        return 2

    embedder = OpenAIEmbeddings(
        model=args.embedding_model,
        api_key=args.embedding_api_key or "",
        base_url=args.embedding_base_url,
        dimensions=args.embedding_dimensions,
    )
    client = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key, timeout=120)
    syncer = Syncer(
        client=client,
        collection=args.collection,
        repo=GitRepo(Path(args.path)),
        source=Source(args.repository, args.scope, args.server_url),
        chunker=MarkdownChunker(max_tokens=args.max_tokens, encoding=args.tokenizer),
        embedder=embedder,
        selector=FileSelector(
            include=_globs(args.include) or DEFAULT_INCLUDE,
            exclude=DEFAULT_EXCLUDE + _globs(args.exclude),
        ),
        vector_name=args.vector_name,
        seed=args.seed,
        full_reindex=args.full_reindex,
        log=lambda msg: print(msg, file=sys.stderr, flush=True),
    )
    try:
        stats = syncer.run()
    except (SyncError, GitError) as exc:
        stage = getattr(exc, "stage", "git")
        path = getattr(exc, "path", None)
        if in_actions:
            location = f"file={path}," if path else ""
            print(f"::error {location}title=qdrant-md-sync failed at stage '{stage}'::{exc}", flush=True)
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(stats.as_dict(), indent=2))
    _write_github_files(stats)
    return 0


def _write_github_files(stats: Stats) -> None:
    if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(render_summary(stats))
    if output_path := os.environ.get("GITHUB_OUTPUT"):
        with open(output_path, "a", encoding="utf-8") as f:
            for key, value in stats.as_dict().items():
                f.write(f"{key.replace('_', '-')}={'' if value is None else str(value).lower() if isinstance(value, bool) else value}\n")


def render_summary(s: Stats) -> str:
    base = f"`{s.base_commit[:12]}` → " if s.base_commit and s.mode == "incremental" else ""
    rows = [
        ("Revision", f"{base}`{s.commit[:12]}`"),
        ("Mode", f"{s.mode} — {s.reason}"),
        ("Files added / modified / renamed / deleted", f"{s.files_added} / {s.files_modified} / {s.files_renamed} / {s.files_deleted}"),
        ("Files unchanged (skipped by hash)", str(s.files_unchanged)),
        ("Chunks embedded / vectors reused", f"{s.chunks_embedded} / {s.chunks_reused}"),
        ("Chunks updated in place (payload only)", str(s.chunks_updated)),
        ("Stale chunks deleted", str(s.chunks_deleted)),
        ("Duration", f"{s.duration_seconds:.1f}s"),
        ("Revision recorded", "yes" if s.state_written else "no"),
    ]
    lines = ["### Qdrant Markdown sync", "", "| | |", "|---|---|", *(f"| {k} | {v} |" for k, v in rows), ""]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
