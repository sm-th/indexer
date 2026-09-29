"""Read-only Git access and file selection for the checked-out repository."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pathspec

DEFAULT_INCLUDE = ("**/*.md",)
DEFAULT_EXCLUDE = (
    ".git/",
    "node_modules/",
    "vendor/",
    "third_party/",
    "dist/",
    "build/",
    "out/",
    "target/",
    "site/",
    "_site/",
    ".venv/",
    "venv/",
    "__pycache__/",
    ".next/",
    ".docusaurus/",
)


class GitError(RuntimeError):
    pass


class FileSelector:
    def __init__(self, include: tuple[str, ...] = DEFAULT_INCLUDE, exclude: tuple[str, ...] = DEFAULT_EXCLUDE):
        self.include = tuple(include)
        self.exclude = tuple(exclude)
        self._include = pathspec.GitIgnoreSpec.from_lines(self.include)
        self._exclude = pathspec.GitIgnoreSpec.from_lines(self.exclude)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps({"include": self.include, "exclude": self.exclude}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def matches(self, path: str) -> bool:
        return self._include.match_file(path) and not self._exclude.match_file(path)


@dataclass(frozen=True)
class Change:
    """One entry of `git diff --name-status`: status is A, M, D or R (copies arrive as A)."""

    status: str
    path: str
    old_path: str | None = None


class GitRepo:
    def __init__(self, root: Path):
        self.root = root

    def _git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
        proc = subprocess.run(["git", "-C", str(self.root), *args], capture_output=True)
        if check and proc.returncode != 0:
            raise GitError(f"git {' '.join(args)} failed: {proc.stderr.decode(errors='replace').strip()}")
        return proc

    def head(self) -> str:
        return self._git("rev-parse", "HEAD").stdout.decode().strip()

    def has_commit(self, sha: str) -> bool:
        return self._git("cat-file", "-e", f"{sha}^{{commit}}", check=False).returncode == 0

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        proc = self._git("merge-base", "--is-ancestor", ancestor, descendant, check=False)
        if proc.returncode not in (0, 1):
            raise GitError(f"git merge-base failed: {proc.stderr.decode(errors='replace').strip()}")
        return proc.returncode == 0

    def regular_files(self, rev: str, paths: list[str] | None = None) -> list[str]:
        """Regular (non-symlink, non-submodule) files at `rev`, optionally limited to `paths`."""
        args = ["ls-tree", "-r", "-z", "--full-tree", rev]
        if paths is not None:
            if not paths:
                return []
            args += ["--", *(f":(literal){p}" for p in paths)]
        out = self._git(*args).stdout
        files = []
        for entry in out.split(b"\0"):
            if not entry:
                continue
            meta, _, name = entry.partition(b"\t")
            mode, kind, _ = meta.split(b" ", 2)
            if kind == b"blob" and mode in (b"100644", b"100755"):
                files.append(name.decode())
        return sorted(files)

    def read_blobs(self, rev: str, paths: list[str]) -> dict[str, bytes]:
        """Read many files at `rev` through one `git cat-file --batch` process."""
        blobs: dict[str, bytes] = {}
        if not paths:
            return blobs
        proc = subprocess.Popen(
            ["git", "-C", str(self.root), "cat-file", "--batch"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert proc.stdin is not None and proc.stdout is not None
        try:
            for path in paths:
                proc.stdin.write(f"{rev}:{path}\n".encode())
                proc.stdin.flush()
                header = proc.stdout.readline().decode().split()
                if len(header) != 3 or header[1] != "blob":
                    raise GitError(f"cannot read {path} at {rev}: {' '.join(header)}")
                size = int(header[2])
                blobs[path] = proc.stdout.read(size)
                proc.stdout.read(1)  # trailing newline
        finally:
            proc.stdin.close()
            proc.wait()
        return blobs

    def diff(self, base: str, head: str) -> list[Change]:
        out = self._git("diff", "--name-status", "-z", "-M", base, head).stdout
        fields = [f.decode() for f in out.split(b"\0") if f]
        changes: list[Change] = []
        i = 0
        while i < len(fields):
            status = fields[i]
            code = status[0]
            if code in "RC":
                old, new = fields[i + 1], fields[i + 2]
                i += 3
                changes.append(Change("R", new, old) if code == "R" else Change("A", new))
            else:
                path = fields[i + 1]
                i += 2
                changes.append(Change({"T": "M"}.get(code, code), path))
        return changes
