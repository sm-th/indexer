"""Deterministic, heading-aware Markdown chunking.

The document is parsed into a heading tree (ATX headings outside fenced code).
A section that fits the token budget becomes one chunk; an oversized section is
split into its own intro plus its child sections, and adjacent whole pieces are
packed back together while they fit. Oversized leaf text is split on block
boundaries (paragraphs, fenced code kept whole), and a single oversized block is
split by Chonkie's RecursiveChunker.

Every chunk is a verbatim span of the normalized document text.
"""

from __future__ import annotations

import bisect
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Protocol

import tiktoken
from chonkie import RecursiveChunker
from chonkie.tokenizer import Tokenizer as ChonkieTokenizer

MARKDOWN_CHUNKER_VERSION = "markdown-v3"

_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_ATX_RE = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?(?:[ \t]+#+)?[ \t]*$")
_FRONTMATTER_TITLE_RE = re.compile(r"^title:[ \t]*(.+?)[ \t]*$", re.MULTILINE)
_LINK_RE = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_SLUG_DROP_RE = re.compile(r"[^\w\- ]", re.UNICODE)
_LEADING_BLANK_RE = re.compile(r"(?:[ \t]*\n)*")


@dataclass(frozen=True)
class Chunk:
    """One indexable span of a document."""

    text: str
    headings: tuple[str, ...]
    anchor: str | None
    start_line: int
    end_line: int


@dataclass(frozen=True)
class ChunkedDocument:
    title: str | None
    chunks: list[Chunk]


class Chunker(Protocol):
    @property
    def fingerprint(self) -> str:
        """Stable description of everything that changes chunk output."""
        ...

    def chunk(self, path: str, text: str) -> ChunkedDocument: ...


class _DataTokenizer(ChonkieTokenizer):
    """tiktoken wrapper that treats special-token text as plain data."""

    def __init__(self, encoding: tiktoken.Encoding) -> None:
        super().__init__()
        self._enc = encoding

    def __repr__(self) -> str:
        return f"_DataTokenizer({self._enc.name})"

    def encode(self, text: str) -> Sequence[int]:
        return self._enc.encode(text, disallowed_special=())

    def decode(self, tokens: Sequence[int]) -> str:
        return self._enc.decode(list(tokens))

    def tokenize(self, text: str) -> Sequence[str | int]:
        return self.encode(text)

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))


@dataclass
class _Section:
    level: int
    title: str | None
    headings: tuple[str, ...]
    anchor: str | None
    start: int  # offset of heading line (or document body start for root)
    body_end: int  # end of heading + intro text, before the first child
    end: int = 0
    children: list[_Section] = field(default_factory=list)


@dataclass(frozen=True)
class _Piece:
    start: int
    end: int
    headings: tuple[str, ...]
    anchor: str | None
    mergeable: bool


def github_slug(heading: str) -> str:
    text = _LINK_RE.sub(r"\1", heading)
    text = text.replace("`", "").replace("*", "")
    text = _SLUG_DROP_RE.sub("", text.strip().lower())
    return text.replace(" ", "-")


class MarkdownChunker:
    def __init__(self, max_tokens: int = 256, encoding: str = "cl100k_base") -> None:
        if max_tokens < 16:
            raise ValueError("max_tokens must be at least 16")
        self.max_tokens = max_tokens
        self.encoding_name = encoding
        self._tokenizer = _DataTokenizer(tiktoken.get_encoding(encoding))
        self._splitters: dict[int, RecursiveChunker] = {}

    def _splitter(self, budget: int) -> RecursiveChunker:
        splitter = self._splitters.get(budget)
        if splitter is None:
            splitter = RecursiveChunker(tokenizer=self._tokenizer, chunk_size=budget, min_characters_per_chunk=1)
            self._splitters[budget] = splitter
        return splitter

    @cached_property
    def fingerprint(self) -> str:
        return json.dumps(
            {"chunker": MARKDOWN_CHUNKER_VERSION, "max_tokens": self.max_tokens, "encoding": self.encoding_name},
            sort_keys=True,
        )

    def _tokens(self, text: str) -> int:
        return self._tokenizer.count_tokens(text)

    def chunk(self, path: str, text: str) -> ChunkedDocument:
        text = text.removeprefix("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
        line_starts = [0] + [m.end() for m in re.finditer("\n", text)]
        body_start, fm_title = _frontmatter(text)
        root = _parse_sections(text, body_start)
        title = fm_title or _first_h1(root) or None

        pieces = self._emit(text, root)
        chunks: list[Chunk] = []
        for p in pieces:
            start, end = _trim(text, p.start, p.end)
            if start >= end:
                continue
            chunks.append(
                Chunk(
                    text=text[start:end],
                    headings=p.headings,
                    anchor=p.anchor,
                    start_line=bisect.bisect_right(line_starts, start),
                    end_line=bisect.bisect_right(line_starts, end - 1),
                )
            )
        return ChunkedDocument(title=title, chunks=chunks)

    def _emit(self, text: str, node: _Section) -> list[_Piece]:
        if not text[node.start : node.end].strip():
            return []
        if self._tokens(text[node.start : node.end]) <= self.max_tokens:
            # A document that is just one top-level section belongs to that section's heading.
            lead = node
            while len(lead.children) == 1 and not text[lead.start : lead.body_end].strip():
                lead = lead.children[0]
            return [_Piece(node.start, node.end, lead.headings, lead.anchor, mergeable=True)]

        pieces: list[_Piece] = []
        if text[node.start : node.body_end].strip():
            pieces.extend(self._split_blocks(text, node.start, node.body_end, node.headings, node.anchor))
        for child in node.children:
            pieces.extend(self._emit(text, child))
        packed = self._pack(text, pieces)
        # A heading with no intro text leads into its first subsection instead of standing alone.
        if (
            len(packed) > 1
            and _ATX_RE.match(text[packed[0].start : packed[0].end].strip())
            and packed[0].end == packed[1].start
            and self._tokens(text[packed[0].start : packed[1].end]) <= self.max_tokens
        ):
            packed[0:2] = [_Piece(packed[0].start, packed[1].end, packed[1].headings, packed[1].anchor, False)]
        # Fragments of a split section must not be glued onto neighbouring sections.
        return [_Piece(p.start, p.end, p.headings, p.anchor, mergeable=False) for p in packed]

    def _pack(self, text: str, pieces: list[_Piece]) -> list[_Piece]:
        """Merge runs of adjacent whole pieces that fit together under their common heading."""
        out: list[_Piece] = []
        for piece in pieces:
            prev = out[-1] if out else None
            if (
                prev is not None
                and prev.mergeable
                and piece.mergeable
                and prev.end == piece.start
                and self._tokens(text[prev.start : piece.end]) <= self.max_tokens
            ):
                # Breadcrumb: what all packed sections share. Anchor: where the chunk starts, so links land on it.
                common = _common_prefix(prev.headings, piece.headings)
                out[-1] = _Piece(prev.start, piece.end, common, prev.anchor, mergeable=True)
            else:
                out.append(piece)
        return out

    def _split_blocks(
        self, text: str, start: int, end: int, headings: tuple[str, ...], anchor: str | None
    ) -> list[_Piece]:
        pieces: list[_Piece] = []
        blocks = _blocks(text, start, end)
        # A heading line travels with the first block after it (lead = heading span).
        lead_start: int | None = None
        if len(blocks) > 1 and _ATX_RE.match(text[blocks[0][0] : blocks[0][1]].strip()):
            lead_start = blocks.pop(0)[0]
        for b_start, b_end in blocks:
            first = lead_start if lead_start is not None else b_start
            lead_tokens = self._tokens(text[first:b_start]) if first != b_start else 0
            lead_start = None
            if self._tokens(text[first:b_end]) <= self.max_tokens:
                pieces.append(_Piece(first, b_end, headings, anchor, mergeable=True))
                continue
            budget = self.max_tokens - lead_tokens
            if budget < self.max_tokens // 2:  # absurdly long heading: split it like any text
                b_start, budget = first, self.max_tokens
            for i, c in enumerate(self._splitter(budget).chunk(text[b_start:b_end])):
                c_start = first if i == 0 else b_start + c.start_index
                # Same section: _pack below may re-fill fragments up to the budget.
                pieces.append(_Piece(c_start, b_start + c.end_index, headings, anchor, mergeable=True))
        return self._pack(text, pieces)


def _frontmatter(text: str) -> tuple[int, str | None]:
    if not text.startswith("---\n"):
        return 0, None
    m = re.compile(r"^(---|\.\.\.)[ \t]*$", re.MULTILINE).search(text, 4)
    if m is None:
        return 0, None
    end = m.end() + (1 if text[m.end() : m.end() + 1] == "\n" else 0)
    title_m = _FRONTMATTER_TITLE_RE.search(text[4 : m.start()])
    title = title_m.group(1).strip("'\"").strip() if title_m else None
    return end, title or None


def _iter_lines(text: str, start: int, end: int):
    pos = start
    while pos < end:
        nl = text.find("\n", pos, end)
        line_end = end if nl == -1 else nl + 1
        yield pos, text[pos:line_end].rstrip("\n")
        pos = line_end


def _parse_sections(text: str, body_start: int) -> _Section:
    root = _Section(level=0, title=None, headings=(), anchor=None, start=body_start, body_end=len(text))
    stack = [root]
    slug_counts: dict[str, int] = {}
    fence: tuple[str, int] | None = None
    for pos, line in _iter_lines(text, body_start, len(text)):
        fm = _FENCE_RE.match(line)
        if fence is not None:
            if fm and fm.group(1)[0] == fence[0] and len(fm.group(1)) >= fence[1] and not fm.group(2).strip():
                fence = None
            continue
        if fm:
            fence = (fm.group(1)[0], len(fm.group(1)))
            continue
        hm = _ATX_RE.match(line)
        if not hm:
            continue
        level = len(hm.group(1))
        title = (hm.group(2) or "").strip()
        while stack[-1].level >= level:
            done = stack.pop()
            done.end = pos
            if not done.children:
                done.body_end = pos
        parent = stack[-1]
        if not parent.children:
            parent.body_end = pos
        slug = github_slug(title)
        n = slug_counts.get(slug, 0)
        slug_counts[slug] = n + 1
        node = _Section(
            level=level,
            title=title,
            headings=parent.headings + (title,),
            anchor=slug if n == 0 else f"{slug}-{n}",
            start=pos,
            body_end=len(text),
        )
        parent.children.append(node)
        stack.append(node)
    for node in stack:
        node.end = len(text)
    return root


def _first_h1(root: _Section) -> str | None:
    for child in root.children:
        if child.level == 1 and child.title:
            return child.title
    return None


def _blocks(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Split a span into blank-line separated blocks, keeping fenced code intact."""
    blocks: list[tuple[int, int]] = []
    block_start: int | None = None
    fence: tuple[str, int] | None = None
    last_end = start
    for pos, line in _iter_lines(text, start, end):
        line_end = pos + len(line) + 1
        fm = _FENCE_RE.match(line)
        if fence is not None:
            if fm and fm.group(1)[0] == fence[0] and len(fm.group(1)) >= fence[1] and not fm.group(2).strip():
                fence = None
            last_end = line_end
            continue
        if fm:
            fence = (fm.group(1)[0], len(fm.group(1)))
            if block_start is None:
                block_start = pos
            last_end = line_end
            continue
        if not line.strip():
            if block_start is not None:
                blocks.append((block_start, min(last_end, end)))
                block_start = None
            continue
        if block_start is None:
            block_start = pos
        last_end = line_end
    if block_start is not None:
        blocks.append((block_start, min(last_end, end)))
    # Extend each block to the next block start so that packed runs are contiguous spans.
    return [(s, blocks[i + 1][0] if i + 1 < len(blocks) else end) for i, (s, _) in enumerate(blocks)]


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    """Drop leading blank lines (and leading blanks of a mid-line split) and trailing whitespace."""
    start = _LEADING_BLANK_RE.match(text, start, end).end()
    if start > 0 and text[start - 1] != "\n":
        while start < end and text[start] in " \t":
            start += 1
    while end > start and text[end - 1] in " \t\n":
        end -= 1
    return start, end


def _common_prefix(a: tuple[str, ...], b: tuple[str, ...]) -> tuple[str, ...]:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return a[:n]
