from qdrant_md_sync.chunking import MarkdownChunker

FAQ = "# FAQ\n\n## Rotating tokens\n\nRun `token rotate`.\n\n## Logs\n\nLogs go to stdout.\n"


def test_small_single_section_document_keeps_its_heading():
    [chunk] = MarkdownChunker(max_tokens=512).chunk("faq.md", FAQ).chunks
    assert chunk.headings == ("FAQ",)
    assert chunk.anchor == "faq"


def test_small_document_with_intro_or_several_top_sections_has_no_heading():
    for text in ("Intro line.\n\n" + FAQ, FAQ + "\n# Changelog\n\nNone yet.\n"):
        [chunk] = MarkdownChunker(max_tokens=512).chunk("doc.md", text).chunks
        assert chunk.headings == ()
        assert chunk.anchor is None


def test_packed_sections_link_to_the_section_where_the_chunk_starts():
    sections = "".join(f"## Part {i}\n\n" + "word " * 20 + "\n\n" for i in range(8))
    chunks = MarkdownChunker(max_tokens=64).chunk("guide.md", "# Guide\n\n" + sections).chunks
    packed = [c for c in chunks if c.text.count("## Part") > 1]
    assert packed, "fixture must produce chunks that pack several sections"
    for c in packed:
        first_heading = c.text.split("\n", 1)[0].lstrip("# ")
        assert c.headings == ("Guide",)
        assert c.anchor == first_heading.lower().replace(" ", "-")
    assert any(c.anchor != "guide" for c in packed)
