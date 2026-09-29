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
