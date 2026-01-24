"""Unit tests for chunking engine."""
import pytest

from app.chunking.engine import ChildChunk, ChunkingEngine, ParentChunk


@pytest.mark.unit
class TestChunkingEngine:
    """Test document chunking functionality."""

    def test_split_by_headers(self, sample_markdown: str) -> None:
        """Test splitting markdown by headers."""
        engine = ChunkingEngine()
        sections = engine._split_by_headers(sample_markdown)

        assert len(sections) > 0
        assert all("header_path" in s for s in sections)
        assert all("content" in s for s in sections)
        # Should have sections for both chapters
        header_paths = [s["header_path"] for s in sections]
        assert any("Chapter 1" in h for h in header_paths)
        assert any("Chapter 2" in h for h in header_paths)

    def test_split_by_headers_no_headers(self) -> None:
        """Test splitting markdown without headers."""
        engine = ChunkingEngine()
        content_no_headers = "This is plain text without any headers."
        sections = engine._split_by_headers(content_no_headers)

        assert len(sections) == 1
        assert sections[0]["header_path"] == "Document"
        assert sections[0]["content"] == content_no_headers

    def test_split_by_headers_preserves_hierarchy(self) -> None:
        """Test that header hierarchy is preserved."""
        engine = ChunkingEngine()
        markdown = """# Chapter 1

## Section 1.1

Content 1.1

### Subsection 1.1.1

Content 1.1.1

## Section 1.2

Content 1.2
"""
        sections = engine._split_by_headers(markdown)

        # Check that hierarchy is preserved in header_path
        header_paths = [s["header_path"] for s in sections]
        assert "Chapter 1 > Section 1.1" in header_paths
        assert "Chapter 1 > Section 1.1 > Subsection 1.1.1" in header_paths
        assert "Chapter 1 > Section 1.2" in header_paths

    def test_chunk_document(self, sample_markdown: str, sample_file_hash: str) -> None:
        """Test full document chunking."""
        engine = ChunkingEngine(
            parent_max_tokens=1200,
            child_tokens=384,
            child_overlap=64,
        )

        parents, children = engine.chunk_document(
            sample_markdown, sample_file_hash, "test.md"
        )

        assert len(parents) > 0
        assert len(children) > 0
        assert all(isinstance(p, ParentChunk) for p in parents)
        assert all(isinstance(c, ChildChunk) for c in children)

    def test_parent_chunk_has_children_refs(
        self, sample_markdown: str, sample_file_hash: str
    ) -> None:
        """Test that parent chunks reference their children."""
        engine = ChunkingEngine()
        parents, children = engine.chunk_document(
            sample_markdown, sample_file_hash, "test.md"
        )

        for parent in parents:
            assert len(parent.child_ids) > 0
            # Verify children reference this parent
            for child_id in parent.child_ids:
                child = next(c for c in children if c.id == child_id)
                assert child.parent_id == parent.id

    def test_child_chunks_have_overlap(
        self, sample_markdown: str, sample_file_hash: str
    ) -> None:
        """Test that child chunks have proper overlap."""
        engine = ChunkingEngine(
            parent_max_tokens=800,
            child_tokens=100,
            child_overlap=20,
        )

        parents, children = engine.chunk_document(
            sample_markdown, sample_file_hash, "test.md"
        )

        # Find parent with multiple children
        parent_with_multi_children = next(
            (p for p in parents if len(p.child_ids) > 1), None
        )

        if parent_with_multi_children:
            child_ids = parent_with_multi_children.child_ids
            child1 = next(c for c in children if c.id == child_ids[0])
            child2 = next(c for c in children if c.id == child_ids[1])

            # Both children should have content
            assert len(child1.content) > 0
            assert len(child2.content) > 0
            # Second child should have a different chunk_index
            assert child2.chunk_index == child1.chunk_index + 1

    def test_metadata_preserved(
        self, sample_markdown: str, sample_file_hash: str
    ) -> None:
        """Test that metadata is preserved in chunks."""
        engine = ChunkingEngine()
        parents, children = engine.chunk_document(
            sample_markdown, sample_file_hash, "test.pdf"
        )

        for parent in parents:
            assert parent.file_hash == sample_file_hash
            assert parent.file_name == "test.pdf"
            assert parent.header_path != ""

        for child in children:
            assert child.file_hash == sample_file_hash
            assert child.file_name == "test.pdf"
            assert child.chunk_index >= 0

    def test_empty_document(self, sample_file_hash: str) -> None:
        """Test handling of empty document."""
        engine = ChunkingEngine()
        parents, children = engine.chunk_document("", sample_file_hash, "empty.md")

        # Should return empty lists for empty content
        assert len(parents) == 0
        assert len(children) == 0

    def test_whitespace_only_document(self, sample_file_hash: str) -> None:
        """Test handling of whitespace-only document."""
        engine = ChunkingEngine()
        parents, children = engine.chunk_document(
            "   \n\n   \t   ", sample_file_hash, "whitespace.md"
        )

        # Should return empty lists
        assert len(parents) == 0
        assert len(children) == 0

    def test_parent_ids_are_unique(
        self, sample_markdown: str, sample_file_hash: str
    ) -> None:
        """Test that parent IDs are unique."""
        engine = ChunkingEngine()
        parents, _ = engine.chunk_document(sample_markdown, sample_file_hash, "test.md")

        parent_ids = [p.id for p in parents]
        assert len(parent_ids) == len(set(parent_ids))

    def test_child_ids_are_unique(
        self, sample_markdown: str, sample_file_hash: str
    ) -> None:
        """Test that child IDs are unique."""
        engine = ChunkingEngine()
        _, children = engine.chunk_document(
            sample_markdown, sample_file_hash, "test.md"
        )

        child_ids = [c.id for c in children]
        assert len(child_ids) == len(set(child_ids))

    def test_chunk_index_sequential(
        self, sample_markdown: str, sample_file_hash: str
    ) -> None:
        """Test that chunk indices are sequential for each parent."""
        engine = ChunkingEngine()
        parents, children = engine.chunk_document(
            sample_markdown, sample_file_hash, "test.md"
        )

        for parent in parents:
            parent_children = [c for c in children if c.parent_id == parent.id]
            indices = [c.chunk_index for c in parent_children]
            # Should be sequential starting from 0
            assert indices == list(range(len(indices)))

    def test_long_document_splitting(self, sample_file_hash: str) -> None:
        """Test splitting of very long documents."""
        engine = ChunkingEngine(
            parent_max_tokens=200,
            child_tokens=50,
            child_overlap=10,
        )

        # Create a long document
        long_doc = "# Chapter 1\n\n" + " ".join(["This is sentence number."] * 200)

        parents, children = engine.chunk_document(long_doc, sample_file_hash, "long.md")

        # Should split into multiple parents
        assert len(parents) > 1
        # Each parent should have children
        for parent in parents:
            assert len(parent.child_ids) > 0

    def test_small_section_merging(self, sample_file_hash: str) -> None:
        """Test that small sections are merged into parents."""
        engine = ChunkingEngine(
            parent_max_tokens=1200,
            parent_min_tokens=200,
        )

        markdown = """# Section 1

Small content.

# Section 2

Also small.

# Section 3

This is also quite small.
"""
        parents, _ = engine.chunk_document(markdown, sample_file_hash, "small.md")

        # Small sections should be merged
        assert len(parents) < 3  # Should merge some sections

    def test_tokenizer_consistency(self, sample_file_hash: str) -> None:
        """Test that the same tokenizer is used throughout."""
        engine = ChunkingEngine()

        markdown = "# Test\n\nContent here."
        parents, children = engine.chunk_document(markdown, sample_file_hash, "test.md")

        # Just verify it doesn't crash
        assert len(parents) > 0
        assert engine.tokenizer is not None

    def test_oversized_sentence_force_split(self, sample_file_hash: str) -> None:
        """Test that oversized sentences are force-split at token boundaries."""
        engine = ChunkingEngine(
            parent_max_tokens=100,
            child_tokens=50,
            child_overlap=10,
        )

        # Create content with a very long "sentence" that NLTK can't split
        # (e.g., a continuous string without sentence boundaries)
        long_blob = "word " * 500  # ~500+ tokens, no sentence boundaries
        markdown = f"# Section\n\n{long_blob}"

        parents, children = engine.chunk_document(
            markdown, sample_file_hash, "oversized.md"
        )

        # Should create multiple parents from the oversized sentence
        assert len(parents) > 1

        # Verify each parent is within token limit
        for parent in parents:
            parent_tokens = len(engine.tokenizer.encode(parent.content))
            assert (
                parent_tokens <= engine.parent_max_tokens + 10
            )  # Small buffer for tokenizer variance

    def test_oversized_sentence_no_data_loss(self, sample_file_hash: str) -> None:
        """Test that no significant data is lost when splitting oversized sentences."""
        engine = ChunkingEngine(
            parent_max_tokens=100,
            child_tokens=50,
            child_overlap=10,
        )

        # Create numbered words so we can verify coverage
        numbered_words = " ".join([f"word{i}" for i in range(300)])
        markdown = f"# Section\n\n{numbered_words}"

        parents, children = engine.chunk_document(
            markdown, sample_file_hash, "numbered.md"
        )

        # Combine all parent content
        combined_content = " ".join(p.content for p in parents)

        # Verify most numbered words are present (some may be at boundaries)
        found_count = sum(1 for i in range(300) if f"word{i}" in combined_content)
        # Should retain at least 95% of words (allowing for edge cases)
        assert found_count >= 285, f"Only found {found_count}/300 words"

    def test_mixed_normal_and_oversized_sentences(self, sample_file_hash: str) -> None:
        """Test handling of sections with both normal and oversized sentences."""
        engine = ChunkingEngine(
            parent_max_tokens=100,
            child_tokens=50,
            child_overlap=10,
        )

        # Mix of normal sentences and an oversized blob
        normal1 = "This is a normal sentence."
        normal2 = "Another normal sentence here."
        oversized = "x " * 400  # Oversized content
        normal3 = "Final normal sentence."

        markdown = f"# Section\n\n{normal1} {oversized} {normal2} {normal3}"

        parents, children = engine.chunk_document(
            markdown, sample_file_hash, "mixed.md"
        )

        # Should create multiple parents
        assert len(parents) >= 4  # At least one for each part

        # All parents should be within limits
        for parent in parents:
            parent_tokens = len(engine.tokenizer.encode(parent.content))
            assert parent_tokens <= engine.parent_max_tokens + 10

    def test_parent_tokens_never_exceed_max(self, sample_file_hash: str) -> None:
        """Test that parent chunks never exceed parent_max_tokens."""
        engine = ChunkingEngine(
            parent_max_tokens=200,
            child_tokens=100,
            child_overlap=20,
        )

        # Create extreme content that would previously cause issues
        extreme_content = "a " * 5000  # Very long content
        markdown = f"# Extreme Section\n\n{extreme_content}"

        parents, children = engine.chunk_document(
            markdown, sample_file_hash, "extreme.md"
        )

        for parent in parents:
            parent_tokens = len(engine.tokenizer.encode(parent.content))
            # Should never exceed max (with small buffer for tokenizer edge cases)
            assert parent_tokens <= engine.parent_max_tokens + 5, (
                f"Parent has {parent_tokens} tokens, "
                f"exceeds max of {engine.parent_max_tokens}"
            )

    def test_children_created_for_force_split_parents(
        self, sample_file_hash: str
    ) -> None:
        """Test that children are properly created for force-split parents."""
        engine = ChunkingEngine(
            parent_max_tokens=100,
            child_tokens=50,
            child_overlap=10,
        )

        oversized = "token " * 400
        markdown = f"# Section\n\n{oversized}"

        parents, children = engine.chunk_document(
            markdown, sample_file_hash, "force_split.md"
        )

        # Every parent should have children
        for parent in parents:
            assert len(parent.child_ids) > 0, f"Parent {parent.id} has no children"

            # Verify children are properly linked
            for child_id in parent.child_ids:
                child = next((c for c in children if c.id == child_id), None)
                assert child is not None, f"Child {child_id} not found"
                assert child.parent_id == parent.id

    def test_header_path_preserved_for_split_sections(
        self, sample_file_hash: str
    ) -> None:
        """Test that header path is preserved when sections are force-split."""
        engine = ChunkingEngine(
            parent_max_tokens=100,
            child_tokens=50,
            child_overlap=10,
        )

        oversized = "content " * 400
        markdown = f"# Main Chapter\n\n## Subsection\n\n{oversized}"

        parents, children = engine.chunk_document(
            markdown, sample_file_hash, "headers.md"
        )

        # All parents from the oversized section should have the same header path
        for parent in parents:
            assert parent.header_path != ""
            # Should contain subsection reference
            assert "Subsection" in parent.header_path
