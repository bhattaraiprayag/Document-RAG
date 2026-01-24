"""Integration tests for document conversion."""
from pathlib import Path

import pytest
from markitdown import MarkItDown


@pytest.mark.integration
class TestDocumentConversion:
    """Test MarkItDown conversion for various file types."""

    def test_markitdown_available(self) -> None:
        """Test that MarkItDown is available and can be instantiated."""
        converter = MarkItDown()
        assert converter is not None

    def test_convert_text_file(self, tmp_path: Path) -> None:
        """Test converting a simple text file to markdown."""
        # Create a test text file
        test_file = tmp_path / "test.txt"
        test_content = "# Test Document\n\nThis is a test document."
        test_file.write_text(test_content)

        converter = MarkItDown()
        result = converter.convert(str(test_file))

        assert result.text_content is not None
        assert len(result.text_content) > 0
        assert "Test Document" in result.text_content

    def test_convert_markdown_file(self, tmp_path: Path) -> None:
        """Test converting a markdown file (passthrough)."""
        test_file = tmp_path / "test.md"
        test_content = """# Chapter 1

## Section 1.1
Content here.

# Chapter 2
More content.
"""
        test_file.write_text(test_content)

        converter = MarkItDown()
        result = converter.convert(str(test_file))

        assert result.text_content is not None
        assert "Chapter 1" in result.text_content
        assert "Chapter 2" in result.text_content
