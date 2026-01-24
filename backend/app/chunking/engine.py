"""Parent-child hierarchical chunking engine."""
import hashlib
import re
import uuid
from dataclasses import dataclass

import nltk
from transformers import AutoTokenizer


@dataclass
class ParentChunk:
    """Parent chunk containing full section context."""

    id: str
    content: str
    file_hash: str
    file_name: str
    header_path: str
    child_ids: list[str]


@dataclass
class ChildChunk:
    """Child chunk for retrieval with overlap."""

    id: str
    content: str
    parent_id: str
    file_hash: str
    file_name: str
    chunk_index: int
    header_path: str


class ChunkingEngine:
    """
    Parent-Child Hierarchical Chunking with Markdown Awareness.

    Strategy:
    1. Split by headers (H1, H2, H3) to create semantic boundaries
    2. Create parent chunks from header sections (~1200 tokens max)
    3. Split parents into overlapping children (384 tokens, 64 overlap)
    4. Children are retrieval targets, parents are context sources
    """

    def __init__(
        self,
        parent_max_tokens: int = 1200,
        parent_min_tokens: int = 200,
        child_tokens: int = 384,
        child_overlap: int = 64,
        tokenizer_name: str = "BAAI/bge-m3",
    ) -> None:
        """
        Initialize chunking engine.

        Args:
            parent_max_tokens: Maximum tokens per parent chunk
            parent_min_tokens: Minimum tokens per parent chunk
            child_tokens: Tokens per child chunk
            child_overlap: Overlap between child chunks
            tokenizer_name: HuggingFace tokenizer to use
        """
        self.parent_max_tokens = parent_max_tokens
        self.parent_min_tokens = parent_min_tokens
        self.child_tokens = child_tokens
        self.child_overlap = child_overlap
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Markdown header pattern
        self.header_pattern = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

        # Ensure NLTK punkt is available
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

    def chunk_document(
        self, markdown_content: str, file_hash: str, file_name: str
    ) -> tuple[list[ParentChunk], list[ChildChunk]]:
        """
        Main entry point: converts markdown to parent and child chunks.

        Args:
            markdown_content: Markdown text to chunk
            file_hash: SHA256 hash of the file
            file_name: Original filename

        Returns:
            Tuple of (parent_chunks, child_chunks)
        """
        # Handle empty content
        if not markdown_content.strip():
            return [], []

        # Step 1: Split by headers into sections
        sections = self._split_by_headers(markdown_content)

        # Step 2: Create parent chunks
        parent_chunks = self._create_parents(sections, file_hash, file_name)

        # Step 3: Create child chunks with overlap
        child_chunks = self._create_children(parent_chunks)

        return parent_chunks, child_chunks

    def _split_by_headers(self, content: str) -> list[dict[str, str]]:
        """
        Split markdown by H1/H2/H3 headers, preserving hierarchy.

        Args:
            content: Markdown content

        Returns:
            List of section dicts with header_path and content
        """
        sections = []
        current_headers = ["", "", ""]  # H1, H2, H3

        # Find all headers and their positions
        headers = list(self.header_pattern.finditer(content))

        if not headers:
            # No headers, treat entire content as one section
            return [{"header_path": "Document", "content": content.strip()}]

        for i, match in enumerate(headers):
            level = len(match.group(1))  # Number of # symbols
            title = match.group(2).strip()

            # Update header hierarchy
            current_headers[level - 1] = title
            for j in range(level, 3):
                current_headers[j] = ""

            # Get content until next header (or end)
            start = match.end()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(content)
            section_content = content[start:end].strip()

            if section_content:
                header_path = " > ".join(h for h in current_headers if h)
                sections.append(
                    {
                        "header_path": header_path,
                        "content": f"## {title}\n\n{section_content}",
                    }
                )

        return sections

    def _create_parents(
        self, sections: list[dict[str, str]], file_hash: str, file_name: str
    ) -> list[ParentChunk]:
        """
        Create parent chunks, merging small sections and splitting large ones.

        Args:
            sections: List of section dicts
            file_hash: File hash
            file_name: Filename

        Returns:
            List of ParentChunk objects
        """
        parents = []
        buffer_content = ""
        buffer_header = ""

        for section in sections:
            section_tokens = len(self.tokenizer.encode(section["content"]))

            if section_tokens > self.parent_max_tokens:
                # Flush buffer first
                if buffer_content:
                    parents.append(
                        self._make_parent(
                            buffer_content, buffer_header, file_hash, file_name
                        )
                    )
                    buffer_content = ""

                # Split large section into multiple parents
                split_parents = self._split_large_section(section, file_hash, file_name)
                parents.extend(split_parents)

            elif section_tokens < self.parent_min_tokens:
                # Accumulate small sections
                if buffer_content:
                    combined = buffer_content + "\n\n" + section["content"]
                    combined_tokens = len(self.tokenizer.encode(combined))

                    if combined_tokens > self.parent_max_tokens:
                        # Flush buffer, start new
                        parents.append(
                            self._make_parent(
                                buffer_content, buffer_header, file_hash, file_name
                            )
                        )
                        buffer_content = section["content"]
                        buffer_header = section["header_path"]
                    else:
                        buffer_content = combined
                else:
                    buffer_content = section["content"]
                    buffer_header = section["header_path"]
            else:
                # Normal sized section
                if buffer_content:
                    parents.append(
                        self._make_parent(
                            buffer_content, buffer_header, file_hash, file_name
                        )
                    )
                    buffer_content = ""

                parents.append(
                    self._make_parent(
                        section["content"], section["header_path"], file_hash, file_name
                    )
                )

        # Flush remaining buffer
        if buffer_content:
            parents.append(
                self._make_parent(buffer_content, buffer_header, file_hash, file_name)
            )

        return parents

    def _split_large_section(
        self, section: dict[str, str], file_hash: str, file_name: str
    ) -> list[ParentChunk]:
        """
        Split a large section into multiple parents at sentence boundaries.

        Handles oversized sentences that NLTK cannot split (e.g., code blocks,
        tables, or poorly formatted text blobs) by force-splitting them at
        token boundaries to prevent data loss.

        Args:
            section: Section dict with header_path and content
            file_hash: File hash
            file_name: Filename

        Returns:
            List of ParentChunk objects
        """
        sentences = nltk.sent_tokenize(section["content"])
        parents = []
        current_chunk = ""

        for sentence in sentences:
            sentence_tokens = len(self.tokenizer.encode(sentence))

            if sentence_tokens > self.parent_max_tokens:
                if current_chunk:
                    parents.append(
                        self._make_parent(
                            current_chunk, section["header_path"], file_hash, file_name
                        )
                    )
                    current_chunk = ""

                tokens = self.tokenizer.encode(sentence)
                for i in range(0, len(tokens), self.parent_max_tokens):
                    chunk_tokens = tokens[i : i + self.parent_max_tokens]
                    chunk_text = self.tokenizer.decode(
                        chunk_tokens, skip_special_tokens=True
                    )
                    parents.append(
                        self._make_parent(
                            chunk_text, section["header_path"], file_hash, file_name
                        )
                    )
            else:
                test_chunk = (
                    current_chunk + " " + sentence if current_chunk else sentence
                )
                if len(self.tokenizer.encode(test_chunk)) > self.parent_max_tokens:
                    if current_chunk:
                        parents.append(
                            self._make_parent(
                                current_chunk,
                                section["header_path"],
                                file_hash,
                                file_name,
                            )
                        )
                    current_chunk = sentence
                else:
                    current_chunk = test_chunk

        if current_chunk:
            parents.append(
                self._make_parent(
                    current_chunk, section["header_path"], file_hash, file_name
                )
            )

        return parents

    def _make_parent(
        self, content: str, header_path: str, file_hash: str, file_name: str
    ) -> ParentChunk:
        """
        Create a ParentChunk with unique UUID.

        Args:
            content: Parent content
            header_path: Header path
            file_hash: File hash
            file_name: Filename

        Returns:
            ParentChunk object
        """
        # Generate deterministic UUID from content hash
        content_hash = hashlib.md5(
            f"{file_hash}:{header_path}:{content[:100]}".encode()
        ).hexdigest()
        # Use first 32 chars of hash as UUID (valid format)
        parent_uuid = str(uuid.UUID(content_hash))

        return ParentChunk(
            id=parent_uuid,
            content=content.strip(),
            file_hash=file_hash,
            file_name=file_name,
            header_path=header_path,
            child_ids=[],
        )

    def _create_children(self, parents: list[ParentChunk]) -> list[ChildChunk]:
        """
        Create overlapping child chunks from parents.

        Args:
            parents: List of parent chunks

        Returns:
            List of ChildChunk objects
        """
        children = []
        # BGE-M3 max sequence length
        MAX_TOKENS = 8192

        for parent in parents:
            tokens = self.tokenizer.encode(parent.content)

            # Defensive fallback: truncate if exceeds model max
            # This should not happen with proper _split_large_section handling
            if len(tokens) > MAX_TOKENS:
                print(
                    f"⚠️  Unexpected: Parent chunk exceeds {MAX_TOKENS} tokens "
                    f"({len(tokens)} tokens). This indicates a bug in parent creation. "
                    f"Truncating as fallback."
                )
                tokens = tokens[:MAX_TOKENS]
                parent.content = self.tokenizer.decode(tokens, skip_special_tokens=True)

            stride = self.child_tokens - self.child_overlap

            chunk_index = 0
            for start in range(0, len(tokens), stride):
                end = min(start + self.child_tokens, len(tokens))
                chunk_tokens = tokens[start:end]

                # Decode back to text
                chunk_text = self.tokenizer.decode(
                    chunk_tokens, skip_special_tokens=True
                )

                # Generate deterministic UUID for child
                child_hash = hashlib.md5(
                    f"{parent.id}:{chunk_index}:{chunk_text[:50]}".encode()
                ).hexdigest()
                child_id = str(uuid.UUID(child_hash))
                parent.child_ids.append(child_id)

                children.append(
                    ChildChunk(
                        id=child_id,
                        content=chunk_text,
                        parent_id=parent.id,
                        file_hash=parent.file_hash,
                        file_name=parent.file_name,
                        chunk_index=chunk_index,
                        header_path=parent.header_path,
                    )
                )

                chunk_index += 1

                # Stop if we've reached the end
                if end >= len(tokens):
                    break

        return children
