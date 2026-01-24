"""RAG orchestrator coordinating retrieval, reranking, and generation."""
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Optional

import httpx

from ..config import settings
from ..database.qdrant_client import QdrantDB
from ..models.model_factory import ModelFactory
from ..observability.metrics import metrics, timed

# Retrieval parameters
HYBRID_SEARCH_LIMIT = 30
RERANK_TOP_K = 5


@dataclass
class RetrievedContext:
    """Retrieved context with metadata."""

    parent_id: str
    parent_content: str
    file_name: str
    header_path: str
    child_score: float


class RAGOrchestrator:
    """Main RAG pipeline orchestrator."""

    def __init__(self) -> None:
        """Initialize RAG orchestrator."""
        self.db = QdrantDB()
        self.http_client = httpx.AsyncClient(timeout=settings.rag_http_timeout)
        self.model_factory = ModelFactory()
        self.llm_provider = self.model_factory.get_provider()

    @timed("rag_query_e2e")
    async def query(
        self,
        user_query: str,
        selected_files: Optional[list[str]] = None,
        chat_history: Optional[list[dict[str, str]]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Main RAG pipeline.

        Args:
            user_query: User's question
            selected_files: Optional list of file hashes to filter
            chat_history: Optional chat history for context

        Yields:
            Response tokens from LLM
        """
        metrics.increment("rag_queries")

        # Step 1: Embed the query
        query_vectors = await self._embed_query(user_query)

        # Step 2: Hybrid search
        search_results = self.db.hybrid_search(
            query_dense=query_vectors["dense"],
            query_sparse=query_vectors["sparse"],
            file_hashes=selected_files,
            limit=HYBRID_SEARCH_LIMIT,
        )

        if not search_results:
            yield "I couldn't find any relevant information in the selected documents."
            metrics.increment("rag_queries_no_results")
            return

        # Step 3: Rerank
        documents = [r["content"] for r in search_results]
        reranked = await self._rerank(user_query, documents)

        # Step 4: Get parent contexts (deduplicated)
        parent_ids = []
        seen_parents = set()
        for idx, _ in reranked[:RERANK_TOP_K]:
            parent_id = search_results[idx]["parent_id"]
            if parent_id not in seen_parents:
                parent_ids.append(parent_id)
                seen_parents.add(parent_id)

        parents = self.db.get_parents(parent_ids)

        # Build context objects
        contexts = []
        for parent in parents:
            best_score = 0.0
            for idx, score in reranked:
                if search_results[idx]["parent_id"] == parent["id"]:
                    best_score = max(best_score, score)
                    break

            contexts.append(
                RetrievedContext(
                    parent_id=parent["id"],
                    parent_content=parent["content"],
                    file_name=parent["file_name"],
                    header_path=parent["header_path"],
                    child_score=best_score,
                )
            )

        # Step 5: Generate response
        async for token in self._generate(user_query, contexts, chat_history):
            yield token

    @timed("embedding")
    async def _embed_query(self, query: str) -> dict[str, Any]:
        """
        Embed query using the embedding API.

        Args:
            query: Query text

        Returns:
            Dict with 'dense' and 'sparse' vectors
        """
        response = await self.http_client.post(
            f"{settings.embed_api_url}/embed",
            json={"text": query, "is_query": True},
        )
        response.raise_for_status()
        data = response.json()

        return {"dense": data["dense_vecs"][0], "sparse": data["sparse_vecs"][0]}

    @timed("reranking")
    async def _rerank(
        self, query: str, documents: list[str]
    ) -> list[tuple[int, float]]:
        """
        Rerank documents using the reranking API.

        Args:
            query: Query text
            documents: List of documents to rerank

        Returns:
            List of (index, score) tuples sorted by score descending
        """
        response = await self.http_client.post(
            f"{settings.rerank_api_url}/rerank",
            json={
                "query": query,
                "documents": documents,
                "top_k": RERANK_TOP_K * 2,  # Get more for parent dedup
            },
        )
        response.raise_for_status()
        data = response.json()

        return [(r["index"], r["score"]) for r in data["results"]]

    @timed("generation")
    async def _generate(
        self,
        query: str,
        contexts: list[RetrievedContext],
        chat_history: Optional[list[dict[str, str]]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Generate response using LLM provider.

        Args:
            query: User query
            contexts: Retrieved contexts
            chat_history: Optional chat history

        Yields:
            Response tokens from LLM
        """
        # Build context string
        context_parts = []
        for i, ctx in enumerate(contexts, 1):
            context_parts.append(
                f"[Source {i}: {ctx.file_name} > {ctx.header_path}]\n"
                f"{ctx.parent_content}\n"
            )
        context_str = "\n---\n".join(context_parts)

        # System prompt
        system_prompt = (
            "You are a helpful assistant that answers questions based on the provided "
            "context.\n\n"
            "RULES:\n"
            "1. Answer ONLY using information from the provided context.\n"
            "2. If the context doesn't contain the answer, say "
            '"I don\'t have information about that in the provided documents."\n'
            "3. Cite your sources using [Source N] notation.\n"
            "4. Be concise and direct.\n"
            "5. Use markdown formatting for readability."
        )

        # Build messages
        messages = [{"role": "system", "content": system_prompt}]

        if chat_history:
            messages.extend(chat_history[-6:])  # Last 3 turns

        user_message = f"""CONTEXT:
{context_str}

QUESTION: {query}

Provide a helpful answer based on the context above."""

        messages.append({"role": "user", "content": user_message})

        # Stream from LLM provider
        async for token in self.llm_provider.generate_streaming(
            messages=messages, temperature=0.3, max_tokens=1024
        ):
            yield token

        # Append source references
        yield "\n\n---\n**Sources:**\n"
        for i, ctx in enumerate(contexts, 1):
            yield f"- [{i}] {ctx.file_name} > {ctx.header_path}\n"
