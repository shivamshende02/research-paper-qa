
"""Two-tier research-paper retrieval service."""

import numpy as np
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from groq import Groq
import torch

from src.services.vector_ingestion import (
    IngestionSettings,
    VECTOR_SIZE,
)


class ResearchPaperRetriever:
    """Retrieve research-paper evidence using summary-first search."""

    SUMMARY_SCORE_THRESHOLD = 0.50

    def __init__(self) -> None:
        self.settings = IngestionSettings()

        # Automatically select an available device.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[Retrieval] Using device: {self.device}")

        # Load the same embedding model used during ingestion.
        self.embedding_model = SentenceTransformer(
            self.settings.EMBEDDING_MODEL,
            device=self.device,
        )

        self.reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device=self.device,
        )

        self.groq = Groq(
            api_key=self.settings.GROQ_API_KEY.get_secret_value()
        )

        self.qdrant = QdrantClient(url=self.settings.QDRANT_URL)

        self.collection = self.settings.QDRANT_COLLECTION

        dimension = self.embedding_model.get_embedding_dimension()

        if dimension != VECTOR_SIZE:
            raise ValueError(
                f"Expected {VECTOR_SIZE}-dimensional embeddings, "
                f"received {dimension}"
            )

    def embed_query(self, query: str) -> list[float]:
        """Embed a user query using Nomic's search_query instruction."""

        query = query.strip()

        if not query:
            raise ValueError("Query cannot be empty")

        vector = self.embedding_model.encode(
            [f"search_query: {query}"],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]

        vector = np.asarray(vector, dtype=np.float32)

        if vector.shape != (VECTOR_SIZE,):
            raise ValueError(
                f"Invalid query embedding shape: {vector.shape}"
            )

        return vector.tolist()

    
    def retrieve_summaries(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict]:
        """Retrieve the most relevant section summaries."""

        query_vector = self.embed_query(query)

        summary_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="node_type",
                    match=models.MatchValue(value="summary"),
                )
            ]
        )

        response = self.qdrant.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=summary_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        results = []

        for point in response.points:
            payload = point.payload or {}

            results.append(
                {
                    "score": float(point.score),
                    "uuid": payload.get("uuid"),
                    "section_id": payload.get("section_id"),
                    "header": payload.get("header"),
                    "page": payload.get("page"),
                    "summary": payload.get("text"),
                }
            )

        return results

    def screen_query(self, query: str) -> dict:
        """Screen a query using section summaries before content retrieval."""

        summaries = self.retrieve_summaries(query, limit=5)

        if not summaries:
            return {
                "accepted": False,
                "reason": "no_matching_summaries",
                "answer": (
                    "The provided papers do not contain information "
                    "to support this query"
                ),
                "top_score": None,
                "summaries": [],
            }

        top_score = summaries[0]["score"]

        if top_score < self.SUMMARY_SCORE_THRESHOLD:
            return {
                "accepted": False,
                "reason": "summary_score_below_threshold",
                "answer": (
                    "The provided papers do not contain information "
                    "to support this query"
                ),
                "top_score": top_score,
                "summaries": summaries,
            }

        return {
            "accepted": True,
            "reason": "relevant_summary_found",
            "answer": None,
            "top_score": top_score,
            "summaries": summaries,
        }

    def retrieve_section_content(
        self,
        query: str,
        matched_summary: dict,
        limit: int = 20,
    ) -> list[dict]:
        """Retrieve content chunks from one matched document section."""

        document_uuid = matched_summary.get("uuid")
        section_id = matched_summary.get("section_id")

        if not document_uuid or section_id is None:
            raise ValueError(
                "Matched summary must contain uuid and section_id"
            )

        query_vector = self.embed_query(query)

        content_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="node_type",
                    match=models.MatchValue(value="content_store"),
                ),
                models.FieldCondition(
                    key="uuid",
                    match=models.MatchValue(value=document_uuid),
                ),
                models.FieldCondition(
                    key="section_id",
                    match=models.MatchValue(value=section_id),
                ),
            ]
        )

        response = self.qdrant.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=content_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        results = []

        for point in response.points:
            payload = point.payload or {}

            results.append(
                {
                    "score": float(point.score),
                    "uuid": payload.get("uuid"),
                    "section_id": payload.get("section_id"),
                    "header": payload.get("header"),
                    "page": payload.get("page"),
                    "chunk_index": payload.get("chunk_index"),
                    "text": payload.get("text"),
                }
            )

        return results


    def rerank_chunks(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """Rerank retrieved content chunks using a cross-encoder."""

        if not chunks:
            return []

        pairs = [
            (query, chunk["text"])
            for chunk in chunks
            if chunk.get("text")
        ]

        valid_chunks = [
            chunk for chunk in chunks
            if chunk.get("text")
        ]

        if not pairs:
            return []

        scores = self.reranker.predict(
            pairs,
            show_progress_bar=False,
        )

        ranked_chunks = []

        for chunk, score in zip(valid_chunks, scores):
            ranked_chunk = chunk.copy()
            ranked_chunk["rerank_score"] = float(score)
            ranked_chunks.append(ranked_chunk)

        ranked_chunks.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        return ranked_chunks[:top_k]

    def build_context(
        self,
        matched_summary: dict,
        ranked_chunks: list[dict],
    ) -> str:
        """Build grounded context with summary and chunk citations."""

        document_uuid = matched_summary["uuid"]
        section_id = matched_summary["section_id"]

        context_parts = [
            "PARENT SECTION SUMMARY",
            f"Document UUID: {document_uuid}",
            f"Section ID: {section_id}",
            f"Section: {matched_summary.get('header')}",
            f"Page: {matched_summary.get('page')}",
            matched_summary.get("summary") or "",
            "",
            "RETRIEVED EVIDENCE CHUNKS",
        ]

        for index, chunk in enumerate(ranked_chunks, start=1):
            context_parts.extend(
                [
                    "",
                    f"Evidence {index}",
                    f"Document UUID: {chunk.get('uuid')}",
                    f"Section ID: {chunk.get('section_id')}",
                    f"Section: {chunk.get('header')}",
                    f"Page: {chunk.get('page')}",
                    f"Chunk index: {chunk.get('chunk_index')}",
                    "Content:",
                    chunk.get("text") or "",
                ]
            )

        return "\n".join(context_parts)

    def generate_answer(self, query: str, context: str) -> str:
        """Generate a citation-grounded answer from retrieved evidence."""

        if not context.strip():
            raise ValueError("Context cannot be empty")

        response = self.groq.chat.completions.create(
            model=self.settings.GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a research-paper question-answering system. "
                        "Answer using only the supplied research-paper context. "
                        "Do not use outside knowledge or make unsupported claims. "
                        "Every factual claim must include a citation using this "
                        "format: [Paper UUID: <uuid>, Section: <header>, "
                        "Page: <page>]. Use only citation metadata present "
                        "in the supplied context. If the evidence is insufficient "
                        "or does not answer the question, respond exactly: "
                        "'The provided papers do not contain information "
                        "to support this query'. Do not invent paper titles, "
                        "section names, page numbers, or experimental results."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"QUESTION:\n{query}\n\n"
                        f"RESEARCH PAPER CONTEXT:\n{context}"
                    ),
                },
            ],
            temperature=0.1,
            max_tokens=1200,
        )

        answer = response.choices[0].message.content

        if not answer or not answer.strip():
            raise RuntimeError("Groq returned an empty answer")

        return answer.strip()

    def retrieve(self, query: str) -> dict:
        """Run the complete two-tier retrieval and answer pipeline."""

        # Stage 1: Screen query against section summaries.
        screening = self.screen_query(query)

        if not screening["accepted"]:
            return {
                "status": "rejected",
                "answer": screening["answer"],
                "reason": screening["reason"],
                "top_score": screening["top_score"],
            }

        # Stage 2: Use the highest-scoring summary.
        matched_summary = screening["summaries"][0]

        # Stage 3: Retrieve content from the matched section only.
        chunks = self.retrieve_section_content(
            query=query,
            matched_summary=matched_summary,
            limit=20,
        )

        if not chunks:
            return {
                "status": "rejected",
                "answer": (
                    "The provided papers do not contain information "
                    "to support this query"
                ),
                "reason": "no_content_chunks_found",
                "top_score": screening["top_score"],
            }

        # Stage 4: Rerank retrieved chunks.
        ranked_chunks = self.rerank_chunks(
            query=query,
            chunks=chunks,
            top_k=5,
        )

        if not ranked_chunks:
            return {
                "status": "rejected",
                "answer": (
                    "The provided papers do not contain information "
                    "to support this query"
                ),
                "reason": "no_valid_chunks_after_reranking",
                "top_score": screening["top_score"],
            }

        # Stage 5: Build evidence context.
        context = self.build_context(
            matched_summary=matched_summary,
            ranked_chunks=ranked_chunks,
        )

        # Stage 6: Generate grounded answer.
        answer = self.generate_answer(
            query=query,
            context=context,
        )

        return {
            "status": "success",
            "answer": answer,
            "document_uuid": matched_summary["uuid"],
            "section_id": matched_summary["section_id"],
            "section": matched_summary["header"],
            "page": matched_summary["page"],
            "summary_score": screening["top_score"],
            "retrieved_chunks": len(chunks),
            "reranked_chunks": len(ranked_chunks),
        }
