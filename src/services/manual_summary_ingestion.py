"""Store manually generated section summaries in Qdrant."""

from __future__ import annotations

import uuid
from functools import lru_cache
from pathlib import Path

import numpy as np
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from src.core.config import PROJECT_ROOT


VECTOR_SIZE = 768
PREFIX = "search_document: "
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_BATCH_SIZE = 32
UPSERT_BATCH_SIZE = 64
MAX_CONTEXT_TOKENS = 8192


class ManualSummarySettings(BaseSettings):
    """Configuration for manual summary ingestion."""

    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "research_papers"
    EMBEDDING_MODEL: str = "nomic-ai/nomic-embed-text-v1.5"
    EMBEDDING_DEVICE: str = "cpu"

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class ManualSummaryIngestion:
    """Embed and upsert manually written section summaries."""

    def __init__(self) -> None:
        self.settings = ManualSummarySettings()

        self.qdrant = QdrantClient(
            url=self.settings.QDRANT_URL
        )

        self.collection = self.settings.QDRANT_COLLECTION

        if not self.qdrant.collection_exists(self.collection):
            raise RuntimeError(
                f"Qdrant collection '{self.collection}' does not exist"
            )

        self.model = None

    def _load_model(self) -> None:
        """Load Nomic only when the first summary is submitted."""

        if self.model is None:
            self.model = SentenceTransformer(
                self.settings.EMBEDDING_MODEL,
                device=self.settings.EMBEDDING_DEVICE,
            )

            if self.model.get_embedding_dimension() != VECTOR_SIZE:
                raise ValueError(
                    f"Expected {VECTOR_SIZE}-dimensional embeddings"
                )

    @staticmethod
    def point_id(document_id: str, section_id: str) -> str:
        """Generate the same deterministic ID as the existing pipeline."""

        identity = (
            f"{document_id}:{section_id}:summary:0"
        )

        return str(
            uuid.uuid5(uuid.NAMESPACE_URL, identity)
        )

    def store_summary(
        self,
        document_id: str,
        section_id: str,
        header: str,
        page: int | None,
        summary: str,
    ) -> dict:
        """Embed a manual summary and store it in Qdrant."""

        self._load_model()

        vector = self.model.encode(
            [f"{PREFIX}{summary}"],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        vector = np.asarray(vector, dtype=np.float32)[0]

        if vector.shape != (VECTOR_SIZE,):
            raise ValueError(
                f"Unexpected embedding shape: {vector.shape}"
            )

        normalized_document_id = str(
            uuid.UUID(document_id)
        )

        point = models.PointStruct(
            id=self.point_id(
                normalized_document_id,
                section_id,
            ),
            vector=vector.tolist(),
            payload={
                "node_type": "summary",
                "uuid": normalized_document_id,
                "section_id": section_id,
                "text": summary,
                "header": header,
                "page": page,
            },
        )

        self.qdrant.upsert(
            collection_name=self.collection,
            points=[point],
            wait=True,
        )

        return {
            "status": "stored",
            "document_id": normalized_document_id,
            "section_id": section_id,
            "header": header,
            "page": page,
        }
    def store_content(
        self,
        document_id: str,
        section_id: str,
        header: str,
        page: int | None,
        content: str,
        ) -> dict:
        """Chunk, embed, and store original section content."""

        self._load_model()

        if not content.strip():
            raise ValueError("Section content is empty")

        tokenizer = self.model.tokenizer

        # Match the existing ingestion pipeline:
        # include the section heading in the content.
        source_text = f"{header}\n\n{content}".strip()

        token_ids = tokenizer.encode(
            source_text,
            add_special_tokens=False,
        )

        # Reserve space for the Nomic task prefix and special tokens.
        prefix_tokens = len(
            tokenizer.encode(PREFIX, add_special_tokens=False)
        )
        special_tokens = tokenizer.num_special_tokens_to_add(pair=False)

        window_size = min(
            CHUNK_SIZE,
            MAX_CONTEXT_TOKENS - prefix_tokens - special_tokens,
        )
        step_size = window_size - CHUNK_OVERLAP

        chunks = []

        for index, start in enumerate(range(0, len(token_ids), step_size)):
            window = token_ids[start:start + window_size]

            if not window:
                continue

            chunk_text = tokenizer.decode(
                window,
                skip_special_tokens=True,
            ).strip()

            if chunk_text:
                chunks.append({
                    "chunk_index": index,
                    "chunk_size": len(window),
                    "text": chunk_text,
                })

            if start + window_size >= len(token_ids):
                break

        if not chunks:
            raise ValueError("No content chunks were generated")

        document_id = str(uuid.UUID(document_id))
        points = []

        # Generate normalized Nomic embeddings in batches.
        for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            batch = chunks[start:start + EMBEDDING_BATCH_SIZE]

            vectors = self.model.encode(
                [f"{PREFIX}{chunk['text']}" for chunk in batch],
                batch_size=EMBEDDING_BATCH_SIZE,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            vectors = np.asarray(vectors, dtype=np.float32)

            if vectors.shape != (len(batch), VECTOR_SIZE):
                raise ValueError(
                    f"Unexpected embedding shape: {vectors.shape}"
                )

            for chunk, vector in zip(batch, vectors):
                identity = (
                    f"{document_id}:{section_id}:"
                    f"content_store:{chunk['chunk_index']}"
                )

                point_id = str(
                    uuid.uuid5(uuid.NAMESPACE_URL, identity)
                )

                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=vector.tolist(),
                        payload={
                            "node_type": "content_store",
                            "uuid": document_id,
                            "section_id": section_id,
                            "chunk_index": chunk["chunk_index"],
                            "chunk_size": chunk["chunk_size"],
                            "text": chunk["text"],
                            "header": header,
                            "page": page,
                        },
                    )
                )

        # Upsert in batches.
        for start in range(0, len(points), UPSERT_BATCH_SIZE):
            self.qdrant.upsert(
                collection_name=self.collection,
                points=points[start:start + UPSERT_BATCH_SIZE],
                wait=True,
            )

        return {
            "status": "stored",
            "document_id": document_id,
            "section_id": section_id,
            "header": header,
            "page": page,
            "chunks_created": len(chunks),
            "points_upserted": len(points),
            "chunk_size": CHUNK_SIZE,
            "overlap": CHUNK_OVERLAP,
        }


@lru_cache(maxsize=1)
def get_manual_summary_service() -> ManualSummaryIngestion:
    """Reuse one service instance across API requests."""

    return ManualSummaryIngestion()
