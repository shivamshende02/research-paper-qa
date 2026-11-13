"""Dual-path Nomic embedding and Qdrant ingestion."""

from __future__ import annotations
import time

from groq import RateLimitError

import argparse
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from groq import Groq
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor

from src.models import document
from mistralai.client import Mistral

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREFIX = "search_document: "
VECTOR_SIZE = 768
MAX_CONTEXT_TOKENS = 8192
EMBEDDING_BATCH_SIZE = 32
UPSERT_BATCH_SIZE = 64


class IngestionSettings(BaseSettings):
    """Configuration loaded from the project's .env file."""

    GROQ_API_KEY: SecretStr
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "research_papers"
    EMBEDDING_MODEL: str = "nomic-ai/nomic-embed-text-v1.5"
    MISTRAL_API_KEY: SecretStr
    MISTRAL_MODEL: str = "mistral-small-latest"

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@dataclass
class IngestionSection:
    section_id: str
    header: str
    content: str
    page: int | None


@dataclass
class ContentChunk:
    section_id: str
    chunk_index: int
    chunk_size: int
    text: str


class VectorIngestion:
    """Create section summaries and content chunks in one Qdrant collection."""

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> None:
        if chunk_size not in {250, 500, 1000}:
            raise ValueError("chunk_size must be 250, 500, or 1000")

        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.settings = IngestionSettings()

        # Explicitly require CUDA; do not silently fall back to CPU.
        self.model = SentenceTransformer(
        self.settings.EMBEDDING_MODEL,
        device="cuda",
        )

        self.tokenizer = self.model.tokenizer

        self.groq = Groq(
            api_key=self.settings.GROQ_API_KEY.get_secret_value()
        )
        self.mistral = Mistral(
            api_key=self.settings.MISTRAL_API_KEY.get_secret_value()
        )

        self.qdrant = QdrantClient(url=self.settings.QDRANT_URL)
        self.collection = self.settings.QDRANT_COLLECTION

        # Verify the configured model's output dimension.
        if self.model.get_embedding_dimension() != VECTOR_SIZE:
            raise ValueError(
                f"Expected {VECTOR_SIZE}-dimensional embeddings"
            )

        if not self.qdrant.collection_exists(self.collection):
            self.qdrant.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                ),
            )

    def _token_ids(self, text: str) -> list[int]:
        return self.tokenizer.encode(
            text,
            add_special_tokens=False,
        )

    def _decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
        ).strip()

    def _content_token_budget(self) -> int:
        """Reserve context space for prefix and special tokens."""
        prefix_tokens = len(self._token_ids(PREFIX))
        special_tokens = self.tokenizer.num_special_tokens_to_add(pair=False)

        budget = MAX_CONTEXT_TOKENS - prefix_tokens - special_tokens

        if self.chunk_size > budget:
            raise ValueError("Chunk size exceeds the Nomic context limit")

        return self.chunk_size

    def chunk_text(
        self,
        text: str,
        section_id: str,
    ) -> list[ContentChunk]:
        """Split section text into tokenizer-based sliding windows."""

        token_ids = self._token_ids(text)
        window_size = self._content_token_budget()
        step_size = window_size - self.overlap

        chunks: list[ContentChunk] = []

        for index, start in enumerate(range(0, len(token_ids), step_size)):
            window = token_ids[start : start + window_size]

            if not window:
                continue

            chunk = self._decode(window)

            if chunk:
                chunks.append(
                    ContentChunk(
                        section_id=section_id,
                        chunk_index=index,
                        chunk_size=len(window),
                        text=chunk,
                    )
                )

            if start + window_size >= len(token_ids):
                break

        return chunks

    def extract_sections(
        self,
        document: dict[str, Any],
    ) -> list[IngestionSection]:
        """Flatten parser sections while preserving hierarchical IDs."""

        extracted: list[IngestionSection] = []

        def visit(items: list[dict[str, Any]], parent_id: str = "") -> None:
            for index, item in enumerate(items):
                section_id = (
                    f"{parent_id}.{index}" if parent_id else str(index)
                )

                header = str(item.get("header") or "Untitled section")
                content = str(item.get("content") or "")

                metadata = item.get("metadata") or {}
                page = metadata.get("page")

                extracted.append(
                    IngestionSection(
                        section_id=section_id,
                        header=header,
                        content=content,
                        page=page,
                    )
                )

                children = item.get("subsections") or item.get("children") or []

                if children:
                    visit(children, section_id)

        sections = document.get("sections")

        if not isinstance(sections, list):
            raise ValueError("Input JSON must contain a sections list")

        visit(sections)

        if not extracted:
            raise ValueError("No sections found in the input JSON")

        return extracted

    
    def generate_summary(self, section: IngestionSection) -> str:
        """Generate a concise summary with Groq rate-limit retries."""

        source_text = (
            f"Section heading: {section.header}\n\n"
            f"{section.content}"
        )

        if not source_text.strip():
            return ""

        max_retries = 6
        base_delay = 5.0

        for attempt in range(max_retries + 1):
            try:
                response = self.groq.chat.completions.create(
                    model=self.settings.GROQ_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Summarize the supplied research-paper section. "
                                "Produce 150–300 tokens when the source contains "
                                "enough information. Preserve technical terms, "
                                "claims, methods, equations, and findings. "
                                "Do not invent information. "
                                "For short sections, summarize only the "
                                "information actually present."
                            ),
                        },
                        {
                            "role": "user",
                            "content": source_text,
                        },
                    ],
                    temperature=0.1,
                    max_tokens=350,
                )

                summary = (
                    response.choices[0].message.content or ""
                ).strip()

                if not summary:
                    raise ValueError(
                        f"Groq returned an empty summary for "
                        f"section {section.section_id}"
                    )

                return summary

            except RateLimitError as exc:
                if attempt >= max_retries:
                    raise

                # Prefer the retry timing provided by Groq.
                delay = base_delay * (2 ** attempt)

                try:
                    headers = exc.response.headers

                    retry_after = headers.get("retry-after")
                    reset_tokens = headers.get(
                        "x-ratelimit-reset-tokens"
                    )

                    if retry_after:
                        delay = float(retry_after.rstrip("s"))
                    elif reset_tokens:
                        delay = float(reset_tokens.rstrip("s"))

                except (AttributeError, TypeError, ValueError):
                    pass

                delay = min(max(delay, 1.0), 120.0)

                print(
                    f"[Groq rate limit] Section {section.section_id}: "
                    f"retry {attempt + 1}/{max_retries} "
                    f"in {delay:.1f} seconds"
                )

                time.sleep(delay)

        raise RuntimeError(
            f"Summary generation failed for section {section.section_id}"
        )


    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed text using the required Nomic task prefix."""

        if not texts:
            return np.empty((0, VECTOR_SIZE), dtype=np.float32)

        prefixed_texts = [f"{PREFIX}{text}" for text in texts]

        vectors = self.model.encode(
            prefixed_texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        vectors = np.asarray(vectors, dtype=np.float32)

        if vectors.shape[1] != VECTOR_SIZE:
            raise ValueError(f"Unexpected embedding shape: {vectors.shape}")

        return vectors

    @staticmethod
    def point_id(
        document_id: str,
        section_id: str,
        node_type: str,
        item_index: int,
    ) -> str:
        """Create stable IDs so repeated ingestion is idempotent."""

        identity = (
            f"{document_id}:{section_id}:{node_type}:{item_index}"
        )

        return str(
            uuid.uuid5(uuid.NAMESPACE_URL, identity)
        )

    def ingest(self, document: dict[str, Any]) -> dict[str, int]:
        """Run summary generation and content embedding concurrently."""

        document_id = str(uuid.UUID(str(document["uuid"])))
        sections = self.extract_sections(document)

        summary_count = 0
        chunk_count = 0

        # Prepare all section-level content chunks.
        section_chunks = []

        for section in sections:
            section_text = (
                f"{section.header}\n\n{section.content}"
            ).strip()

            if not section_text:
                continue

            chunks = self.chunk_text(
                section_text,
                section.section_id,
            )

            section_chunks.extend(
                (section, chunk) for chunk in chunks
            )

            # Start Groq summary generation concurrently.
        summary_futures = {}

        with ThreadPoolExecutor(max_workers=1) as executor:

            for section in sections:
                if not section.content.strip():
                    continue

                future = executor.submit(
                    self.generate_summary,
                    section,
                )

                summary_futures[future] = section

                # GPU content embedding runs while Groq requests execute.
            content_points = []

            for start in range(0, len(section_chunks), EMBEDDING_BATCH_SIZE):

                batch = section_chunks[
                    start:start + EMBEDDING_BATCH_SIZE
                ]

                texts = [
                    chunk.text for _, chunk in batch
                ]

                vectors = self.embed(texts)

                for (section, chunk), vector in zip(batch, vectors):

                    content_points.append(
                        models.PointStruct(
                        id=self.point_id(
                        document_id,
                        section.section_id,
                        "content_store",
                        chunk.chunk_index,
                    ),
                        vector=vector.tolist(),
                        payload={
                        "node_type": "content_store",
                        "uuid": document_id,
                        "section_id": section.section_id,
                        "chunk_index": chunk.chunk_index,
                        "chunk_size": chunk.chunk_size,
                        "text": chunk.text,
                        "header": section.header,
                        "page": section.page,
                    },
                    )
                    )

                    chunk_count += 1

                    # Collect completed Groq summaries.
            summary_records = []

            for future, section in summary_futures.items():

                summary = future.result()

                if not summary:
                    continue

                summary_records.append(
                    (section, summary)
                )

                # Embed summaries on GPU.
            summary_vectors = self.embed(
                [summary for _, summary in summary_records]
            )

            summary_points = []

            for (section, summary), vector in zip(
                summary_records,
                summary_vectors,
            ):

                summary_points.append(
                    models.PointStruct(
                    id=self.point_id(
                    document_id,
                    section.section_id,
                    "summary",
                    0,
                ),
                    vector=vector.tolist(),
                    payload={
                    "node_type": "summary",
                    "uuid": document_id,
                    "section_id": section.section_id,
                    "text": summary,
                    "header": section.header,
                    "page": section.page,
                },
                )
                )

                summary_count += 1

                # Combine both paths.
        all_points = summary_points + content_points

        # Upsert into one Qdrant collection.
        for start in range(0, len(all_points), UPSERT_BATCH_SIZE):

            self.qdrant.upsert(
                collection_name=self.collection,
                points=all_points[
                start:start + UPSERT_BATCH_SIZE
            ],
                wait=True,
            )

        return {
            "sections": len(sections),
            "summaries": summary_count,
            "content_chunks": chunk_count,
            "points_upserted": len(all_points),
        }



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest extracted research-paper JSON into Qdrant."
    )

    parser.add_argument("--input", required=True, help="Structured JSON path")
    parser.add_argument(
        "--chunk-size",
        type=int,
        choices=[250, 500, 1000],
        default=500,
    )
    parser.add_argument("--overlap", type=int, default=50)

    args = parser.parse_args()

    input_path = Path(args.input)

    with input_path.open("r", encoding="utf-8") as file:
        document = json.load(file)

    ingestion = VectorIngestion(
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    result = ingestion.ingest(document)

    print("\nINGESTION COMPLETED")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
