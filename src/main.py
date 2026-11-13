
"""Vaani FastAPI application."""

from fastapi import FastAPI

from src.api.documents import router as documents_router

app = FastAPI(
    title="Vaani",
    description="Research Paper RAG Pipeline",
    version="0.1.0",
)

app.include_router(documents_router)


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "vaani"}
