"""Embedding provider factory for local HF or Pinecone hosted embeddings."""

from __future__ import annotations

from config import Settings


def build_embeddings(settings: Settings):
    provider = (settings.embedding_provider or "pinecone").strip().lower()
    if provider == "hf":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=settings.embedding_model)
    if provider == "pinecone":
        from langchain_pinecone import PineconeEmbeddings

        return PineconeEmbeddings(
            model=settings.embedding_model,
            pinecone_api_key=settings.pinecone_api_key,
        )
    raise ValueError(
        f"Unsupported EMBEDDING_PROVIDER='{settings.embedding_provider}'. "
        "Use 'pinecone' or 'hf'."
    )

