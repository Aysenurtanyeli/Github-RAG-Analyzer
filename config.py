"""Application configuration for GitHub RAG Analyzer."""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    github_token: str | None
    groq_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_cloud: str
    pinecone_region: str
    repo_url: str
    repo_branch: str
    embedding_model: str
    retriever_k: int
    llm_model: str
    chunk_size: int
    chunk_overlap: int

    @property
    def repo_name(self) -> str:
        """
        Return GitHub repo as 'owner/repo'.

        Supports both full URLs and direct owner/repo format.
        """
        if "/" in self.repo_url and not self.repo_url.startswith("http"):
            raw = self.repo_url.strip("/")
            owner, repo = raw.split("/", 1)
            return f"{owner}/{repo.removesuffix('.git')}"

        parsed = urlparse(self.repo_url)
        path_parts = [part for part in parsed.path.strip("/").split("/") if part]
        if len(path_parts) < 2:
            raise ValueError(
                "GITHUB_REPO_URL must be 'owner/repo' or a valid GitHub URL."
            )
        return f"{path_parts[0]}/{path_parts[1].removesuffix('.git')}"


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def get_settings() -> Settings:
    github_token_raw = (os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN") or "").strip()
    return Settings(
        github_token=github_token_raw or None,
        groq_api_key=_required_env("GROQ_API_KEY"),
        pinecone_api_key=_required_env("PINECONE_API_KEY"),
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "github-rag-analyzer"),
        pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws"),
        pinecone_region=os.getenv("PINECONE_REGION", "us-east-1"),
        repo_url=_required_env("GITHUB_REPO_URL"),
        repo_branch=os.getenv("GITHUB_REPO_BRANCH", "main"),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        retriever_k=int(os.getenv("RETRIEVER_TOP_K", "4")),
        llm_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1200")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "180")),
    )
