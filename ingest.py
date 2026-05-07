"""Ingest GitHub repository files into Pinecone."""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from urllib.parse import urlparse

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_community.document_loaders.github import GithubFileLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import NotFoundException
from rich.console import Console
from rich.panel import Panel

from config import get_settings

console = Console()

LANGUAGE_SPLITTER_MAP = {
    ".py": Language.PYTHON,
    ".java": Language.JAVA,
    ".cs": Language.CSHARP,
}

ALLOWED_EXTENSIONS = {
    ".py",
    ".java",
    ".cs",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".md",
    ".mdx",
    ".txt",
    ".rst",
    ".yml",
    ".yaml",
    ".json",
    ".toml",
    ".ini",
}


class IngestCancelled(Exception):
    """Raised when user cancels ingestion."""


def _file_filter(path: str) -> bool:
    lowered = path.lower()
    if "/.git/" in lowered or "/.github/" in lowered:
        return False
    if "/node_modules/" in lowered or "/venv/" in lowered or "/.venv/" in lowered:
        return False
    return Path(lowered).suffix in ALLOWED_EXTENSIONS


def _build_splitter(
    extension: str, chunk_size: int, chunk_overlap: int
) -> RecursiveCharacterTextSplitter:
    language = LANGUAGE_SPLITTER_MAP.get(extension)
    if language:
        return RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def _split_documents(
    docs: list[Document],
    chunk_size: int,
    chunk_overlap: int,
    cancel_event: threading.Event | None = None,
) -> list[Document]:
    chunks: list[Document] = []
    for doc in docs:
        if cancel_event and cancel_event.is_set():
            raise IngestCancelled()
        source = str(doc.metadata.get("path") or doc.metadata.get("source") or "")
        extension = Path(source).suffix.lower()
        splitter = _build_splitter(
            extension=extension,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks.extend(splitter.split_documents([doc]))
    return chunks


def _stable_id(namespace: str, source: str, content: str) -> str:
    h = hashlib.sha1()  # noqa: S324 - stable IDs, not for security
    h.update(namespace.encode("utf-8"))
    h.update(b"\0")
    h.update(source.encode("utf-8", errors="ignore"))
    h.update(b"\0")
    h.update(content.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def _repo_name_from_url(repo_url: str) -> str:
    if "/" in repo_url and not repo_url.startswith("http"):
        raw = repo_url.strip("/")
        owner, repo = raw.split("/", 1)
        return f"{owner}/{repo.removesuffix('.git')}"
    parsed = urlparse(repo_url)
    path_parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(path_parts) < 2:
        raise ValueError("repo_url must be 'owner/repo' or a valid GitHub URL.")
    return f"{path_parts[0]}/{path_parts[1].removesuffix('.git')}"


def _ensure_pinecone_index(
    *,
    api_key: str,
    index_name: str,
    dimension: int,
    cloud: str,
    region: str,
):
    pc = Pinecone(api_key=api_key)
    existing = {idx["name"] for idx in pc.list_indexes()}
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
    return pc.Index(index_name)


def ingest_repo(
    *,
    repo_url: str,
    repo_branch: str,
    namespace: str,
    force: bool = False,
    cancel_event: threading.Event | None = None,
) -> dict:
    if cancel_event and cancel_event.is_set():
        raise IngestCancelled()

    settings = get_settings()
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)

    # Ensure Pinecone index exists with correct dimension
    dim_probe = embeddings.embed_query("dimension probe")
    index = _ensure_pinecone_index(
        api_key=settings.pinecone_api_key,
        index_name=settings.pinecone_index_name,
        dimension=len(dim_probe),
        cloud=settings.pinecone_cloud,
        region=settings.pinecone_region,
    )
    if force:
        try:
            index.delete(delete_all=True, namespace=namespace)
        except NotFoundException:
            # Namespace yoksa sorun değil: temiz başlangıç için devam.
            pass

    repo_name = _repo_name_from_url(repo_url)
    console.print(
        Panel.fit(
            f"Repo yükleniyor: [bold]{repo_name}[/bold] @ {repo_branch}",
            title="GitHub Loader",
            border_style="cyan",
        )
    )

    loader = GithubFileLoader(
        repo=repo_name,
        branch=repo_branch,
        access_token=settings.github_token,
        file_filter=_file_filter,
    )
    docs = loader.load()
    if not docs:
        raise RuntimeError("GithubFileLoader herhangi bir belge dondurmedi.")
    if cancel_event and cancel_event.is_set():
        raise IngestCancelled()

    console.print(f"[green]{len(docs)} belge yüklendi.[/green]")
    chunks = _split_documents(
        docs=docs,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        cancel_event=cancel_event,
    )
    console.print(f"[green]{len(chunks)} chunk üretildi.[/green]")
    if cancel_event and cancel_event.is_set():
        raise IngestCancelled()

    ids: list[str] = []
    for doc in chunks:
        if cancel_event and cancel_event.is_set():
            raise IngestCancelled()
        source = str(doc.metadata.get("path") or doc.metadata.get("source") or "")
        ids.append(_stable_id(namespace, source, doc.page_content))

    vectorstore = PineconeVectorStore(
        index=index, embedding=embeddings, namespace=namespace
    )

    # Büyük ingest’i tek seferde yapmak yerine batch’e böl: iptal daha hızlı duyulur.
    batch_size = 40
    for start in range(0, len(chunks), batch_size):
        if cancel_event and cancel_event.is_set():
            raise IngestCancelled()
        end = min(start + batch_size, len(chunks))
        vectorstore.add_documents(
            chunks[start:end],
            ids=ids[start:end],
        )

    console.print(
        Panel.fit(
            (
                "Ingest tamamlandı.\n"
                f"Index: {settings.pinecone_index_name}\n"
                f"Namespace: {namespace}\n"
                f"Chunk: {len(chunks)}"
            ),
            title="Başarılı",
            border_style="green",
        )
    )
    return {
        "repo_name": repo_name,
        "branch": repo_branch,
        "namespace": namespace,
        "chunks": len(chunks),
        "index": settings.pinecone_index_name,
    }


def main() -> None:
    s = get_settings()
    namespace = f"{s.repo_name.replace('/', '__')}__{s.repo_branch}"
    ingest_repo(
        repo_url=s.repo_url,
        repo_branch=s.repo_branch,
        namespace=namespace,
        force=False,
    )


if __name__ == "__main__":
    main()
