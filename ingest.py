"""Ingest GitHub repository files into Pinecone."""

from __future__ import annotations

import hashlib
import os
import re
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

DEFAULT_IGNORE_SUBSTRINGS = [
    "/components/",
    "/ui/",
    "/styles/",
    "/assets/",
    "/public/",
    "/layouts/",
]

# If focus list is non-empty, ONLY these paths/files are ingested.
DEFAULT_FOCUS_SUBSTRINGS = [
    "/services/",
    "/store/",
    "/hooks/",
    "/routes/",
    "/context/",
    "/utils/",
    "/src/services/",
    "/src/store/",
    "/src/hooks/",
    "/src/routes/",
    "/src/context/",
    "/src/utils/",
    "/app/services/",
    "/app/store/",
    "/app/hooks/",
    "/app/routes/",
    "/app/context/",
    "/app/utils/",
    "/main.ts",
    "/main.tsx",
    "/main.js",
    "/index.js",
    "/index.ts",
    "/app.ts",
    "/app.tsx",
]


class IngestCancelled(Exception):
    """Raised when user cancels ingestion."""


def _file_filter(path: str) -> bool:
    lowered = path.lower()
    if "/.git/" in lowered or "/.github/" in lowered:
        return False
    if "/node_modules/" in lowered or "/venv/" in lowered or "/.venv/" in lowered:
        return False

    suffix = Path(lowered).suffix
    if suffix not in ALLOWED_EXTENSIONS:
        return False

    # Allow overriding ignore/focus lists via env (comma-separated substrings).
    # Examples:
    #   INGEST_IGNORE_PATHS="/components/,/assets/"
    #   INGEST_FOCUS_PATHS="/services/,/utils/,/main.tsx"
    ignore_raw = os.getenv("INGEST_IGNORE_PATHS", "")
    focus_raw = os.getenv("INGEST_FOCUS_PATHS", "")
    ignore_list = [s.strip().lower() for s in ignore_raw.split(",") if s.strip()] or DEFAULT_IGNORE_SUBSTRINGS
    focus_list = [s.strip().lower() for s in focus_raw.split(",") if s.strip()] or []

    # Ignore always wins.
    if any(sub in lowered for sub in ignore_list):
        return False

    # If focus list provided (env) use it; else default focus list can be enabled via flag.
    enable_default_focus = (os.getenv("INGEST_ENABLE_DEFAULT_FOCUS", "true").strip().lower() in {"1", "true", "yes", "y"})
    effective_focus = focus_list if focus_list else (DEFAULT_FOCUS_SUBSTRINGS if enable_default_focus else [])
    if effective_focus and not any(sub in lowered for sub in effective_focus):
        return False

    return True


_SIG_KEEP_RE = re.compile(
    r"^\s*(export\s+)?(default\s+)?(async\s+)?"
    r"(function|class|interface|type|enum|const|let|var)\b|^\s*import\b|^\s*from\b",
    re.IGNORECASE,
)


def _signature_only_text(text: str) -> str:
    """
    Heuristic "signature-only" reducer for JS/TS (including TSX/JSX).
    Keeps imports/exports and top-level declarations, plus nearby comments.
    Avoids pulling large JSX/HTML bodies into embeddings.
    """
    lines = text.splitlines()
    out: list[str] = []
    comment_buffer: list[str] = []

    for line in lines:
        stripped = line.strip()

        # Keep short comment blocks only if followed by a signature.
        if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*") or stripped.startswith("*/"):
            comment_buffer.append(line)
            if len(comment_buffer) > 8:
                comment_buffer = comment_buffer[-8:]
            continue

        if _SIG_KEEP_RE.search(line):
            if comment_buffer:
                out.extend(comment_buffer)
                comment_buffer = []
            out.append(line)
            continue

        # Drop likely JSX/HTML-heavy lines
        if "<" in line and ">" in line:
            comment_buffer = []
            continue

        # Reset comment buffer when encountering unrelated code to avoid attaching stale comments.
        if stripped:
            comment_buffer = []

    return "\n".join(out).strip()


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

        # Optional: shrink JS/TS/TSX/JSX to signatures only before chunking/embedding.
        sig_only = os.getenv("INGEST_SIGNATURE_ONLY", "true").strip().lower() in {"1", "true", "yes", "y"}
        if sig_only and extension in {".js", ".ts", ".tsx", ".jsx"} and isinstance(doc.page_content, str):
            reduced = _signature_only_text(doc.page_content)
            if reduced:
                doc.page_content = reduced

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

    # NOTE: file_filter is applied before downloading file content.
    # If focus filters are too strict (common for small repos), loader may return 0 docs.
    # In that case we retry once with focus disabled (ignore list still applies).

    def _filter_relaxed(path: str) -> bool:
        # Temporarily disable focus list by overriding env knob.
        prev = os.getenv("INGEST_ENABLE_DEFAULT_FOCUS")
        try:
            os.environ["INGEST_ENABLE_DEFAULT_FOCUS"] = "false"
            os.environ.pop("INGEST_FOCUS_PATHS", None)
            return _file_filter(path)
        finally:
            # restore previous env to avoid side-effects outside this call
            if prev is None:
                os.environ.pop("INGEST_ENABLE_DEFAULT_FOCUS", None)
            else:
                os.environ["INGEST_ENABLE_DEFAULT_FOCUS"] = prev

    loader = GithubFileLoader(
        repo=repo_name,
        branch=repo_branch,
        access_token=settings.github_token or "",
        file_filter=_file_filter,
    )
    docs = loader.load()
    if not docs:
        # Retry with relaxed focus if focus was enabled.
        enable_default_focus = (
            os.getenv("INGEST_ENABLE_DEFAULT_FOCUS", "true").strip().lower()
            in {"1", "true", "yes", "y"}
        )
        has_focus_env = bool(os.getenv("INGEST_FOCUS_PATHS", "").strip())
        if enable_default_focus or has_focus_env:
            loader2 = GithubFileLoader(
                repo=repo_name,
                branch=repo_branch,
                access_token=settings.github_token or "",
                file_filter=_filter_relaxed,
            )
            docs = loader2.load()
    if not docs:
        raise RuntimeError(
            "GithubFileLoader herhangi bir belge dondurmedi. "
            "Repo bos olabilir, branch yanlis olabilir veya filtreler (focus/ignore) tum dosyalari elemis olabilir. "
            "Cozum: INGEST_ENABLE_DEFAULT_FOCUS=false yapip tekrar dene."
        )
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
