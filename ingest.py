"""Ingest GitHub repository files into Pinecone."""

from __future__ import annotations

import hashlib
import os
import re
import threading
from collections import defaultdict
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

from config import get_settings, namespace_for_repo

console = Console()

LANGUAGE_SPLITTER_MAP = {
    ".py": Language.PYTHON,
    ".java": Language.JAVA,
    ".cs": Language.CSHARP,
}

CODE_EXTENSIONS = {".py", ".java", ".cs", ".js", ".ts", ".tsx", ".jsx"}
DOC_EXTENSIONS = {".md", ".mdx", ".txt", ".rst"}
CONFIG_EXTENSIONS = {".json", ".toml", ".yml", ".yaml", ".ini"}
ALLOWED_EXTENSIONS = (
    CODE_EXTENSIONS | DOC_EXTENSIONS | CONFIG_EXTENSIONS | {".html", ".htm", ".csproj"}
)

# Orijinal ignore listesi (UI / statik içerik)
DEFAULT_IGNORE_SUBSTRINGS = [
    "/components/",
    "/ui/",
    "/styles/",
    "/assets/",
    "/public/",
    "/layouts/",
]

# Orijinal focus listesi (iş mantığı / giriş noktaları)
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

# Focus path içinde geçen giriş dosyası kalıpları (path eşleşmesi)
FOCUS_ENTRY_PATH_MARKERS = [
    "/main.ts",
    "/main.tsx",
    "/main.js",
    "/index.js",
    "/index.ts",
    "/app.ts",
    "/app.tsx",
]

DEFAULT_DENY_SUBSTRINGS = [
    *DEFAULT_IGNORE_SUBSTRINGS,
    "/.git/",
    "/.github/",
    "/node_modules/",
    "/venv/",
    "/.venv/",
    "/vendor/",
    "/vendors/",
    "/lib/",
    "/libs/",
    "/third_party/",
    "/third-party/",
    "/dist/",
    "/build/",
    "/.next/",
    "/coverage/",
    "/__pycache__/",
    "/component/",
    "/widgets/",
    "/widget/",
    "/storybook/",
    "/stories/",
    "/__tests__/",
    "/__mocks__/",
    "/static/vendor/",
    "/public/vendor/",
    "/img/",
    "/images/",
    "/fonts/",
    "/font/",
    "/media/",
    "/locales/",
    "/i18n/",
    "fontawesome",
    "font-awesome",
    "/swiper",
    "swiperjs",
    "/bootstrap/",
    "/jquery/",
    "/cdn/",
    # ASP.NET / .NET
    "/migrations/",
    "/obj/",
    "/bin/",
    "/wwwroot/",
    "/properties/",
    "/test/",
    "/tests/",
    "/testresults/",
    "/generated/",
]

DEFAULT_DENY_FILENAMES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "composer.lock",
    "poetry.lock",
    "cargo.lock",
}

# Focus + ASP / React Native / backend ek yüksek öncelik yolları
HIGH_PRIORITY_SUBSTRINGS = [
    *DEFAULT_FOCUS_SUBSTRINGS,
    "/controllers/",
    "/api/",
    "/navigation/",
    "/navigators/",
    "/screens/",
    "/redux/",
    "/slices/",
    "/features/",
    "/pages/",
    "/backend/",
    "/server/",
    "/middleware/",
    "/handlers/",
    "/data/",
    "/repositories/",
    "/repository/",
    "/src/navigation/",
    "/src/screens/",
    "/src/api/",
    "/app/navigation/",
    "/app/api/",
]

ENTRY_BASENAMES = {
    "readme.md",
    "readme.rst",
    "readme.txt",
    "package.json",
    "pyproject.toml",
    "composer.json",
    "cargo.toml",
    "index.html",
    "index.htm",
    "main.py",
    "app.py",
    "main.ts",
    "main.tsx",
    "main.js",
    "app.ts",
    "app.tsx",
    "app.js",
    "index.ts",
    "index.tsx",
    "index.js",
    "script.js",
    "program.cs",
    "startup.cs",
    "appsettings.json",
    "app.config",
    "web.config",
}

_VENDOR_LINE_RE = re.compile(
    r"(swiper|font-?awesome|bootstrap|jquery|unpkg\.com|cdnjs\.|jsdelivr|googleapis\.com/fonts)",
    re.IGNORECASE,
)
_HTML_TAG_RE = re.compile(r"<(title|meta|h[1-3])\b", re.IGNORECASE)
_HTML_STRIP_RE = re.compile(r"<(script|style|link)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)

# Path segment names treated as presentational UI (React / React Native).
_UI_DIR_SEGMENTS = frozenset(
    {
        "components",
        "component",
        "widgets",
        "widget",
        "ui",
        "styles",
        "layouts",
        "layout",
        "storybook",
        "stories",
        "__tests__",
        "__mocks__",
        "test",
        "tests",
    }
)


class IngestCancelled(Exception):
    """Raised when user cancels ingestion."""


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y"}


def _norm_repo_path(path: str) -> str:
    """Normalize repo-relative paths for substring checks."""
    return "/" + path.lower().replace("\\", "/").strip("/") + "/"


def _deny_lists() -> tuple[list[str], set[str]]:
    ignore_raw = os.getenv("INGEST_IGNORE_PATHS", "")
    extra = [s.strip().lower() for s in ignore_raw.split(",") if s.strip()]
    deny_names_raw = os.getenv("INGEST_DENY_FILENAMES", "")
    extra_names = {s.strip().lower() for s in deny_names_raw.split(",") if s.strip()}
    return DEFAULT_DENY_SUBSTRINGS + extra, DEFAULT_DENY_FILENAMES | extra_names


def _is_denied_path(path: str) -> bool:
    lowered = path.lower().replace("\\", "/")
    norm = _norm_repo_path(path)
    name = Path(lowered).name

    deny_substrings, deny_names = _deny_lists()
    if name in deny_names:
        return True
    if any(sub in norm for sub in deny_substrings):
        return True

    suffix = Path(lowered).suffix
    if suffix in {".css", ".scss", ".sass", ".less", ".map", ".min.css"}:
        return True
    if name.endswith(".min.js") or name.endswith(".bundle.js"):
        return True
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp", ".woff", ".woff2", ".ttf", ".eot"}:
        return True

    # Config noise: lockfiles and huge autogen json unless explicitly allowed
    if name.endswith(".lock") or name in deny_names:
        return True

    if suffix == ".cs":
        if (
            name.endswith(".designer.cs")
            or name.endswith(".g.cs")
            or name.endswith(".generated.cs")
            or "assemblyinfo" in name
        ):
            return True
        if name.endswith("migrationsnapshot.cs") or "/snapshot." in lowered:
            return True

    if suffix == ".json" and name.startswith("appsettings.") and name != "appsettings.json":
        return True

    parts = [p for p in norm.strip("/").split("/") if p]
    if any(part in _UI_DIR_SEGMENTS for part in parts):
        return True

    if suffix in {".ts", ".tsx", ".js", ".jsx"} and (
        ".styles." in name or ".style." in name or name.endswith(".styles.ts") or name.endswith(".styles.tsx")
    ):
        return True

    return False


def _path_priority(path: str) -> int:
    """Higher score = more important for RAG (project purpose & core code)."""
    lowered = path.lower().replace("\\", "/")
    norm = _norm_repo_path(path)
    name = Path(lowered).name

    if name.startswith("readme"):
        return 100
    if name.endswith(".csproj"):
        return 92
    if name in {"package.json", "pyproject.toml", "composer.json", "cargo.toml"}:
        return 90
    if name in {"appsettings.json", "app.config", "web.config"}:
        return 88
    if name in {"index.html", "index.htm"}:
        return 85
    if name in ENTRY_BASENAMES:
        return 80
    if any(marker in norm for marker in FOCUS_ENTRY_PATH_MARKERS):
        return 80
    if name.endswith("controller.cs"):
        return 78

    focus_raw = os.getenv("INGEST_FOCUS_PATHS", "")
    focus_list = [s.strip().lower() for s in focus_raw.split(",") if s.strip()]
    effective_focus = focus_list or (
        HIGH_PRIORITY_SUBSTRINGS
        if _env_bool("INGEST_ENABLE_SPA_FOCUS", False)
        else DEFAULT_FOCUS_SUBSTRINGS
    )
    if any(sub in norm for sub in effective_focus):
        return 75

    if any(seg in norm for seg in ("/models/", "/entities/", "/viewmodels/", "/dto/")):
        return 55

    parts = [p for p in lowered.split("/") if p]
    if len(parts) <= 2 and Path(lowered).suffix in CONFIG_EXTENSIONS | DOC_EXTENSIONS:
        return 60

    suffix = Path(lowered).suffix
    if suffix in CODE_EXTENSIONS:
        return 40
    if suffix in DOC_EXTENSIONS:
        return 55
    if suffix in {".html", ".htm"}:
        return 30
    return 0


def _file_filter(path: str) -> bool:
    lowered = path.lower().replace("\\", "/")
    if _is_denied_path(lowered):
        return False

    suffix = Path(lowered).suffix
    if suffix not in ALLOWED_EXTENSIONS:
        return False

    min_priority = int(os.getenv("INGEST_MIN_PRIORITY", "50"))
    return _path_priority(lowered) >= min_priority


def _file_filter_priority_only(path: str) -> bool:
    """Fallback when strict filter returns nothing: keep high-signal entry/docs only."""
    lowered = path.lower().replace("\\", "/")
    if _is_denied_path(lowered):
        return False
    suffix = Path(lowered).suffix
    if suffix not in ALLOWED_EXTENSIONS:
        return False
    return _path_priority(lowered) >= 55


_SIG_KEEP_RE = re.compile(
    r"^\s*(export\s+)?(default\s+)?(async\s+)?"
    r"(function|class|interface|type|enum|const|let|var)\b|^\s*import\b|^\s*from\b",
    re.IGNORECASE,
)

_CS_LINE_KEEP_RE = re.compile(
    r"^\s*("
    r"using\s|namespace\s|#pragma|#region|#endregion|"
    r"\[.*\]\s*$|"
    r"(public|private|protected|internal|static|sealed|abstract|partial|\s)+"
    r"(class|interface|enum|struct|record)\s+"
    r"|(public|private|protected|internal).*\([^;]*\)\s*(\{|=>|;)\s*$"
    r"|(public|private|protected|internal).*\{\s*get\s*;"
    r")",
    re.IGNORECASE,
)


def _strip_vendor_lines(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if _VENDOR_LINE_RE.search(line):
            continue
        lines.append(line)
    return "\n".join(lines)


def _html_semantic_only(text: str) -> str:
    """Keep title, meta description, headings and short text — drop scripts/styles."""
    cleaned = _HTML_STRIP_RE.sub("", text)
    kept: list[str] = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _HTML_TAG_RE.search(stripped) or stripped.startswith("<p"):
            kept.append(stripped)
        elif stripped.startswith("<!") or stripped.lower().startswith("<html"):
            kept.append(stripped)
    if not kept:
        # fallback: first 120 lines without script blocks
        kept = [ln for ln in cleaned.splitlines() if "<script" not in ln.lower()][:120]
    return "\n".join(kept[:80]).strip()


def _signature_only_text(text: str) -> str:
    """
    Heuristic "signature-only" reducer for JS/TS (including TSX/JSX).
    Keeps imports/exports and top-level declarations, plus nearby comments.
    """
    text = _strip_vendor_lines(text)
    lines = text.splitlines()
    out: list[str] = []
    comment_buffer: list[str] = []

    for line in lines:
        stripped = line.strip()

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

        if "<" in line and ">" in line:
            comment_buffer = []
            continue

        if stripped:
            comment_buffer = []

    return "\n".join(out).strip()


def _csharp_signature_only(text: str) -> str:
    """Keep namespaces, types and member signatures — drop method bodies."""
    text = _strip_vendor_lines(text)
    lines = text.splitlines()
    out: list[str] = []
    comment_buffer: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
            comment_buffer.append(line)
            if len(comment_buffer) > 6:
                comment_buffer = comment_buffer[-6:]
            continue

        if _CS_LINE_KEEP_RE.search(line):
            if comment_buffer:
                out.extend(comment_buffer)
                comment_buffer = []
            out.append(line)
            continue

        if stripped:
            comment_buffer = []

    return "\n".join(out).strip()


def _chunks_per_file_limit(extension: str, default_limit: int) -> int:
    per_ext = {
        ".cs": int(os.getenv("INGEST_MAX_CHUNKS_PER_CS", "10")),
        ".java": int(os.getenv("INGEST_MAX_CHUNKS_PER_JAVA", "12")),
        ".py": int(os.getenv("INGEST_MAX_CHUNKS_PER_PY", "15")),
        ".js": int(os.getenv("INGEST_MAX_CHUNKS_PER_JS", "12")),
        ".ts": int(os.getenv("INGEST_MAX_CHUNKS_PER_JS", "12")),
        ".tsx": int(os.getenv("INGEST_MAX_CHUNKS_PER_JS", "12")),
        ".jsx": int(os.getenv("INGEST_MAX_CHUNKS_PER_JS", "12")),
    }
    return per_ext.get(extension, default_limit)


def _prepare_document_content(doc: Document) -> Document:
    source = str(doc.metadata.get("path") or doc.metadata.get("source") or "")
    extension = Path(source).suffix.lower()
    text = doc.page_content if isinstance(doc.page_content, str) else str(doc.page_content)

    if extension in {".html", ".htm"}:
        doc.page_content = _html_semantic_only(text)
    elif extension in {".js", ".ts", ".tsx", ".jsx"} and _env_bool("INGEST_SIGNATURE_ONLY", True):
        reduced = _signature_only_text(text)
        if reduced:
            doc.page_content = reduced
    elif extension == ".cs" and _env_bool("INGEST_SIGNATURE_ONLY", True):
        reduced = _csharp_signature_only(text)
        if reduced:
            doc.page_content = reduced
        else:
            doc.page_content = _strip_vendor_lines(text)
    else:
        doc.page_content = _strip_vendor_lines(text)

    max_chars = int(os.getenv("INGEST_MAX_CHARS_PER_FILE", "12000"))
    if extension == ".cs":
        max_chars = int(os.getenv("INGEST_MAX_CHARS_PER_CS", "8000"))
    if len(doc.page_content) > max_chars:
        doc.page_content = doc.page_content[:max_chars]

    doc.metadata["ingest_priority"] = _path_priority(source)
    doc.metadata["path"] = source
    return doc


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
    max_chunks_per_file: int,
    cancel_event: threading.Event | None = None,
) -> list[Document]:
    chunks: list[Document] = []
    for doc in docs:
        if cancel_event and cancel_event.is_set():
            raise IngestCancelled()

        doc = _prepare_document_content(doc)
        if not doc.page_content.strip():
            continue

        source = str(doc.metadata.get("path") or doc.metadata.get("source") or "")
        extension = Path(source).suffix.lower()
        splitter = _build_splitter(
            extension=extension,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        file_chunks = splitter.split_documents([doc])
        file_limit = _chunks_per_file_limit(extension, max_chunks_per_file)
        if len(file_chunks) > file_limit:
            file_chunks = file_chunks[:file_limit]
        chunks.extend(file_chunks)
    return chunks


def _cap_chunks(chunks: list[Document], max_total: int) -> list[Document]:
    if max_total <= 0 or len(chunks) <= max_total:
        return chunks

    by_source: dict[str, list[Document]] = defaultdict(list)
    for chunk in chunks:
        source = str(chunk.metadata.get("path") or chunk.metadata.get("source") or "")
        by_source[source].append(chunk)

    sources_sorted = sorted(
        by_source.keys(),
        key=lambda s: -int(by_source[s][0].metadata.get("ingest_priority", _path_priority(s))),
    )

    kept: list[Document] = []
    for source in sources_sorted:
        for chunk in by_source[source]:
            if len(kept) >= max_total:
                return kept
            kept.append(chunk)
    return kept


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
        access_token=settings.github_token or "",
        file_filter=_file_filter,
    )
    docs = loader.load()
    if not docs:
        loader2 = GithubFileLoader(
            repo=repo_name,
            branch=repo_branch,
            access_token=settings.github_token or "",
            file_filter=_file_filter_priority_only,
        )
        docs = loader2.load()
    if not docs:
        raise RuntimeError(
            "GithubFileLoader belge döndürmedi. "
            "Repo boş olabilir, branch yanlış olabilir veya filtreler tüm dosyaları eledi. "
            "INGEST_MIN_PRIORITY=20 ile deneyin veya INGEST_IGNORE_PATHS değerini gevşetin."
        )
    if cancel_event and cancel_event.is_set():
        raise IngestCancelled()

    console.print(f"[green]{len(docs)} belge yüklendi.[/green]")
    chunks = _split_documents(
        docs=docs,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        max_chunks_per_file=settings.ingest_max_chunks_per_file,
        cancel_event=cancel_event,
    )
    before_cap = len(chunks)
    chunks = _cap_chunks(chunks, settings.ingest_max_chunks)
    if before_cap != len(chunks):
        console.print(
            f"[yellow]{before_cap} chunk üretildi, {len(chunks)} chunk ile sınırlandı "
            f"(max={settings.ingest_max_chunks}).[/yellow]"
        )
    else:
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
                f"Belge: {len(docs)}\n"
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
        "documents": len(docs),
        "index": settings.pinecone_index_name,
    }


def main() -> None:
    s = get_settings()
    if not s.repo_url:
        raise SystemExit(
            "GITHUB_REPO_URL tanımlı değil. UI'dan ingest yapın veya .env'e "
            "GITHUB_REPO_URL=owner/repo ekleyip tekrar deneyin."
        )
    ingest_repo(
        repo_url=s.repo_url,
        repo_branch=s.repo_branch,
        namespace=namespace_for_repo(s.repo_url, s.repo_branch),
        force=False,
    )


if __name__ == "__main__":
    main()
