"""Shared RAG prompt and retrieval reranking for ask.py and rag_api.py."""

from __future__ import annotations

from pathlib import Path

RAG_SYSTEM_INSTRUCTIONS = """
Sen bir GitHub repo analiz asistanısın.
Yanıt önceliği:
1) Projenin ana fikri / amacı (ör. portfolyo sitesi, e-ticaret API, mobil uygulama)
2) Kullanılan teknoloji yığını ve mimari
3) Önemli modüller, giriş noktaları ve işlevler

Üçüncü parti UI kütüphaneleri (Swiper, Font Awesome, Bootstrap CDN vb.) ana fikir değildir;
bunlara sadece kısa atıf yap, asıl odağı proje amacında tut.
Sadece verilen bağlamı kullan; yetersizse açıkça belirt.
""".strip()


def build_rag_prompt(*, question: str, context: str) -> str:
    return (
        f"{RAG_SYSTEM_INSTRUCTIONS}\n\n"
        f"Soru:\n{question}\n\n"
        f"Bağlam:\n{context}"
    )


def _score_path(path: str) -> int:
    lowered = path.lower().replace("\\", "/")
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
    if name.endswith("controller.cs"):
        return 78
    if name in {
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
    }:
        return 80
    for sub in (
        "/services/",
        "/routes/",
        "/controllers/",
        "/api/",
        "/src/",
        "/app/",
        "/pages/",
        "/hooks/",
        "/store/",
        "/context/",
        "/utils/",
        "/middleware/",
        "/handlers/",
    ):
        if sub in lowered:
            return 70
    if "/models/" in lowered or "/entities/" in lowered or "/viewmodels/" in lowered:
        return 55
    parts = [p for p in lowered.split("/") if p]
    if len(parts) <= 2 and Path(lowered).suffix in {".md", ".json", ".yml", ".yaml", ".toml"}:
        return 60
    if Path(lowered).suffix in {".py", ".java", ".cs", ".js", ".ts", ".tsx", ".jsx"}:
        return 40
    return 20


def rerank_retrieved_docs(docs: list, k: int) -> list:
    """Prefer README, package.json, entry files over vendor/library chunks."""
    if not docs or len(docs) <= k:
        return docs

    def sort_key(doc):
        path = str(
            doc.metadata.get("path")
            or doc.metadata.get("source")
            or doc.metadata.get("file_path")
            or ""
        )
        explicit = doc.metadata.get("ingest_priority")
        priority = int(explicit) if explicit is not None else _score_path(path)
        return (-priority, path)

    ranked = sorted(docs, key=sort_key)
    return ranked[:k]
