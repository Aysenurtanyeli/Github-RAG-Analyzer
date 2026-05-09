from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import time
from typing import List, Optional
from urllib.parse import urlparse
import threading
import uuid
from dataclasses import asdict
import multiprocessing as mp

import requests

# Kendi modüllerini import et
from config import get_settings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from ingest import IngestCancelled, ingest_repo

app = FastAPI(
    title="GitHub Repo RAG API",
    description="eventHorizon ve diğer repolar için analiz API'si",
    version="1.1.0"
)

# CORS ayarları (Frontend bağlanabilmesi için)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Servislerin Hazırlanması ─────────────────────────────────────
# Not: Zorunlu env değişkenleri yoksa uygulamanın "boot" etmesini engellemek
# yerine API'nin çalışmasını sağlar, ilgili endpoint'lerde anlamlı hata döner.
settings = None
embeddings = None
llm = None
pc = None
pinecone_index = None
SERVICE_INIT_ERROR: str | None = None


def _init_services() -> None:
    global settings, embeddings, llm, pc, pinecone_index, SERVICE_INIT_ERROR
    if pinecone_index is not None and embeddings is not None and llm is not None and settings is not None:
        return
    try:
        settings = get_settings()
        embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        llm = ChatGroq(temperature=0, model_name=settings.llm_model, groq_api_key=settings.groq_api_key)

        pc = Pinecone(api_key=settings.pinecone_api_key)
        existing = {idx["name"] for idx in pc.list_indexes()}
        dim_probe = embeddings.embed_query("dimension probe")
        if settings.pinecone_index_name not in existing:
            pc.create_index(
                name=settings.pinecone_index_name,
                dimension=len(dim_probe),
                metric="cosine",
                spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
            )
        pinecone_index = pc.Index(settings.pinecone_index_name)
        SERVICE_INIT_ERROR = None
    except Exception as e:
        SERVICE_INIT_ERROR = str(e)


def _require_services() -> None:
    _init_services()
    if SERVICE_INIT_ERROR:
        raise HTTPException(
            status_code=503,
            detail=(
                "Servisler başlatılamadı. Muhtemelen .env / environment değişkenleri eksik. "
                f"Hata: {SERVICE_INIT_ERROR}"
            ),
        )

INGEST_TASKS: dict[str, dict] = {}
INGEST_LOCK = threading.Lock()

def _ingest_worker(
    queue: "mp.Queue",
    *,
    repo_url: str,
    repo_branch: str,
    namespace: str,
    force: bool,
    cancel_event,
) -> None:
    """
    Separate process worker for ingestion.
    This allows hard-cancel (terminate) when GithubFileLoader blocks.
    """
    try:
        result = ingest_repo(
            repo_url=repo_url,
            repo_branch=repo_branch,
            namespace=namespace,
            force=force,
            cancel_event=cancel_event,
        )
        queue.put(("completed", result, None))
    except IngestCancelled:
        queue.put(("cancelled", None, None))
    except Exception as e:
        queue.put(("failed", None, str(e)))

def _default_namespace() -> str:
    _require_services()
    return f"{settings.repo_name.replace('/', '__')}__{settings.repo_branch}"

def _vectorstore(namespace: str) -> PineconeVectorStore:
    _require_services()
    return PineconeVectorStore(index=pinecone_index, embedding=embeddings, namespace=namespace)

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

# ── Pydantic Modelleri ─────────────────────────────────────────────
class SoruIstek(BaseModel):
    soru: str = Field(..., min_length=2, description="Sormak istediğiniz soru")
    top_k: int = Field(default=4, ge=1, le=10)
    namespace: Optional[str] = Field(default=None, description="Repo namespace (boşsa default repo)")

class IngestIstek(BaseModel):
    repo_url: str = Field(..., min_length=3, description="GitHub repo URL veya owner/repo")
    repo_branch: str = Field(default="main")
    force: bool = Field(default=False, description="True ise namespace temizlenip tekrar ingest yapılır")

class IngestYanit(BaseModel):
    namespace: str
    chunks: int
    index: str
    repo_name: str
    branch: str

class KaynakBilgi(BaseModel):
    dosya: str
    benzerlik_skoru: float

class SoruYanit(BaseModel):
    yanit: str
    kaynaklar: List[KaynakBilgi]
    islem_suresi_ms: int

# ── Endpoint'ler ───────────────────────────────────────────────────

@app.get("/")
def status():
    _init_services()
    if SERVICE_INIT_ERROR or not settings:
        return {
            "durum": "degraded",
            "hata": SERVICE_INIT_ERROR or "Settings yüklenemedi.",
            "gerekli_env": [
                "GITHUB_PERSONAL_ACCESS_TOKEN",
                "GROQ_API_KEY",
                "PINECONE_API_KEY",
                "GITHUB_REPO_URL",
            ],
            "ui": "/ui",
        }

    ns = f"{settings.repo_name.replace('/', '__')}__{settings.repo_branch}"
    try:
        stats = pinecone_index.describe_index_stats()
        ns_count = int((stats.get("namespaces") or {}).get(ns, {}).get("vector_count") or 0)
    except Exception:
        ns_count = None
    return {
        "durum": "online",
        "default_repo": settings.repo_name,
        "default_branch": settings.repo_branch,
        "pinecone_index": settings.pinecone_index_name,
        "default_namespace": ns,
        "toplam_chunk": ns_count,
        "ui": "/ui",
        "settings": {k: v for k, v in asdict(settings).items() if k not in {"github_token", "groq_api_key", "pinecone_api_key"}},
    }

@app.post("/ingest", response_model=IngestYanit)
def ingest_endpoint(istek: IngestIstek):
    _require_services()
    repo_name = istek.repo_url.strip()
    safe = repo_name.replace("https://github.com/", "").replace("http://github.com/", "").strip("/")
    safe = safe.replace("/", "__").replace(".git", "")
    namespace = f"{safe}__{istek.repo_branch}"
    result = ingest_repo(
        repo_url=istek.repo_url,
        repo_branch=istek.repo_branch,
        namespace=namespace,
        force=istek.force,
    )
    return IngestYanit(**result)

@app.post("/ingest_async")
def ingest_async_endpoint(istek: IngestIstek):
    _require_services()
    repo_url = istek.repo_url.strip()
    safe = repo_url.replace("https://github.com/", "").replace("http://github.com/", "").strip("/")
    safe = safe.replace("/", "__").replace(".git", "")
    namespace = f"{safe}__{istek.repo_branch}"

    task_id = uuid.uuid4().hex
    cancel_event = mp.Event()
    result_queue: "mp.Queue" = mp.Queue()

    with INGEST_LOCK:
        INGEST_TASKS[task_id] = {
            "status": "running",
            "namespace": namespace,
            "repo_url": repo_url,
            "repo_branch": istek.repo_branch,
            "force": istek.force,
            "cancel_event": cancel_event,
            "process": None,
            "queue": result_queue,
            "result": None,
            "error": None,
        }

    def _runner() -> None:
        proc = mp.Process(
            target=_ingest_worker,
            kwargs={
                "queue": result_queue,
                "repo_url": repo_url,
                "repo_branch": istek.repo_branch,
                "namespace": namespace,
                "force": istek.force,
                "cancel_event": cancel_event,
            },
            daemon=True,
        )
        proc.start()
        with INGEST_LOCK:
            if task_id in INGEST_TASKS:
                INGEST_TASKS[task_id]["process"] = proc

        # Wait for completion/cancel/terminate and update task status
        while True:
            if cancel_event.is_set():
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)
                with INGEST_LOCK:
                    if task_id in INGEST_TASKS:
                        INGEST_TASKS[task_id]["status"] = "cancelled"
                        INGEST_TASKS[task_id]["result"] = None
                return

            try:
                status, result, err = result_queue.get(timeout=0.5)
            except Exception:
                if not proc.is_alive():
                    with INGEST_LOCK:
                        if task_id in INGEST_TASKS and INGEST_TASKS[task_id]["status"] == "running":
                            INGEST_TASKS[task_id]["status"] = "failed"
                            INGEST_TASKS[task_id]["error"] = "Ingest process ended unexpectedly."
                    return
                continue

            with INGEST_LOCK:
                if task_id not in INGEST_TASKS:
                    return
                INGEST_TASKS[task_id]["status"] = status
                INGEST_TASKS[task_id]["result"] = result
                INGEST_TASKS[task_id]["error"] = err
            if proc.is_alive():
                proc.join(timeout=2)
            return

    threading.Thread(target=_runner, daemon=True).start()

    return {"task_id": task_id, "status": "running", "namespace": namespace}


@app.post("/ingest_cancel")
def ingest_cancel_endpoint(task_id: str):
    with INGEST_LOCK:
        task = INGEST_TASKS.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="İşlem bulunamadı.")
        if task["status"] not in {"running"}:
            return {"task_id": task_id, "status": task["status"]}
        task["cancel_event"].set()
        task["status"] = "cancelling"
        proc = task.get("process")
        if proc is not None and getattr(proc, "is_alive", None) and proc.is_alive():
            try:
                proc.terminate()
            except Exception:
                pass
        # UI polling'in takılmaması için "best-effort" iptal:
        # Proses terminate denendi; task durumu hemen "cancelled" yapılır.
        task["status"] = "cancelled"
        task["result"] = None
        task["error"] = None
    return {"task_id": task_id, "status": "cancelled"}


@app.get("/ingest_status")
def ingest_status_endpoint(task_id: str):
    with INGEST_LOCK:
        task = INGEST_TASKS.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="İşlem bulunamadı.")

        out = {
            "task_id": task_id,
            "status": task["status"],
            "namespace": task["namespace"],
        }
        if task.get("result"):
            out["result"] = task["result"]
        if task.get("error"):
            out["error"] = task["error"]
        return out

@app.get("/branches")
def list_branches(repo_url: str):
    """
    Repo branch listesini döner (GitHub API).
    UI branch input'u için öneri listesi olarak kullanılır.
    """
    try:
        repo_name = _repo_name_from_url(repo_url.strip())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    _require_services()
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"
    branches: list[str] = []
    page = 1
    while page <= 5:  # pratik limit: ilk 500 branch yeterli
        r = requests.get(
            f"https://api.github.com/repos/{repo_name}/branches",
            headers=headers,
            params={"per_page": 100, "page": page},
            timeout=30,
        )
        if r.status_code == 404:
            raise HTTPException(status_code=404, detail="Repo bulunamadı veya erişim yok.")
        if r.status_code == 401 or r.status_code == 403:
            raise HTTPException(status_code=403, detail="GitHub token yetkisiz/limitli olabilir.")
        if not r.ok:
            raise HTTPException(status_code=500, detail=f"GitHub API hata: {r.status_code}")
        data = r.json() or []
        if not data:
            break
        branches.extend([b.get("name") for b in data if b.get("name")])
        if len(data) < 100:
            break
        page += 1

    # sık kullanılan branch'ları öne al
    preferred = ["main", "master", "develop", "dev", "staging", "production"]
    uniq = list(dict.fromkeys(branches))
    uniq.sort(key=lambda x: (0 if x in preferred else 1, x.lower()))
    return {"repo": repo_name, "branches": uniq[:200]}

@app.post("/sor", response_model=SoruYanit)
async def ask(istek: SoruIstek):
    _require_services()
    baslangic = time.time()
    
    # 1. Retrieval (Benzerlik araması ve skorlama)
    ns = istek.namespace or _default_namespace()
    try:
        stats = pinecone_index.describe_index_stats()
        ns_count = int((stats.get("namespaces") or {}).get(ns, {}).get("vector_count") or 0)
    except Exception:
        ns_count = None
    if ns_count == 0:
        raise HTTPException(
            status_code=404,
            detail="Bu repo için veri bulunamadı. Önce 'Ingest Başlat' ile yükleyin.",
        )
    results = _vectorstore(ns).similarity_search_with_score(istek.soru, k=istek.top_k)
    
    if not results:
        raise HTTPException(status_code=404, detail="İlgili bilgi bulunamadı.")

    # 2. Augment (Bağlam oluşturma)
    context_text = ""
    sources = []
    for doc, score in results:
        context_text += f"\n---\nKAYNAK: {doc.metadata.get('source')}\nİÇERİK: {doc.page_content}\n"
        sources.append(KaynakBilgi(
            dosya=doc.metadata.get("source", "Bilinmiyor"),
            benzerlik_skoru=round(float(score), 4)
        ))

    # 3. Generate (LLM Yanıtı)
    sistem_mesaji = f"Sen bir yazılım uzmanısın. Aşağıdaki kod parçalarını ve dökümanları kullanarak soruyu yanıtla.\n\nBAĞLAM:\n{context_text}"
    full_prompt = f"{sistem_mesaji}\n\nSORU: {istek.soru}"
    
    response = llm.invoke(full_prompt)
    
    return SoruYanit(
        yanit=response.content,
        kaynaklar=sources,
        islem_suresi_ms=int((time.time() - baslangic) * 1000)
    )

# ── Web UI ───────────────────────────────────────────────────
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

@app.get("/ui")
def ui():
    index = WEB_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="UI dosyası bulunamadı. web/index.html eksik.")
    return FileResponse(str(index))