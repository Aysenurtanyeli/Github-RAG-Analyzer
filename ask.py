"""Interactive terminal Q&A over ingested GitHub repository data."""

from __future__ import annotations

from typing import Iterable

from groq import BadRequestError
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from config import get_settings, namespace_for_repo
from embeddings_provider import build_embeddings
from rag_prompts import build_rag_prompt, rerank_retrieved_docs

console = Console()
FALLBACK_MODELS = ("llama-3.3-70b-versatile", "mixtral-8x7b-32768")


def _format_sources(documents: Iterable) -> str:
    unique_sources: list[str] = []
    seen: set[str] = set()

    for doc in documents:
        source = (
            doc.metadata.get("path")
            or doc.metadata.get("source")
            or doc.metadata.get("file_path")
            or "unknown"
        )
        source = str(source)
        if source not in seen:
            seen.add(source)
            unique_sources.append(source)

    if not unique_sources:
        return "- unknown"
    return "\n".join(f"- {item}" for item in unique_sources)


def main() -> None:
    settings = get_settings()
    if not settings.repo_url:
        console.print(
            Panel.fit(
                "GITHUB_REPO_URL tanımlı değil.\n"
                "Web arayüzünden repo seçip ingest yapın veya .env'e GITHUB_REPO_URL ekleyin.",
                title="Yapılandırma",
                border_style="red",
            )
        )
        raise SystemExit(1)

    embeddings = build_embeddings(settings)
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
    index = pc.Index(settings.pinecone_index_name)
    namespace = namespace_for_repo(settings.repo_url, settings.repo_branch)
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings, namespace=namespace)

    fetch_k = min(max(settings.retriever_k * 3, settings.retriever_k), 12)
    retriever = vectorstore.as_retriever(search_kwargs={"k": fetch_k})
    llm = ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.llm_model,
        temperature=0.1,
    )

    console.print(
        Panel.fit(
            (
                "GitHub RAG Analyzer hazır.\n"
                "Çıkmak için: [bold]exit[/bold] veya [bold]quit[/bold]"
            ),
            title="Terminal Chat",
            border_style="cyan",
        )
    )

    while True:
        question = Prompt.ask("\n[bold blue]Soru[/bold blue]").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            console.print("[yellow]Görüşmek üzere.[/yellow]")
            break

        docs = retriever.invoke(question)
        docs = rerank_retrieved_docs(docs, settings.retriever_k)
        if not docs:
            console.print(
                Panel.fit(
                    "Uygun kaynak bulunamadı.",
                    title="Sonuç Yok",
                    border_style="yellow",
                )
            )
            continue

        context = "\n\n".join(
            f"[Kaynak {i + 1}] {doc.page_content}" for i, doc in enumerate(docs)
        )
        prompt = build_rag_prompt(question=question, context=context)
        try:
            response = llm.invoke(prompt)
        except BadRequestError as err:
            message = str(err)
            if "model_decommissioned" in message or "decommissioned" in message:
                switched = False
                for fallback_model in FALLBACK_MODELS:
                    if fallback_model == settings.llm_model:
                        continue
                    try:
                        llm = ChatGroq(
                            api_key=settings.groq_api_key,
                            model=fallback_model,
                            temperature=0.1,
                        )
                        response = llm.invoke(prompt)
                        switched = True
                        console.print(
                            f"[yellow]Model kapali oldugu icin su modele gecildi: {fallback_model}[/yellow]"
                        )
                        break
                    except BadRequestError:
                        continue
                if not switched:
                    console.print(
                        Panel.fit(
                            (
                                "GROQ_MODEL degeri kullanilamiyor. "
                                "Lutfen .env dosyasinda aktif bir model kullanin "
                                "(onerilen: llama-3.3-70b-versatile)."
                            ),
                            title="Model Hatasi",
                            border_style="red",
                        )
                    )
                    continue
            else:
                raise
        answer = response.content if hasattr(response, "content") else str(response)

        console.print(
            Panel(
                answer,
                title="Cevap",
                border_style="green",
            )
        )
        console.print(
            Panel(
                _format_sources(docs),
                title="Kaynaklar",
                border_style="magenta",
            )
        )


if __name__ == "__main__":
    main()
