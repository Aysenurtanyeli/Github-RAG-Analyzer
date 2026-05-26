# GitHub Repo Analyzer

A RAG (Retrieval-Augmented Generation) application that indexes GitHub repositories into the Pinecone vector database and answers questions in repo context using the Groq LLM.

Across web, mobile, backend API, desktop, data/ML scripts, and monorepos, it focuses on **what the project does**, **the technologies used**, and **the core architecture**; it filters out unnecessary UI libraries and static-file noise.

---

## Features

- Fetch files from GitHub (`GithubFileLoader`) with smart filtering (ignore / focus lists)
- HuggingFace embeddings + Pinecone namespaces (isolation per repo + branch)
- Web UI: repo ingest, branch selection, streaming chat
- REST API: `/ingest_async`, `/sor`, `/sor/stream` (SSE)
- Terminal: `ingest.py`, `ask.py`
- C# / JS / TS signature mode (method bodies dropped, fewer chunks)
- Source file prioritization (README, `package.json`, controllers, services…)

---

## Supported project types

### Web and frontend

- Static sites, portfolios (HTML/CSS/JS)
- React, Next.js, Vue, Angular, Svelte
- SPA / SSR (`.js`, `.ts`, `.tsx`, `.jsx`)

### Mobile

- **React Native** (Expo / bare): `App.tsx`, `navigation/`, `screens/`, `hooks/`, `services/`
- Mobile + API: backend `services/`, `api/`, `controllers/` (same or separate repo)

### Backend and API

- **ASP.NET / .NET**: `Program.cs`, `Startup.cs`, `*Controller.cs`, `.csproj`, `appsettings.json`
- **Node.js**: `routes/`, `controllers/`, `middleware/`, `services/`
- **Python**: FastAPI, Django, Flask — `main.py`, `app.py`, `routes/`, `api/`
- **Java**: Spring-style `controllers/`, `services/`, `.java`

### Data, automation, tooling

- Python scripts / small libraries (`.py`)
- Manifest files such as `pyproject.toml`, `composer.json`, `cargo.toml`

### Other

- Monorepos (`apps/`, `packages/` layouts)
- Documentation-heavy repos (`README`, `.md`)
- Full-stack (frontend + API in one repo)

**Note:** Extensions such as `.dart`, `.go`, `.rb`, `.php` are not loaded unless listed in the default set. To extend support, update `ALLOWED_EXTENSIONS` in `ingest.py`.

---

## Which files are analyzed?

### Loaded extensions

| Category | Extensions |
|----------|------------|
| Code | `.py`, `.java`, `.cs`, `.js`, `.ts`, `.tsx`, `.jsx` |
| Docs | `.md`, `.mdx`, `.txt`, `.rst` |
| Config | `.json`, `.toml`, `.yml`, `.yaml`, `.ini` |
| Web entry | `.html`, `.htm` (title, meta, h1–h3 summary) |
| .NET | `.csproj` |

### High priority (processed first)

- `README*`, `package.json`, `pyproject.toml`, `composer.json`, `cargo.toml`
- Entry points: `App.tsx`, `main.ts`, `index.js`, `Program.cs`, `Startup.cs`, `appsettings.json`
- Business logic: `services/`, `hooks/`, `routes/`, `store/`, `context/`, `utils/`
- API: `controllers/`, `api/`, `middleware/`, `handlers/`, `repositories/`, `data/`
- Mobile: `navigation/`, `navigators/`, `screens/` (excluding `components/`)
- State: `redux/`, `slices/`, `features/`
- Targeted folders under `src/...` and `app/...` (not all of `src/**`)

### Intentionally excluded

- UI: `components/`, `ui/`, `styles/`, `layouts/`, `widgets/`, `storybook/`
- Static: `assets/`, `public/`, `wwwroot/`, `img/`, `fonts/`, `media/`
- Build / dependencies: `node_modules/`, `vendor/`, `dist/`, `build/`, `.next/`, `obj/`, `bin/`
- Tests: `tests/`, `__tests__/`, `test/`
- Noise: lock files, `.min.js`, `.css`/`.scss`, images, EF `Migrations/`, `*.Designer.cs`

### Content processing

- **JS/TS/JSX/TSX:** signature mode (imports, class and method signatures)
- **C#:** signature mode (namespace, types, method signatures)
- **HTML:** semantic summary
- Limits: ~400 chunks per repo (configurable), separate caps per file and for `.cs`

---

## Architecture

```
GitHub Repo  →  ingest.py  →  Chunk + Embed  →  Pinecone (namespace)
                                                      ↓
User question  →  Similarity search  →  Context + Groq LLM  →  Response (stream)
```

| File | Role |
|------|------|
| `ingest.py` | Filtering, chunking, writing to Pinecone |
| `rag_api.py` | FastAPI, ingest jobs, Q&A, web UI |
| `ask.py` | Terminal chat |
| `config.py` | `.env` settings |
| `rag_prompts.py` | LLM prompt + source rerank |
| `web/` | Static UI |

---

## Setup

### Requirements

- Python 3.11+
- [Groq API](https://console.groq.com/) key
- [Pinecone](https://www.pinecone.io/) key
- (Recommended) [GitHub Personal Access Token](https://github.com/settings/tokens) — private repos and rate limits

### Steps

```powershell
cd Github-RAG-Analyzer
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_key
PINECONE_API_KEY=your_pinecone_key
GITHUB_PERSONAL_ACCESS_TOKEN=ghp_...

PINECONE_INDEX_NAME=github-rag-analyzer
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
GROQ_MODEL=llama-3.3-70b-versatile
```

### API and UI

```powershell
.\.venv\Scripts\python.exe -m uvicorn rag_api:app --reload --host 127.0.0.1 --port 8000
```

- UI: http://127.0.0.1:8000/ui  
- API status: http://127.0.0.1:8000/

If `Activate.ps1` does not work in PowerShell, use `.\.venv\Scripts\python.exe` directly without activating the venv.

### Terminal usage

`GITHUB_REPO_URL=owner/repo` is required in `.env`:

```powershell
.\.venv\Scripts\python.exe ingest.py
.\.venv\Scripts\python.exe ask.py
```

---

## Usage (Web UI)

1. Enter a repo URL or `owner/repo`  
2. Select a branch (list loads automatically)  
3. **Load and index repo data** — if the repo changed, set **Force: Yes**  
4. Ask a question — the answer arrives via **streaming** (character by character)  

Each repo gets its own Pinecone **namespace**: `owner__repo__branch`

---

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | Yes | — | LLM |
| `PINECONE_API_KEY` | Yes | — | Vector DB |
| `GITHUB_PERSONAL_ACCESS_TOKEN` | No | — | Private repo / limits |
| `GITHUB_REPO_URL` | No | — | Terminal ingest/ask only |
| `PINECONE_INDEX_NAME` | No | `github-rag-analyzer` | Index name |
| `CHUNK_SIZE` | No | `1000` | Chunk size |
| `CHUNK_OVERLAP` | No | `80` | Overlap |
| `INGEST_MAX_CHUNKS` | No | `400` | Max chunks per repo |
| `INGEST_MAX_CHUNKS_PER_FILE` | No | `15` | Max chunks per file |
| `INGEST_MAX_CHUNKS_PER_CS` | No | `10` | Per `.cs` file |
| `INGEST_SIGNATURE_ONLY` | No | `true` | JS/TS/C# signature mode |
| `INGEST_MIN_PRIORITY` | No | `50` | Drop low-priority files |
| `INGEST_IGNORE_PATHS` | No | — | Extra ignore paths (comma-separated) |
| `INGEST_FOCUS_PATHS` | No | — | Extra focus paths |
| `INGEST_ENABLE_SPA_FOCUS` | No | `false` | Extended focus list |
| `RETRIEVER_TOP_K` | No | `4` | Number of sources to retrieve |

---

## Ingest filters

### Ignore (not loaded)

`components`, `ui`, `styles`, `assets`, `public`, `layouts`, `wwwroot`, `migrations`, test folders, lock files, minified bundles, and similar paths.

### Focus (high priority)

`services`, `store`, `hooks`, `routes`, `context`, `utils`, entry files (`App.tsx`, `main.ts`, `Program.cs`…), README, `package.json`, controllers.

Customization:

```env
INGEST_IGNORE_PATHS=/legacy/,/vendor/
INGEST_FOCUS_PATHS=/core/,/domain/
```

---

## API summary

| Endpoint | Description |
|----------|-------------|
| `GET /` | Service status |
| `GET /ui` | Web UI |
| `GET /branches?repo_url=...` | Branch list |
| `POST /ingest_async` | Background ingest |
| `GET /ingest_status?task_id=...` | Ingest status |
| `POST /sor` | Single JSON response |
| `POST /sor/stream` | SSE streaming response |

---

## Challenges encountered and solutions

### 1. Thousands of chunks on a simple repo (e.g. 3475)

**Problem:** The SPA-focused filter did not match small repos like portfolios; “relaxed” mode loaded almost everything. Large `script.js` and `CHUNK_SIZE=1200` inflated chunk count.

**Solution:**
- Priority-based file selection (README and entry files first)
- `INGEST_MAX_CHUNKS` (default 400) and per-file limits
- JS/TS/C# **signature mode** (imports + class/method signatures instead of full bodies)
- Filtering vendor/CDN lines

---

### 2. Third-party libraries dominating answers

**Problem:** Third-party library lines dominated embeddings; README stayed weak.

**Solution:**
- High `ingest_priority` for README and `package.json`
- Post-retrieval **rerank** (`rag_prompts.py`)
- LLM prompt: “main idea / tech stack first, library details later”

---

### 3. ASP.NET repos constantly hitting the 800-chunk cap

**Problem:** Dozens of full `.cs` files + broad `/src/` matching; Designer, Migration, `wwwroot` noise.

**Solution:**
- C# signature mode, `INGEST_MAX_CHUNKS_PER_CS=10`
- Deny: `Migrations/`, `obj/`, `bin/`, `wwwroot/`, `*.Designer.cs`
- Priority: `Program.cs`, `*Controller.cs`, `appsettings.json`, `.csproj`
- Targeted subpaths (`services`, `controllers`, `data`) instead of all of `/src/`

---

### 4. React Native: documents/chunks with `components`

**Problem:** Everything under `/src/` and `/app/` (including components) was ingested with high priority.

**Solution:**
- Ignore `/components/`, `/ui/`, `/styles/`, `/layouts/`
- Path segment checks (`components`, `widgets`, `ui`…)
- Drop `*.styles.ts(x)` files
- Focus: `hooks`, `services`, `navigation`, `screens` (excluding components)

---

### 5. Asking questions without ingest

**Problem:** Empty Pinecone namespace led to hallucinations or errors.

**Solution:** API returns `404` + UI: “Index first”. Namespace is set when ingest completes.

---

### 6. GithubFileLoader slow / cancel

**Problem:** Loader can take a long time on large repos.

**Solution:** `ingest_async` in a separate process; cancel via `ingest_cancel`. Batch embedding (groups of 40).

---

## Project structure

```
Github-RAG-Analyzer/
├── ingest.py          # Indexing logic
├── rag_api.py         # FastAPI server
├── ask.py             # Terminal chat
├── config.py          # Settings
├── rag_prompts.py     # Prompt + rerank
├── requirements.txt
├── render.yaml        # Render Blueprint (optional)
├── runtime.txt        # Python version for Render
├── web/
│   ├── index.html
│   ├── app.js
│   └── styles.css
└── .env               # (gitignore — you can add .env.example)
```
