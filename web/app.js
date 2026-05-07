const $ = (id) => document.getElementById(id);

const state = {
  namespace: null,
  busy: false,
  ingestTaskId: null,
  ingestPollTimer: null,
};

function setNamespace(ns) {
  state.namespace = ns || null;
  $("activeNs").textContent = state.namespace || "-";
  $("nsRow").hidden = !state.namespace;
}

function setBusy(b) {
  state.busy = b;
  $("btnIngest").disabled = b;
  $("btnAsk").disabled = b;
  $("btnStopIngest").disabled = !(b && !!state.ingestTaskId);
  $("status").textContent = b ? "Çalışıyor…" : "Hazır";
}

function addMessage(role, text) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role === "you" ? "you" : "bot"}`;

  const r = document.createElement("div");
  r.className = "role";
  r.textContent = role === "you" ? "Sen" : "Asistan";

  const t = document.createElement("div");
  t.className = "text";
  t.textContent = text;

  wrap.appendChild(r);
  wrap.appendChild(t);
  $("messages").appendChild(wrap);
  $("messages").scrollTop = $("messages").scrollHeight;
}

function showSources(items) {
  const box = $("sources");
  box.innerHTML = "";
  if (!items || items.length === 0) {
    box.hidden = true;
    return;
  }
  box.hidden = false;

  const title = document.createElement("div");
  title.className = "sourcesTitle";
  title.textContent = "Kaynaklar";
  box.appendChild(title);

  const header = document.createElement("div");
  header.className = "sourceItem header";
  const hLeft = document.createElement("div");
  hLeft.className = "badge";
  hLeft.textContent = "Kaynak";
  const hRight = document.createElement("div");
  hRight.className = "score";
  hRight.textContent = "Benzerlik oranı";
  header.appendChild(hLeft);
  header.appendChild(hRight);
  box.appendChild(header);

  for (const it of items) {
    const row = document.createElement("div");
    row.className = "sourceItem";

    const left = document.createElement("div");
    left.className = "badge";
    left.textContent = it.dosya || "Bilinmiyor";

    const right = document.createElement("div");
    right.className = "score";
    right.textContent = String(it.benzerlik_skoru);

    row.appendChild(left);
    row.appendChild(right);
    box.appendChild(row);
  }
}

async function api(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try {
      const j = await res.json();
      msg = j?.detail || j?.message || msg;
    } catch {}
    throw new Error(msg);
  }
  return await res.json();
}

async function apiGet(path) {
  const res = await fetch(path, { method: "GET" });
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try {
      const j = await res.json();
      msg = j?.detail || j?.message || msg;
    } catch {}
    throw new Error(msg);
  }
  return await res.json();
}

let branchFetchTimer = null;
async function loadBranchesForRepo() {
  const repo_url = $("repoUrl").value.trim();
  if (!repo_url) return;

  try {
    const out = await apiGet(`/branches?repo_url=${encodeURIComponent(repo_url)}`);
    const sel = $("branch");
    const current = sel.value || "main";
    sel.innerHTML = "";
    for (const b of out.branches || []) {
      const opt = document.createElement("option");
      opt.value = b;
      opt.textContent = b;
      sel.appendChild(opt);
    }
    // eski seçimi mümkünse koru, yoksa main/master seç
    const values = new Set(Array.from(sel.options).map((o) => o.value));
    if (values.has(current)) sel.value = current;
    else if (values.has("main")) sel.value = "main";
    else if (values.has("master")) sel.value = "master";

    $("ingestResult").hidden = false;
    $("ingestResult").textContent = `Branch listesi yüklendi (${(out.branches || []).length}).`;
  } catch (e) {
    $("ingestResult").hidden = false;
    $("ingestResult").textContent = `Branch alınamadı: ${e.message}`;
  }
}

async function refreshStatus() {
  try {
    const res = await fetch("/");
    const j = await res.json();
    $("status").textContent = "Hazır";
    // İlk açılışta namespace göstermiyoruz; ingest sonrası set edilecek.
  } catch {
    $("status").textContent = "API yok";
  }
}

function markRepoChanged() {
  // Repo/branch değiştiyse eski namespace ile sorgu yapılmasın.
  setNamespace(null);
  $("ingestResult").hidden = false;
  $("ingestResult").textContent = "Bu repo için önce 'Repo verisini yükle ve indeksle' yapmalısın.";

  // repo değişince branch listesini otomatik yenile (küçük debounce)
  if (branchFetchTimer) clearTimeout(branchFetchTimer);
  branchFetchTimer = setTimeout(() => {
    loadBranchesForRepo();
  }, 350);
}

$("repoUrl").addEventListener("input", markRepoChanged);
$("branch").addEventListener("change", markRepoChanged);

$("btnIngest").addEventListener("click", async () => {
  const repo_url = $("repoUrl").value.trim();
  const repo_branch = $("branch").value || "main";
  const force = $("force").value === "true";
  if (!repo_url) return;

  setBusy(true);
  state.ingestTaskId = null;
  $("ingestResult").hidden = true;
  try {
    const out = await api("/ingest_async", { repo_url, repo_branch, force });
    state.ingestTaskId = out.task_id;
    $("btnStopIngest").disabled = false;

    $("ingestResult").hidden = false;
    $("ingestResult").textContent = "İndeksleme başlatıldı. Bu işlem uzun sürebilir…";

    const pollOnce = async () => {
      if (!state.ingestTaskId) return;
      try {
        const st = await apiGet(
          `/ingest_status?task_id=${encodeURIComponent(state.ingestTaskId)}`
        );

        const taskStatus = st.status || "-";
        if (taskStatus === "running") {
          $("ingestResult").textContent = "Yükleniyor ve indeksleniyor…";
          state.ingestPollTimer = setTimeout(pollOnce, 1200);
          return;
        }
        if (taskStatus === "cancelling") {
          $("ingestResult").textContent = "Durduruluyor…";
          state.ingestPollTimer = setTimeout(pollOnce, 1200);
          return;
        }

        if (taskStatus === "completed") {
          if (st.result?.namespace) setNamespace(st.result.namespace);
          $("ingestResult").textContent = `İndeksleme tamamlandı: ${st.result?.repo_name || ""} (${st.result?.branch || ""})`;
          state.ingestTaskId = null;
          $("btnStopIngest").disabled = true;
          setBusy(false);
          state.ingestPollTimer = null;
          return;
        }

        if (taskStatus === "cancelled") {
          $("ingestResult").textContent = "İndeksleme iptal edildi.";
          state.ingestTaskId = null;
          $("btnStopIngest").disabled = true;
          setBusy(false);
          state.ingestPollTimer = null;
          return;
        }

        if (taskStatus === "failed") {
          $("ingestResult").textContent = `İndeksleme başarısız: ${st.error || "bilinmeyen hata"}`;
          state.ingestTaskId = null;
          $("btnStopIngest").disabled = true;
          setBusy(false);
          state.ingestPollTimer = null;
          return;
        }

        $("ingestResult").textContent = `Bilinmeyen durum: ${taskStatus}`;
        state.ingestTaskId = null;
        $("btnStopIngest").disabled = true;
        setBusy(false);
        state.ingestPollTimer = null;
      } catch (e) {
        $("ingestResult").hidden = false;
        $("ingestResult").textContent = `Polling hatası: ${e.message}`;
        state.ingestTaskId = null;
        $("btnStopIngest").disabled = true;
        setBusy(false);
        state.ingestPollTimer = null;
      }
    };

    pollOnce();
  } catch (e) {
    $("ingestResult").hidden = false;
    $("ingestResult").textContent = `Hata: ${e.message}`;
    state.ingestTaskId = null;
    $("btnStopIngest").disabled = true;
    setBusy(false);
  } finally {
    // başarılı/iptal/başarısız durumları polling içinde set ediliyor
  }
});

$("btnStopIngest").addEventListener("click", async () => {
  if (!state.ingestTaskId) return;
  setBusy(true);
  $("btnStopIngest").disabled = true;
  $("ingestResult").hidden = false;
  $("ingestResult").textContent = "Durduruluyor…";
  try {
    await fetch(
      `/ingest_cancel?task_id=${encodeURIComponent(state.ingestTaskId)}`,
      { method: "POST" }
    );
  } catch (e) {
    $("ingestResult").textContent = `Durdurulamadı: ${e.message}`;
  }
});

$("btnAsk").addEventListener("click", async () => {
  const soru = $("question").value.trim();
  const top_k = Number($("topk").value);
  if (!soru) return;

  if (!state.namespace) {
    addMessage("bot", "Önce repo için 'Repo verisini yükle ve indeksle' yapmalısın.");
    return;
  }

  addMessage("you", soru);
  $("question").value = "";
  showSources([]);

  setBusy(true);
  try {
    const out = await api("/sor", { soru, top_k, namespace: state.namespace });
    addMessage("bot", out.yanit || "");
    showSources(out.kaynaklar || []);
  } catch (e) {
    addMessage("bot", `Hata: ${e.message}`);
  } finally {
    setBusy(false);
  }
});

$("btnClear").addEventListener("click", () => {
  $("messages").innerHTML = "";
  showSources([]);
});

refreshStatus();

