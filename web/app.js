const $ = id => document.getElementById(id);
const thread = $("thread");
const form = $("form");
const input = $("question");
const send = $("send");
const selfCorrect = $("selfCorrect");
const tpl = $("answerTpl");
const MAX_UPLOAD = 50 * 1024 * 1024;
const ACTIVE = new Set(["queued", "parsing", "indexing"]);
const STATE_TEXT = { queued: "Queued", parsing: "Parsing…", indexing: "Indexing…" };

let busy = false;
let docCount = 0;
let pollTimer = null;
const localErrors = [];          // client-side rejections (too big, network) shown alongside server jobs
const dismissed = new Set();     // failed jobs the user closed

const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text; // textContent only: document and model text is never parsed as HTML
  return node;
};

const scrollDown = () => thread.scrollTo({ top: thread.scrollHeight, behavior: "smooth" });

/* ---------- library ---------- */

function setDocCount(n) {
  docCount = n;
  $("docCount").textContent = n;
  $("libEmpty").hidden = n > 0;
  const empty = $("empty");
  if (empty) {
    $("emptyTitle").textContent = n ? `Ask about your ${n} document${n > 1 ? "s" : ""}` : "Upload a document to start";
    $("emptyText").textContent = n
      ? "Veritas searches by meaning and keywords, keeps only relevant passages, rewrites the search when they are weak, and checks every answer against the source."
      : "Add PDFs, Word files, slides, spreadsheets or scans. Veritas parses tables and layout, then answers with citations, rewriting its search and checking every answer against the source before replying.";
  }
  updateComposer();
}

function updateComposer() {
  const ready = docCount > 0;
  input.disabled = !ready;
  send.disabled = !ready || busy;
  input.placeholder = ready ? "Ask a question…" : "Upload a document first";
}

// time left for an upload, from its pace so far
function eta(j) {
  if (!j.started || !j.pages_done || j.pages_done >= j.pages_total) return "";
  const left = (Date.now() / 1000 - j.started) / j.pages_done * (j.pages_total - j.pages_done);
  return left < 60 ? " · under a minute left" : ` · about ${Math.round(left / 60)} min left`;
}

async function refreshStats() {
  try {
    const s = await (await fetch("/api/stats")).json();
    $("spend").textContent = `AI spend this month: $${s.spent_usd.toFixed(2)} of $${s.budget_usd.toFixed(0)}`;
  } catch { /* stats are informational */ }
}

function renderJobs(jobs) {
  const list = $("jobs");
  list.replaceChildren();
  const shown = [...localErrors, ...jobs.filter(j => !dismissed.has(j.id))];
  shown.forEach(j => {
    const li = el("li", `job ${j.status === "error" ? "error" : ""}`);
    if (ACTIVE.has(j.status)) li.append(el("span", "spin"));
    li.append(el("span", "name", j.name));
    const progress = j.pages_total ? `${j.pages_done ?? 0}/${j.pages_total} pages${eta(j)} · searchable as it goes` : null;
    li.append(el("span", "state", j.status === "error" ? j.error : (j.status === "parsing" && progress) || STATE_TEXT[j.status] || j.status));
    if (j.status === "error") {
      const close = el("button", "icon-btn", "×");
      close.type = "button";
      close.setAttribute("aria-label", `Dismiss error for ${j.name}`);
      close.onclick = () => {
        if (j.local) localErrors.splice(localErrors.indexOf(j), 1);
        else dismissed.add(j.id);
        refreshLibrary();
      };
      li.append(close);
    }
    list.append(li);
  });
}

function renderDocs(docs) {
  const list = $("docs");
  list.replaceChildren();
  docs.forEach(d => {
    const li = el("li", "doc");
    const info = el("div", "info");
    info.append(el("div", "name", d.name), el("div", "sub",
      `${d.chunks} chunk${d.chunks === 1 ? "" : "s"}${d.tables ? ` · ${d.tables} with tables` : ""}`));
    info.firstChild.title = d.name;
    const remove = el("button", "icon-btn", "×");
    remove.type = "button";
    remove.setAttribute("aria-label", `Remove ${d.name}`);
    remove.onclick = () => removeDoc(d);
    li.append(info, remove);
    list.append(li);
  });
  setDocCount(docs.length);
}

async function refreshLibrary() {
  try {
    const res = await fetch("/api/library");
    const data = await res.json();
    renderDocs(data.documents);
    renderJobs(data.jobs);
    clearTimeout(pollTimer);
    if (data.jobs.some(j => ACTIVE.has(j.status))) pollTimer = setTimeout(refreshLibrary, 2000);
  } catch {
    pollTimer = setTimeout(refreshLibrary, 4000); // server restarting; try again
  }
}

async function removeDoc(d) {
  if (!confirm(`Remove "${d.name}" from the library?`)) return;
  const res = await fetch(`/api/documents/${d.doc_id}`, { method: "DELETE" });
  if (!res.ok) alert((await res.json()).error || "Could not remove document");
  refreshLibrary();
}

async function uploadFiles(files) {
  for (const file of files) {
    if (file.size > MAX_UPLOAD) {
      localErrors.push({ id: crypto.randomUUID(), local: true, name: file.name, status: "error", error: "larger than 50 MB" });
      continue;
    }
    try {
      const res = await fetch(`/api/upload?name=${encodeURIComponent(file.name)}`, { method: "POST", body: file });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      if (data.status === "duplicate") {
        localErrors.push({ id: crypto.randomUUID(), local: true, name: file.name, status: "error", error: "already in the library" });
      }
    } catch (err) {
      localErrors.push({ id: crypto.randomUUID(), local: true, name: file.name, status: "error", error: err.message });
    }
  }
  refreshLibrary();
}

const drop = $("drop");
$("fileInput").addEventListener("change", e => { uploadFiles([...e.target.files]); e.target.value = ""; });
["dragenter", "dragover"].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", e => uploadFiles([...e.dataTransfer.files]));

/* ---------- session report ---------- */

// One row per question this session; kept in sessionStorage so a refresh doesn't lose it (per-tab only).
const turns = (() => {
  try { return JSON.parse(sessionStorage.getItem("veritas-turns")) || []; } catch { return []; }
})();
const sessionPanel = $("session");
const sessionBtn = $("sessionBtn");

function logTurn(question, r, error) {
  turns.push(r ? {
    question, seconds: r.seconds ?? 0, cost: r.cached ? 0 : r.usage?.cost_usd ?? null,
    tokens: r.cached ? 0 : (r.usage ? r.usage.input_tokens + r.usage.output_tokens : null),
    grounded: r.grounded, rewrites: r.attempts, sources: r.sources.length, cached: !!r.cached,
    notFound: !r.sources.length || /couldn't find this/i.test(r.answer),
  } : { question, error });
  try { sessionStorage.setItem("veritas-turns", JSON.stringify(turns)); } catch { /* storage unavailable */ }
  renderSession();
}

const fmtCost = c => (c == null ? "–" : c === 0 ? "$0" : `$${c < 0.01 ? c.toFixed(4) : c.toFixed(3)}`);

function renderSession() {
  $("turnCount").textContent = turns.length;
  const ok = turns.filter(t => !t.error);
  const live = ok.filter(t => !t.cached);
  const checked = ok.filter(t => t.grounded != null);
  const total = ok.reduce((s, t) => s + (t.cost || 0), 0);
  const avg = live.length ? live.reduce((s, t) => s + t.seconds, 0) / live.length : 0;
  const stats = [
    ["Questions", turns.length],
    ["Avg time", live.length ? `${avg.toFixed(1)}s` : "–"],
    ["Slowest", live.length ? `${Math.max(...live.map(t => t.seconds)).toFixed(1)}s` : "–"],
    ["Total cost", fmtCost(total)],
    ["Grounded", checked.length ? `${checked.filter(t => t.grounded).length}/${checked.length}` : "–"],
    ["Cache hits", ok.filter(t => t.cached).length],
    ["Rewrites", ok.reduce((s, t) => s + (t.rewrites || 0), 0)],
    ["Errors", turns.length - ok.length],
  ];
  $("stats").replaceChildren(...stats.map(([k, v]) => {
    const box = el("div", "stat");
    box.append(el("span", "stat-v", String(v)), el("span", "stat-k", k));
    return box;
  }));
  $("turns").replaceChildren(...turns.map((t, i) => {
    const tr = el("tr", t.error ? "err" : "");
    const q = el("td", "q", t.question);
    q.title = t.error ? `${t.question}\n\nError: ${t.error}` : t.question;
    const cells = t.error
      ? [el("td", null, String(i + 1)), q, Object.assign(el("td", "err-cell", `Error: ${t.error}`), { colSpan: 6 })]
      : [el("td", null, String(i + 1)), q, el("td", null, t.cached ? "cached" : `${t.seconds.toFixed(1)}s`),
         el("td", null, fmtCost(t.cost)), el("td", null, t.tokens == null ? "–" : t.tokens.toLocaleString()),
         el("td", t.grounded == null ? "" : t.grounded ? "yes" : "no",
            t.grounded == null ? (t.notFound ? "not found" : "–") : t.grounded ? "✓" : "✗"),
         el("td", null, String(t.rewrites)), el("td", null, String(t.sources))];
    tr.append(...cells);
    return tr;
  }));
  $("turnsEmpty").hidden = turns.length > 0;
}

function toggleSession(open) {
  sessionPanel.hidden = !open;
  sessionBtn.setAttribute("aria-expanded", open);
  if (open) $("sessionClose").focus();
}
sessionBtn.addEventListener("click", () => toggleSession(sessionPanel.hidden));
$("sessionClose").addEventListener("click", () => toggleSession(false));
document.addEventListener("keydown", e => { if (e.key === "Escape" && !sessionPanel.hidden) toggleSession(false); });

$("csvBtn").addEventListener("click", () => {
  const cols = ["question", "seconds", "cost", "tokens", "grounded", "rewrites", "sources", "cached", "error"];
  const esc = v => `"${String(v ?? "").replaceAll('"', '""')}"`;
  const csv = [cols.join(","), ...turns.map(t => cols.map(c => esc(t[c])).join(","))].join("\n");
  const a = el("a");
  a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  a.download = `veritas-session-${new Date().toISOString().slice(0, 16).replace(":", "")}.csv`;
  a.click();
  URL.revokeObjectURL(a.href);
});

$("newSession").addEventListener("click", () => {
  if (turns.length && !confirm("Clear this session's chat and report?")) return;
  try { sessionStorage.removeItem("veritas-turns"); } catch { /* storage unavailable */ }
  location.reload(); // fresh chat, empty report; the library is server-side and stays
});

const libraryBtn = $("libraryBtn");
libraryBtn.addEventListener("click", () => {
  const open = document.body.classList.toggle("show-library");
  libraryBtn.setAttribute("aria-expanded", open);
});
document.addEventListener("click", e => {
  if (document.body.classList.contains("show-library") && !e.target.closest(".library, .library-btn")) {
    document.body.classList.remove("show-library");
    libraryBtn.setAttribute("aria-expanded", "false");
  }
});

/* ---------- chat ---------- */

function addThinking(selfCorrecting) {
  const msg = el("article", "msg bot thinking");
  const dots = el("span", "dots");
  dots.append(el("i"), el("i"), el("i"));
  msg.append(dots, el("span", null, selfCorrecting ? "Searching, answering, checking grounding…" : "Retrieving and answering…"));
  thread.append(msg);
  return msg;
}

function renderAnswer(data) {
  const msg = tpl.content.firstElementChild.cloneNode(true);
  msg.querySelector(".answer").textContent = data.answer;

  const meta = msg.querySelector(".meta");
  if (data.grounded != null) {
    meta.append(el("span", `badge ${data.grounded ? "good" : "warn"}`, data.grounded ? "Grounded" : "Not fully grounded"));
  }
  meta.append(el("span", "badge", data.attempts ? `${data.attempts} rewrite${data.attempts > 1 ? "s" : ""}` : "No rewrites"));
  if (data.cached) meta.append(el("span", "badge good", "Cached · no model calls"));
  else if (data.seconds != null) meta.append(el("span", "badge", `${data.seconds.toFixed(1)}s`));
  const u = data.usage;
  if (u && !data.cached) {
    const tokens = (u.input_tokens + u.output_tokens).toLocaleString();
    const cost = u.cost_usd == null ? "" : ` · $${u.cost_usd < 0.01 ? u.cost_usd.toFixed(4) : u.cost_usd.toFixed(3)}`;
    const badge = el("span", "badge", `${tokens} tokens${cost}`);
    badge.title = `${u.input_tokens.toLocaleString()} input + ${u.output_tokens.toLocaleString()} output (incl. reasoning); estimated from list prices`;
    meta.append(badge);
  }
  if (data.trace_url) {
    const badge = el("span", "badge");
    const link = el("a", null, "Langfuse trace ↗");
    link.href = data.trace_url;
    link.target = "_blank";
    link.rel = "noopener";
    badge.append(link);
    meta.append(badge);
  }

  const rewrites = msg.querySelector(".rewrites");
  if (data.queries.length) {
    rewrites.hidden = false;
    data.queries.forEach(q => {
      const li = el("li", null, "Rewrote query: ");
      li.append(el("code", null, q));
      rewrites.append(li);
    });
  }

  const sources = msg.querySelector(".sources");
  sources.querySelector("summary").textContent = `Sources (${data.sources.length})`;
  if (!data.sources.length) sources.hidden = true;
  const list = sources.querySelector(".source-list");
  data.sources.forEach(s => {
    const card = el("div", "source");
    const h = el("h3", null, `${s.source}${s.page ? ` · p.${s.page}` : ""} `);
    if (s.headings) h.append(el("span", null, s.headings));
    card.append(h, el("pre", null, s.text));
    list.append(card);
  });
  return msg;
}

const STEP_TEXT = {
  retrieve: e => (e.found > e.kept
    ? `Found ${e.found} passages, kept the best ${e.kept}`
    : `Found ${e.kept} passage${e.kept === 1 ? "" : "s"}`),
  rewrite: e => `Rewrote the search: “${e.query}”`,
  generate: e => (e.insufficient ? "Not in those passages, searching again…"
    : e.missing ? `Part of the answer is missing, searching for “${e.missing}”…`
    : "Checking the answer against the sources…"),
  check: e => (e.grounded ? "Answer is supported by the sources" : "Answer not fully supported, searching again…"),
};

// Read server-sent events from a fetch response body ("data: {...}" blocks separated by blank lines).
async function* events(res) {
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      if (block.startsWith("data: ")) yield JSON.parse(block.slice(6));
    }
  }
}

async function ask(question) {
  if (busy || !docCount) return;
  busy = true;
  updateComposer();
  $("empty")?.remove();
  thread.append(el("div", "msg user", question));
  const live = addThinking(selfCorrect.checked);
  const status = live.lastChild;
  let answer = null;
  scrollDown();
  try {
    const res = await fetch("/api/ask/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, self_correct: selfCorrect.checked }),
    });
    if (!res.ok) throw new Error((await res.json()).error || `HTTP ${res.status}`);
    let finished = false;
    for await (const e of events(res)) {
      if (e.type === "step") {
        status.textContent = STEP_TEXT[e.node]?.(e) ?? status.textContent;
        if (e.node === "rewrite" && answer) { answer.remove(); answer = null; } // drop an unsupported draft
      } else if (e.type === "token") {
        if (!answer) {
          answer = el("article", "msg bot answer draft");
          live.after(answer);
          status.textContent = "Writing the answer…";
        }
        answer.textContent += e.text;
        scrollDown();
      } else if (e.type === "done") {
        answer?.remove();
        live.replaceWith(renderAnswer(e.result));
        logTurn(question, e.result);
        finished = true;
      } else if (e.type === "error") {
        throw new Error(e.error);
      }
    }
    if (!finished) throw new Error("the connection closed before the answer finished");
  } catch (err) {
    answer?.remove();
    live.replaceWith(el("article", "msg bot error", `Something went wrong: ${err.message}`));
    logTurn(question, null, err.message);
  } finally {
    busy = false;
    refreshStats();
    updateComposer();
    input.focus();
    scrollDown();
  }
}

form.addEventListener("submit", e => {
  e.preventDefault();
  const q = input.value.trim();
  if (!q) return;
  input.value = "";
  input.style.height = "";
  ask(q);
});

input.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

input.addEventListener("input", () => {
  input.style.height = "";
  input.style.height = `${input.scrollHeight}px`;
});

refreshLibrary();
refreshStats();
renderSession();
