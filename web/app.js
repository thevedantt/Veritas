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

function renderJobs(jobs) {
  const list = $("jobs");
  list.replaceChildren();
  const shown = [...localErrors, ...jobs.filter(j => !dismissed.has(j.id))];
  shown.forEach(j => {
    const li = el("li", `job ${j.status === "error" ? "error" : ""}`);
    if (ACTIVE.has(j.status)) li.append(el("span", "spin"));
    li.append(el("span", "name", j.name));
    li.append(el("span", "state", j.status === "error" ? j.error : STATE_TEXT[j.status] ?? j.status));
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
  msg.append(dots, el("span", null, selfCorrecting ? "Retrieving, grading, checking grounding…" : "Retrieving and answering…"));
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

async function ask(question) {
  if (busy || !docCount) return;
  busy = true;
  updateComposer();
  $("empty")?.remove();
  thread.append(el("div", "msg user", question));
  const thinking = addThinking(selfCorrect.checked);
  scrollDown();
  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, self_correct: selfCorrect.checked }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    thinking.replaceWith(renderAnswer(data));
  } catch (err) {
    thinking.replaceWith(el("article", "msg bot error", `Something went wrong: ${err.message}`));
  } finally {
    busy = false;
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
