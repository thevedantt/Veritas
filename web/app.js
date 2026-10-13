const thread = document.getElementById("thread");
const form = document.getElementById("form");
const input = document.getElementById("question");
const send = document.getElementById("send");
const selfCorrect = document.getElementById("selfCorrect");
const tpl = document.getElementById("answerTpl");

const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text; // textContent only: model output is never parsed as HTML
  return node;
};

const scrollDown = () => window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });

function addThinking(selfCorrecting) {
  const msg = el("article", "msg bot thinking");
  const dots = el("span", "dots");
  dots.append(el("i"), el("i"), el("i"));
  msg.append(dots, el("span", null, selfCorrecting
    ? "Retrieving, grading, checking grounding…"
    : "Retrieving and answering…"));
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
    const h = el("h3", null, `${s.source} · p.${s.page ?? "?"} `);
    if (s.headings) h.append(el("span", null, s.headings));
    card.append(h, el("pre", null, s.text));
    list.append(card);
  });
  return msg;
}

async function ask(question) {
  if (send.disabled) return;
  document.getElementById("empty")?.remove();
  thread.append(el("div", "msg user", question));
  const thinking = addThinking(selfCorrect.checked);
  send.disabled = true;
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
    send.disabled = false;
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

document.querySelectorAll(".chip").forEach(chip => chip.addEventListener("click", () => ask(chip.textContent)));
