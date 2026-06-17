/* =====================================================
   ConvoRAG – Frontend App Logic
   ===================================================== */

let API = localStorage.getItem("CONVORAG_API_URL") || "";

// ── DOM refs ──────────────────────────────────────────
const chatWindow     = document.getElementById("chat-window");
const chatInput      = document.getElementById("chat-input");
const sendBtn        = document.getElementById("send-btn");
const welcomeCard    = document.getElementById("welcome-card");
const personaBtn     = document.getElementById("persona-btn");
const closePanel     = document.getElementById("close-panel");
const personaPanel   = document.getElementById("persona-panel");
const panelOverlay   = document.getElementById("panel-overlay");
const panelContent   = document.getElementById("panel-content");
const topicTimeline  = document.getElementById("topic-timeline");
const sidebarToggle  = document.getElementById("sidebar-toggle");
const sidebar        = document.getElementById("sidebar");
const statusDot      = document.getElementById("status-dot");

// Settings DOM refs
const settingsBtn     = document.getElementById("settings-btn");
const settingsPanel   = document.getElementById("settings-panel");
const closeSettings   = document.getElementById("close-settings");
const apiUrlInput     = document.getElementById("api-url-input");
const saveSettingsBtn = document.getElementById("save-settings-btn");

// ── State ─────────────────────────────────────────────
let isLoading = false;

// =====================================================
// Init
// =====================================================
async function init() {
  statusDot.className = "status-dot loading";
  statusDot.title = "Connecting...";
  
  // Load input value
  apiUrlInput.value = localStorage.getItem("CONVORAG_API_URL") || "";
  
  try {
    await Promise.all([loadStats(), loadTimeline()]);
    statusDot.className = "status-dot online";
    statusDot.title = "System ready";
  } catch (err) {
    statusDot.className = "status-dot";
    statusDot.title = "Connection failed";
  }
}

document.addEventListener("DOMContentLoaded", init);

// =====================================================
// Stats
// =====================================================
async function loadStats() {
  try {
    const res  = await fetch(`${API}/api/stats`);
    const data = await res.json();
    document.getElementById("stat-messages").textContent    = fmtNum(data.total_messages);
    document.getElementById("stat-topics").textContent      = fmtNum(data.topic_segments);
    document.getElementById("stat-checkpoints").textContent = fmtNum(data.hundred_checkpoints);
  } catch (e) { console.error("Stats load failed", e); }
}

// =====================================================
// Topic Timeline
// =====================================================
async function loadTimeline() {
  try {
    const res  = await fetch(`${API}/api/checkpoints`);
    const data = await res.json();
    const cps  = data.topic_checkpoints || [];

    if (!cps.length) {
      topicTimeline.innerHTML = "<div class='timeline-loading'>No topic data yet. Run build_index.py first.</div>";
      return;
    }

    topicTimeline.innerHTML = cps.map((cp, i) => `
      <div class="timeline-item" data-id="${cp.id}" title="${escHtml(cp.summary)}">
        <div class="ti-label">Topic ${i + 1}</div>
        <div class="ti-count">${fmtNum(cp.msg_count)} msgs · ${cp.label.split('(')[1]?.replace(')','') || ''}</div>
        <div class="ti-keywords">${(cp.keywords || []).slice(0,5).join(' · ')}</div>
      </div>
    `).join("");

    // Click → ask about that topic
    topicTimeline.querySelectorAll(".timeline-item").forEach(el => {
      el.addEventListener("click", () => {
        const lbl = el.querySelector(".ti-label").textContent;
        const kws = el.querySelector(".ti-keywords").textContent;
        submitQuery(`Tell me about the conversations in "${lbl}". Keywords: ${kws}`);
      });
    });
  } catch (e) {
    topicTimeline.innerHTML = "<div class='timeline-loading'>Timeline unavailable.</div>";
  }
}

// =====================================================
// Panels (Persona & Settings)
// =====================================================
personaBtn.addEventListener("click", openPersonaPanel);
closePanel.addEventListener("click", closePersonaPanel);

settingsBtn.addEventListener("click", openSettingsPanel);
closeSettings.addEventListener("click", closeSettingsPanel);

panelOverlay.addEventListener("click", () => {
  closePersonaPanel();
  closeSettingsPanel();
});

saveSettingsBtn.addEventListener("click", saveSettings);

function openPersonaPanel() {
  personaPanel.classList.add("open");
  panelOverlay.classList.add("active");
  loadPersona();
}

function closePersonaPanel() {
  personaPanel.classList.remove("open");
  panelOverlay.classList.remove("active");
}

function openSettingsPanel() {
  settingsPanel.classList.add("open");
  panelOverlay.classList.add("active");
}

function closeSettingsPanel() {
  settingsPanel.classList.remove("open");
  panelOverlay.classList.remove("active");
}

async function saveSettings() {
  const url = apiUrlInput.value.trim().replace(/\/$/, ""); // trim trailing slash
  localStorage.setItem("CONVORAG_API_URL", url);
  API = url;
  
  // Clear cached persona so it reloads from new API
  delete panelContent.dataset.loaded;
  
  closeSettingsPanel();
  await init();
}

async function loadPersona() {
  if (panelContent.dataset.loaded === "true") return;
  panelContent.innerHTML = "<div class='panel-loading'>Fetching persona…</div>";
  try {
    const res     = await fetch(`${API}/api/persona`);
    const persona = await res.json();
    renderPersona(persona);
    panelContent.dataset.loaded = "true";
  } catch (e) {
    panelContent.innerHTML = "<div class='panel-loading'>Failed to load persona.</div>";
  }
}

function renderPersona(p) {
  const sections = [];

  // Personality traits
  if (p.personality_traits?.length) {
    sections.push(personaSection("✨ Personality Traits",
      chipList(p.personality_traits, true)));
  }

  // Habits
  if (p.habits?.length) {
    sections.push(personaSection("🔄 Habits",
      chipList(p.habits)));
  }

  // Facts
  const facts = p.personal_facts || {};
  if (facts.mentioned_occupations?.length) {
    sections.push(personaSection("💼 Occupations Mentioned",
      chipList(facts.mentioned_occupations)));
  }
  if (facts.hobbies?.length) {
    sections.push(personaSection("🎯 Hobbies",
      chipList(facts.hobbies)));
  }
  if (facts.pets?.length) {
    sections.push(personaSection("🐾 Pets",
      chipList(facts.pets)));
  }
  if (facts.family?.length) {
    sections.push(personaSection("👨‍👩‍👧 Family Signals",
      chipList(facts.family.map(f => f.replace(/_/g," ")))));
  }
  if (facts.mentioned_locations?.length) {
    sections.push(personaSection("📍 Locations",
      chipList(facts.mentioned_locations)));
  }

  // Communication style
  const style = p.communication_style || {};
  if (Object.keys(style).length) {
    const rows = [
      ["Tone",          style.tone            || "—"],
      ["Avg msg length",`${style.avg_message_length_words || "—"} words`],
      ["Emoji usage",   style.emoji_usage     || "—"],
      ["Question ratio",`${Math.round((style.question_ratio || 0) * 100)}%`],
      ["Exclamation !", `${Math.round((style.exclamation_ratio || 0) * 100)}%`],
    ];
    sections.push(personaSection("💬 Communication Style", kvTable(rows)));
  }

  // Topics of interest
  if (p.topics_of_interest?.length) {
    sections.push(personaSection("🔍 Top Keywords",
      chipList(p.topics_of_interest.slice(0, 12))));
  }

  panelContent.innerHTML = sections.join("") || "<p class='panel-loading'>No persona data yet.</p>";
}

function personaSection(title, inner) {
  return `<div class="persona-section">
    <div class="persona-section-title">${title}</div>
    ${inner}
  </div>`;
}

function chipList(items, highlight = false) {
  return `<div class="persona-chips">
    ${items.map(i => `<span class="persona-chip ${highlight ? "highlight" : ""}">${escHtml(i)}</span>`).join("")}
  </div>`;
}

function kvTable(rows) {
  return `<div class="persona-kv">
    ${rows.map(([k, v]) => `
      <div class="persona-kv-row">
        <span class="persona-kv-key">${escHtml(k)}</span>
        <span class="persona-kv-val">${escHtml(String(v))}</span>
      </div>`).join("")}
  </div>`;
}

// =====================================================
// Sidebar toggle
// =====================================================
sidebarToggle.addEventListener("click", () => {
  sidebar.classList.toggle("collapsed");
});

// =====================================================
// Quick query buttons
// =====================================================
document.querySelectorAll(".quick-btn").forEach(btn => {
  btn.addEventListener("click", () => submitQuery(btn.dataset.query));
});
document.querySelectorAll(".chip").forEach(chip => {
  chip.addEventListener("click", () => submitQuery(chip.dataset.query));
});

// =====================================================
// Input handlers
// =====================================================
chatInput.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
});
chatInput.addEventListener("input", autoResize);
sendBtn.addEventListener("click", handleSend);

function autoResize() {
  chatInput.style.height = "auto";
  chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + "px";
}

function handleSend() {
  const q = chatInput.value.trim();
  if (!q || isLoading) return;
  chatInput.value = "";
  autoResize();
  submitQuery(q);
}

// =====================================================
// Core Chat
// =====================================================
async function submitQuery(query) {
  if (isLoading) return;
  isLoading = true;

  // Hide welcome card
  if (welcomeCard) welcomeCard.style.display = "none";

  // User bubble
  appendMsg("user", escHtml(query));

  // Thinking bubble
  const thinkingId = "thinking-" + Date.now();
  appendThinking(thinkingId);

  // Status
  statusDot.className = "status-dot loading";

  try {
    const res  = await fetch(`${API}/api/chat`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ query }),
    });
    const data = await res.json();

    removeEl(thinkingId);

    if (data.error) {
      appendMsg("bot", `⚠️ Error: ${escHtml(data.error)}`);
    } else {
      appendBotMsg(data.answer, data.sources);
    }
    statusDot.className = "status-dot online";

  } catch (err) {
    removeEl(thinkingId);
    appendMsg("bot", "⚠️ Network error. Is the server running?");
    statusDot.className = "status-dot";
  }

  isLoading = false;
  scrollBottom();
}

// =====================================================
// Rendering helpers
// =====================================================
function appendMsg(role, html) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  wrap.innerHTML = `
    <div class="msg-avatar">${role === "user" ? "👤" : "🧠"}</div>
    <div class="msg-bubble">${html}</div>
  `;
  chatWindow.appendChild(wrap);
  scrollBottom();
}

function appendBotMsg(rawText, sources) {
  const html    = markdownToHtml(rawText);
  const srcHtml = buildSourcesBadge(sources);

  const wrap = document.createElement("div");
  wrap.className = "msg bot";
  wrap.innerHTML = `
    <div class="msg-avatar">🧠</div>
    <div class="msg-bubble">
      ${html}
      ${srcHtml}
    </div>
  `;
  chatWindow.appendChild(wrap);
  scrollBottom();
}

function buildSourcesBadge(sources) {
  if (!sources) return "";
  const field = sources.persona_field;
  if (field) {
    const label = field === "full" ? "Full persona" : field.replace(".", " › ").replace(/_/g, " ");
    return `<div class="sources-badge">🧬 Source: ${escHtml(label)}</div>`;
  }
  const cpCount  = (sources.checkpoints  || []).length;
  const msgCount = (sources.messages     || []).length;
  if (!cpCount && !msgCount) return "";
  return `<div class="sources-badge">📚 ${cpCount} checkpoint${cpCount !== 1 ? "s" : ""} · ${msgCount} messages</div>`;
}

function appendThinking(id) {
  const wrap = document.createElement("div");
  wrap.className = "msg bot";
  wrap.id = id;
  wrap.innerHTML = `
    <div class="msg-avatar">🧠</div>
    <div class="msg-bubble">
      <div class="thinking"><span></span><span></span><span></span></div>
    </div>
  `;
  chatWindow.appendChild(wrap);
  scrollBottom();
}

// Very lightweight markdown → HTML (bold, bullets, newlines)
function markdownToHtml(text) {
  return text
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")   // escape first
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")                      // **bold**
    .replace(/\*(.+?)\*/g,     "<em>$1</em>")                              // *italic*
    .replace(/^•\s(.+)$/gm,    "<li>$1</li>")                              // • bullets
    .replace(/(<li>[\s\S]*?<\/li>)+/g, "<ul>$&</ul>")                     // wrap <ul>
    .replace(/\n\n+/g, "<br><br>")                                         // double newline
    .replace(/\n/g,    "<br>");                                             // single newline
}

// =====================================================
// Utilities
// =====================================================
function scrollBottom() {
  requestAnimationFrame(() => {
    chatWindow.scrollTop = chatWindow.scrollHeight;
  });
}

function removeEl(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function fmtNum(n) {
  if (n == null || n === 0) return "–";
  return Number(n).toLocaleString();
}
