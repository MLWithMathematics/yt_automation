/* dashboard/static/app.js — Fixed SSE, single run spawn, resume/retry */
"use strict";

const AGENTS = [
  "TrendAgent","KeywordAgent","ScriptAgent","VoiceoverAgent",
  "VisualAgent","VideoAgent","ThumbnailAgent","MetadataAgent"
];
const AGENT_ICONS = {
  TrendAgent:"📈", KeywordAgent:"🔍", ScriptAgent:"📝",
  VoiceoverAgent:"🎙️", VisualAgent:"🎨", VideoAgent:"🎬",
  ThumbnailAgent:"🖼️", MetadataAgent:"📋"
};
const AGENT_LABELS = {
  TrendAgent:"Trend", KeywordAgent:"Keyword", ScriptAgent:"Script",
  VoiceoverAgent:"Voiceover", VisualAgent:"Visual", VideoAgent:"Video",
  ThumbnailAgent:"Thumbnail", MetadataAgent:"Metadata"
};

// ── State ─────────────────────────────────────────────────────────────────────
let state = {
  activeRunId: null,
  runs: [], totalRuns: 0, page: 0, pageSize: 10,
  logFilter: "all", logAgentFilter: null, pinLogs: false,
  eventSource: null, globalSource: null,
  logs: [],          // all log entries for current active run
  completedCount: 0, // how many agents completed in active run
  currentConfig: {},
};

// ── Helpers ───────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const esc = s => String(s)
  .replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")
  .replace(/"/g,"&quot;");

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  buildPipelineFlow();
  populateAgentFilter();
  loadRuns();
  connectGlobalSSE();

  // Refresh run table every 15s
  setInterval(loadRuns, 15000);

  // Buttons
  $("btn-start").addEventListener("click", () => openModal("start-modal"));
  $("btn-config").addEventListener("click", () => openModal("config-modal"));
  $("pin-btn").addEventListener("click", togglePin);
  $("clear-log-btn").addEventListener("click", clearLogs);
  $("filter-all").addEventListener("click", () => setLogFilter("all"));
  $("filter-errors").addEventListener("click", () => setLogFilter("ERROR"));
  $("agent-filter-select").addEventListener("change", e => {
    state.logAgentFilter = e.target.value || null;
    redrawLogs();
  });
  $("btn-prev-page").addEventListener("click", () => changePage(-1));
  $("btn-next-page").addEventListener("click", () => changePage(1));
  $("start-modal-confirm").addEventListener("click", startNewRun);

  // Segmented control (video length)
  document.querySelectorAll(".seg-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#length-control .seg-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
    });
  });
});

// ── Modal helpers ─────────────────────────────────────────────────────────────
function openModal(id)  { $(id).classList.add("open"); }
function closeModal(id) { $(id).classList.remove("open"); }

// ── Pipeline Flowchart ────────────────────────────────────────────────────────
function buildPipelineFlow() {
  const container = $("pipeline-flow");
  container.innerHTML = "";
  AGENTS.forEach((name, i) => {
    const box = document.createElement("div");
    box.id = `agent-box-${name}`;
    box.className = "agent-box pending";
    box.innerHTML = `
      <div class="agent-status-dot"></div>
      <span class="agent-icon">${AGENT_ICONS[name]}</span>
      <span class="agent-name">${AGENT_LABELS[name]}</span>
      <span class="agent-dur" id="dur-${name}">—</span>
    `;
    box.addEventListener("click", () => setLogAgentFilter(name));
    container.appendChild(box);

    // Retry button (hidden by default, shown when failed)
    const retryWrap = document.createElement("div");
    retryWrap.className = "agent-retry-btn";
    retryWrap.innerHTML = `
      <button class="btn btn-sm btn-warning" onclick="retryAgent('${name}')">↺ Retry ${AGENT_LABELS[name]}</button>
    `;
    container.appendChild(retryWrap);

    if (i < AGENTS.length - 1) {
      const conn = document.createElement("div");
      conn.className = "agent-connector";
      container.appendChild(conn);
    }
  });
}

function populateAgentFilter() {
  const sel = $("agent-filter-select");
  AGENTS.forEach(name => {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = AGENT_LABELS[name];
    sel.appendChild(opt);
  });
}

function updateAgentBox(agentName, status, duration) {
  const box = $(`agent-box-${agentName}`);
  if (!box) return;
  box.className = `agent-box ${status.toLowerCase()}`;
  const dur = $(`dur-${agentName}`);
  if (dur && duration != null) dur.textContent = `${duration}s`;

  // Count completed for progress bar
  state.completedCount = AGENTS.filter(name => {
    const b = $(`agent-box-${name}`);
    return b && b.classList.contains("completed");
  }).length;
  updateProgressBar();
}

function updateProgressBar() {
  const pct = (state.completedCount / AGENTS.length) * 100;
  const fill = $("pipeline-progress-fill");
  const text = $("pipeline-progress-text");
  if (fill) fill.style.width = pct + "%";
  if (text) text.textContent = `${state.completedCount} / ${AGENTS.length}`;
}

function resetPipelineFlow() {
  AGENTS.forEach(name => {
    const box = $(`agent-box-${name}`);
    if (box) box.className = "agent-box pending";
    const dur = $(`dur-${name}`);
    if (dur) dur.textContent = "—";
  });
  state.completedCount = 0;
  updateProgressBar();
}

// ── SSE ───────────────────────────────────────────────────────────────────────
function connectRunSSE(runId) {
  if (state.eventSource) {
    state.eventSource.close();
    state.eventSource = null;
  }
  const es = new EventSource(`/api/sse/${runId}`);
  state.eventSource = es;
  es.addEventListener("message", e => {
    try { handleSSEEvent(JSON.parse(e.data)); } catch(err) {}
  });
  es.onerror = () => {};
}

function connectGlobalSSE() {
  if (state.globalSource) return;
  const es = new EventSource("/api/sse/global/stream");
  state.globalSource = es;
  es.addEventListener("message", e => {
    try { handleSSEEvent(JSON.parse(e.data)); } catch(err) {}
  });
}

function handleSSEEvent(evt) {
  const {type, data} = evt;
  if (!data) return;

  // Only process events for active run (or all if no active run)
  if (state.activeRunId && data.run_id && data.run_id !== state.activeRunId) return;

  switch (type) {
    case "agent_started":
      updateAgentBox(data.agent_name, "RUNNING", null);
      appendLog({
        run_id: data.run_id, agent_name: data.agent_name,
        log_level: "INFO",
        message: `▶ Started (attempt ${(data.attempt||0)+1})`,
        timestamp: new Date(data.timestamp * 1000).toISOString()
      });
      setActiveRunId(data.run_id);
      break;

    case "agent_completed":
      updateAgentBox(data.agent_name, "COMPLETED", data.duration);
      appendLog({
        run_id: data.run_id, agent_name: data.agent_name,
        log_level: "INFO",
        message: `✅ Done in ${data.duration}s`,
        timestamp: new Date().toISOString()
      });
      break;

    case "agent_failed":
      updateAgentBox(data.agent_name, "FAILED", null);
      appendLog({
        run_id: data.run_id, agent_name: data.agent_name,
        log_level: "ERROR",
        message: `❌ Failed (attempt ${data.retry_count}): ${data.error}`,
        timestamp: new Date().toISOString()
      });
      setRunStatus("FAILED");
      showResumeButton(data.run_id);
      toast("Agent failed: " + data.agent_name, "error");
      break;

    case "log_entry":
      appendLog(data);
      break;

    case "pipeline_done":
      appendLog({
        run_id: data.run_id, agent_name: "Pipeline",
        log_level: "INFO", message: "🎉 Pipeline completed!",
        timestamp: new Date().toISOString()
      });
      setRunStatus("COMPLETED");
      $("run-actions").style.display = "none";
      loadRunOutput(data.run_id);
      loadRuns();
      toast("Pipeline completed successfully!", "success");
      break;
  }
}

// ── Log Feed ──────────────────────────────────────────────────────────────────
function appendLog(entry) {
  state.logs.push(entry);
  if (state.logs.length > 3000) state.logs.shift();
  if (shouldShowLog(entry)) renderLogEntry(entry);
  updateLogCount();
}

function shouldShowLog(entry) {
  if (state.logFilter === "ERROR" && entry.log_level !== "ERROR") return false;
  if (state.logAgentFilter && entry.agent_name !== state.logAgentFilter) return false;
  return true;
}

function renderLogEntry(entry) {
  const feed = $("log-feed");
  // Remove empty state on first log
  const empty = feed.querySelector(".empty-state");
  if (empty) empty.remove();

  const div = document.createElement("div");
  div.className = `log-entry ${entry.log_level || "INFO"}`;
  const ts = entry.timestamp ? entry.timestamp.substring(11, 19) : "--:--:--";
  const agent = (entry.agent_name || "").replace("Agent","");
  div.innerHTML =
    `<span class="log-ts">${ts}</span>` +
    `<span class="log-agent" title="${esc(entry.agent_name||"")}">${esc(agent)}</span>` +
    `<span class="log-level ${entry.log_level||"INFO"}">${entry.log_level||"INFO"}</span>` +
    `<span class="log-msg">${esc(entry.message||"")}</span>`;
  feed.appendChild(div);
  if (!state.pinLogs) {
    feed.scrollTop = feed.scrollHeight;
  }
}

function redrawLogs() {
  const feed = $("log-feed");
  feed.innerHTML = "";
  const filtered = state.logs.filter(shouldShowLog);
  if (!filtered.length) {
    feed.innerHTML = `<div class="empty-state"><div class="empty-icon">📭</div><p>No matching log entries.</p></div>`;
  } else {
    filtered.forEach(renderLogEntry);
  }
  updateLogCount();
}

function updateLogCount() {
  const cnt = $("log-count");
  if (cnt) cnt.textContent = `${state.logs.length} entries`;
}

function setLogFilter(filter) {
  state.logFilter = filter;
  $("filter-all").classList.toggle("active", filter === "all");
  $("filter-errors").classList.toggle("active", filter === "ERROR");
  redrawLogs();
}

function setLogAgentFilter(agentName) {
  if (state.logAgentFilter === agentName) {
    state.logAgentFilter = null;
    $("agent-filter-select").value = "";
  } else {
    state.logAgentFilter = agentName;
    $("agent-filter-select").value = agentName;
  }
  redrawLogs();
}

function togglePin() {
  state.pinLogs = !state.pinLogs;
  $("pin-btn").classList.toggle("active", state.pinLogs);
  $("pin-btn").title = state.pinLogs ? "Unpin scroll" : "Pin scroll";
}

function clearLogs() {
  state.logs = [];
  $("log-feed").innerHTML = `<div class="empty-state"><div class="empty-icon">🚀</div><p>Logs cleared. Start a run to see new logs.</p></div>`;
  updateLogCount();
}

// ── Runs Table ────────────────────────────────────────────────────────────────
async function loadRuns() {
  try {
    const offset = state.page * state.pageSize;
    const res = await fetch(`/api/runs?limit=${state.pageSize}&offset=${offset}`);
    const data = await res.json();
    state.runs = data.runs || [];
    state.totalRuns = data.total || 0;
    renderRunsTable();
  } catch(e) {}
}

function renderRunsTable() {
  const tbody = $("runs-tbody");
  if (!state.runs.length) {
    tbody.innerHTML = `<tr><td colspan="7" class="table-empty">No runs yet. Click "Start New Run" to begin.</td></tr>`;
    return;
  }
  tbody.innerHTML = state.runs.map(run => {
    const created = run.created_at ? run.created_at.substring(0,16).replace("T"," ") : "—";
    const showResume = ["FAILED","PAUSED","RUNNING"].includes(run.status);
    return `<tr>
      <td class="mono-text" style="color:var(--accent);cursor:pointer" onclick="selectRun('${run.id}')">${run.id.substring(0,8)}</td>
      <td style="color:var(--text-dim)">${created}</td>
      <td>${esc(run.niche)}</td>
      <td style="color:var(--text-dim)" id="row-topic-${run.id}">—</td>
      <td><span class="status-badge ${run.status}">${run.status}</span></td>
      <td class="mono-text">—</td>
      <td>
        <div style="display:flex;gap:4px;align-items:center">
          <button class="btn btn-sm btn-ghost" onclick="selectRun('${run.id}')">View</button>
          ${showResume ? `<button class="btn btn-sm btn-warning" onclick="resumeRun('${run.id}')">↺ Resume</button>` : ""}
          <button class="icon-btn" onclick="deleteRun('${run.id}')" title="Delete run" style="width:24px;height:24px">
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/></svg>
          </button>
        </div>
      </td>
    </tr>`;
  }).join("");

  const total = state.totalRuns;
  const start = state.page * state.pageSize + 1;
  const end = Math.min((state.page + 1) * state.pageSize, total);
  $("page-info").textContent = total ? `${start}–${end} of ${total}` : "0 runs";
  $("btn-prev-page").disabled = state.page === 0;
  $("btn-next-page").disabled = (state.page + 1) * state.pageSize >= total;

  // Load topics for visible runs
  state.runs.forEach(run => {
    fetchRunTopic(run.id);
  });
}

async function fetchRunTopic(runId) {
  try {
    const res = await fetch(`/api/runs/${runId}/output`);
    const data = await res.json();
    const cell = $(`row-topic-${runId}`);
    if (cell && data.topic) cell.textContent = data.topic.substring(0, 30);
  } catch(e) {}
}

function changePage(dir) {
  const maxPage = Math.max(0, Math.ceil(state.totalRuns / state.pageSize) - 1);
  state.page = Math.max(0, Math.min(state.page + dir, maxPage));
  loadRuns();
}

// ── Run Actions ───────────────────────────────────────────────────────────────
async function selectRun(runId) {
  state.activeRunId = runId;
  state.logs = [];
  $("log-feed").innerHTML = "";
  resetPipelineFlow();
  setActiveRunId(runId);

  try {
    const res = await fetch(`/api/runs/${runId}`);
    const data = await res.json();

    setRunStatus(data.run.status);

    // Populate agent boxes from saved task data
    (data.tasks || []).forEach(task => {
      updateAgentBox(
        task.agent_name, task.status,
        task.duration_seconds ? task.duration_seconds.toFixed(1) : null
      );
    });

    // Show resume if needed
    if (["FAILED","PAUSED"].includes(data.run.status)) {
      showResumeButton(runId);
    } else {
      $("run-actions").style.display = "none";
    }

    // Load recent logs
    (data.recent_logs || []).forEach(log => appendLog(log));

    // Connect SSE
    connectRunSSE(runId);

    // Load outputs
    loadRunOutput(runId);
  } catch(e) {
    console.error("selectRun error:", e);
  }
}

function setActiveRunId(runId) {
  const el = $("active-run-id");
  if (el) el.textContent = runId ? runId.substring(0,8) + "…" : "None";
  if (runId) state.activeRunId = runId;
}

function setRunStatus(status) {
  const badge = $("run-status-badge");
  if (badge) {
    badge.className = `status-badge ${status}`;
    badge.textContent = status;
  }
}

function showResumeButton(runId) {
  const actDiv = $("run-actions");
  if (actDiv) {
    actDiv.style.display = "flex";
    const btn = $("btn-resume");
    if (btn) btn.onclick = () => resumeRun(runId);
  }
}

async function loadRunOutput(runId) {
  try {
    const res = await fetch(`/api/runs/${runId}/output`);
    const data = await res.json();

    const thumb = $("thumbnail-preview");
    const thumbPh = $("thumb-placeholder");
    if (data.thumbnail && thumb) {
      thumb.src = data.thumbnail + "?t=" + Date.now();
      thumb.style.display = "block";
      thumb.onerror = () => { thumb.style.display="none"; if(thumbPh) thumbPh.style.display="flex"; };
      if (thumbPh) thumbPh.style.display = "none";

      const btn = $("dl-thumbnail");
      if (btn) { btn.disabled = false; btn.onclick = () => downloadFile(data.thumbnail, "thumbnail.jpg"); }
    }

    const vid = $("video-preview");
    const vidPh = $("video-placeholder");
    if (data.video && vid) {
      vid.src = data.video + "?t=" + Date.now();
      vid.style.display = "block";
      if (vidPh) vidPh.style.display = "none";

      const btn = $("dl-video");
      if (btn) { btn.disabled = false; btn.onclick = () => downloadFile(data.video, "video.mp4"); }
    }

    if (data.metadata) {
      const titleEl = $("meta-title");
      if (titleEl) titleEl.textContent = data.metadata.title || "—";
      renderTags(data.metadata.tags || [], data.metadata.hashtags || []);

      const btn = $("dl-metadata");
      if (btn) { btn.disabled = false; btn.onclick = () => downloadJson(data.metadata, "metadata.json"); }
    }

    if (data.script) {
      const btn = $("dl-script");
      if (btn) { btn.disabled = false; btn.onclick = () => downloadJson(data.script, "script.json"); }
    }
  } catch(e) {}
}

function renderTags(tags, hashtags) {
  const container = $("tags-container");
  if (!container) return;
  container.innerHTML = [
    ...hashtags.map(h => `<span class="tag-chip hashtag-chip">${esc(h)}</span>`),
    ...tags.slice(0,10).map(t => `<span class="tag-chip">${esc(t)}</span>`),
  ].join("");
}

// ── Resume & Retry ────────────────────────────────────────────────────────────
async function resumeRun(runId) {
  try {
    const res = await fetch(`/api/runs/${runId}/resume`, { method: "POST" });
    if (!res.ok) { toast("Resume failed", "error"); return; }
    toast("Resuming pipeline…", "info");
    selectRun(runId);
  } catch(e) {
    toast("Resume request failed", "error");
  }
}

async function resumeActiveRun() {
  if (!state.activeRunId) return;
  await resumeRun(state.activeRunId);
}

async function retryAgent(agentName) {
  if (!state.activeRunId) {
    toast("No active run selected", "error");
    return;
  }
  try {
    const res = await fetch(
      `/api/runs/${state.activeRunId}/agents/${agentName}/retry`,
      { method: "POST" }
    );
    if (!res.ok) { toast(`Retry failed for ${agentName}`, "error"); return; }
    toast(`Retrying ${agentName}…`, "info");
    // Reset agent box visually
    updateAgentBox(agentName, "PENDING", null);
  } catch(e) {
    toast("Retry request failed", "error");
  }
}

async function deleteRun(runId) {
  if (!confirm(`Delete run ${runId.substring(0,8)}? This cannot be undone.`)) return;
  try {
    await fetch(`/api/runs/${runId}`, { method: "DELETE" });
    if (state.activeRunId === runId) {
      state.activeRunId = null;
      $("log-feed").innerHTML = `<div class="empty-state"><div class="empty-icon">🚀</div><p>Start a new run to see logs here.</p></div>`;
      resetPipelineFlow();
      setActiveRunId(null);
    }
    loadRuns();
    toast("Run deleted", "info");
  } catch(e) {
    toast("Delete failed", "error");
  }
}

// ── Start New Run ─────────────────────────────────────────────────────────────
async function startNewRun() {
  // Read form values
  const niche = $("start-niche").value.trim() || null;
  const activeLengthBtn = document.querySelector("#length-control .seg-btn.active");
  const videoLength = activeLengthBtn ? activeLengthBtn.dataset.value : null;
  const activeVoiceBtn = document.querySelector(".voice-btn.active");
  const voice = activeVoiceBtn ? activeVoiceBtn.dataset.value : null;

  closeModal("start-modal");
  resetPipelineFlow();
  state.logs = [];
  $("log-feed").innerHTML = "";

  // Build request body (JSON)
  const body = {};
  if (niche) body.niche = niche;
  if (videoLength) body.video_length = videoLength;
  if (voice) body.voice = voice;

  try {
    const res = await fetch("/api/runs/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      toast("Failed to start run: " + res.statusText, "error");
      return;
    }

    const data = await res.json();
    state.activeRunId = data.run_id;
    setActiveRunId(data.run_id);
    setRunStatus("RUNNING");
    $("run-actions").style.display = "none";
    connectRunSSE(data.run_id);
    toast(`Pipeline started! Niche: ${data.niche}`, "success");

    // Update niche badge
    const badge = $("niche-badge");
    if (badge) badge.textContent = data.niche;

    setTimeout(loadRuns, 800);
  } catch(e) {
    toast("Network error starting run", "error");
    console.error(e);
  }
}

// ── Config Save ───────────────────────────────────────────────────────────────
async function saveConfig() {
  const body = {};
  const niche = $("cfg-niche").value.trim();
  const length = $("cfg-length").value;
  const voice = $("cfg-voice").value;
  if (niche) body.niche = niche;
  if (length) body.video_length = length;
  if (voice) body.voice = voice;

  try {
    const res = await fetch("/api/config", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (res.ok) {
      const cfg = await res.json();
      const badge = $("niche-badge");
      if (badge) badge.textContent = cfg.niche;
      closeModal("config-modal");
      toast("Config updated for next run", "success");
    } else {
      toast("Config save failed", "error");
    }
  } catch(e) {
    toast("Network error saving config", "error");
  }
}

// ── Toast ─────────────────────────────────────────────────────────────────────
function toast(msg, type = "info") {
  const icons = { success:"✅", error:"❌", info:"ℹ️" };
  const container = $("toast-container");
  if (!container) return;

  const t = document.createElement("div");
  t.className = `toast ${type}`;
  t.innerHTML = `<span class="toast-icon">${icons[type]||"ℹ️"}</span><span class="toast-msg">${esc(msg)}</span>`;
  t.onclick = () => t.remove();
  container.appendChild(t);
  setTimeout(() => { if(t.parentNode) t.remove(); }, 4000);
}

// ── Downloads ─────────────────────────────────────────────────────────────────
function downloadFile(url, filename) {
  const a = document.createElement("a");
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click();
  document.body.removeChild(a);
}

function downloadJson(obj, filename) {
  const blob = new Blob([JSON.stringify(obj, null, 2)], {type:"application/json"});
  downloadFile(URL.createObjectURL(blob), filename);
}
