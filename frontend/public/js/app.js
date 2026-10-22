/* --------------------------------------------------------------------------
   Sample Chat Bot - UI logic (vanilla JS, no framework, no build step)
   -------------------------------------------------------------------------- */

(() => {
  'use strict';

  // ------------------------------------------------------------------ state
  const state = {
    sessionId: localStorage.getItem('scb.session') || null,
    history: [],
    events: [],
    nodeState: {},
    streaming: false,
    abort: null,
    lastTraceId: null,
    lastIntent: null,
    lastModel: null,
    lastAnswer: null,
    lastError: null,
    paused: false,
  };

  const $ = (id) => document.getElementById(id);
  const el = {
    chatLog: $('chatLog'),
    input: $('input'),
    send: $('send'),
    stop: $('stop'),
    sessionLabel: $('sessionLabel'),
    healthDot: $('healthDot'),
    healthText: $('healthText'),
    healthPill: $('healthPill'),
    clearSession: $('clearSession'),
    timeline: $('timeline'),
    traceMeta: $('traceMeta'),
    traceDetail: $('traceDetail'),
    flow: $('flow'),
    bpList: $('bpList'),
    reqBody: $('reqBody'),
    reqSummary: $('reqSummary'),
    refreshDebug: $('refreshDebug'),
    debugToggle: $('debugToggle'),
  };

  // ------------------------------------------------------------------ utils
  const newSessionId = () =>
    `web-${Math.random().toString(36).slice(2, 8)}-${Date.now().toString(36)}`;

  const WAITING_TEXT = 'Waiting for Gemini…';

  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function formatTime(seconds) {
    if (!seconds) return '-';
    return new Date(seconds * 1000).toLocaleTimeString();
  }

  async function api(path, options = {}) {
    const response = await fetch(path, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    });
    const text = await response.text();
    let body = null;
    try {
      body = text ? JSON.parse(text) : null;
    } catch {
      body = { raw: text };
    }
    if (!response.ok) {
      const detail = body && body.detail ? body.detail : `HTTP ${response.status}`;
      throw new Error(detail);
    }
    return body;
  }

  // ------------------------------------------------------------------ chat
  function addMessage(role, text, meta = '') {
    const wrapper = document.createElement('div');
    wrapper.className = `message ${role === 'user' ? 'user' : 'bot'}`;

    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = text;

    wrapper.appendChild(bubble);

    if (meta) {
      const metaLine = document.createElement('div');
      metaLine.className = 'meta-line';
      metaLine.textContent = meta;
      wrapper.appendChild(metaLine);
    }

    el.chatLog.appendChild(wrapper);
    el.chatLog.scrollTop = el.chatLog.scrollHeight;
    return bubble;
  }

  function autoGrow() {
    el.input.style.height = 'auto';
    el.input.style.height = `${Math.min(el.input.scrollHeight, 160)}px`;
  }

  async function send() {
    if (state.streaming) return;
    const message = el.input.value.trim();
    if (!message) return;

    state.history.push({ role: 'user', content: message });
    addMessage('user', message);
    el.input.value = '';
    autoGrow();

    const bubble = addMessage('bot', '');
    bubble.parentElement.classList.add('cursor');

    state.streaming = true;
    state.events = [];
    state.nodeState = {};
    state.paused = false;
    state.lastAnswer = null;
    state.lastError = null;
    state.lastIntent = null;
    state.lastModel = null;
    el.send.disabled = true;
    el.stop.hidden = false;
    el.timeline.innerHTML = '';
    renderFlow();

    const controller = new AbortController();
    state.abort = controller;

    // If Gemini is slow (or the key is rate limited) show a hint instead of
    // an apparently frozen bubble.
    const waitingTimer = setTimeout(() => {
      if (!bubble.textContent.trim()) {
        bubble.textContent = WAITING_TEXT;
        bubble.dataset.placeholder = '1';
      }
    }, 7000);

    try {
      const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({
          message,
          session_id: state.sessionId,
          history: state.history.slice(0, -1),
          debug: true,
        }),
      });

      if (!response.ok || !response.body) {
        const text = await response.text();
        throw new Error(`HTTP ${response.status}: ${text.slice(0, 200)}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let boundary = buffer.indexOf('\n\n');
        while (boundary !== -1) {
          const frame = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);
          handleFrame(frame);
          boundary = buffer.indexOf('\n\n');
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        bubble.textContent = `Error: ${err.message}`;
        bubble.parentElement.classList.remove('cursor');
      }
    } finally {
      clearTimeout(waitingTimer);
      state.streaming = false;
      state.abort = null;
      el.send.disabled = false;
      el.stop.hidden = true;
      bubble.parentElement.classList.remove('cursor');

      if (!bubble.textContent.trim() || bubble.dataset.placeholder) {
        // No tokens arrived (e.g. the model answered in one shot) - fall back
        // to the final answer carried by the `done` event.
        delete bubble.dataset.placeholder;
        bubble.textContent = state.lastError
          ? `Error: ${state.lastError}`
          : state.lastAnswer || '(no response)';
      }

      const metaBits = [];
      if (state.lastIntent) metaBits.push(`intent: ${state.lastIntent}`);
      if (state.lastModel) metaBits.push(`model: ${state.lastModel}`);
      if (state.lastTraceId) metaBits.push(`trace: ${state.lastTraceId}`);
      if (metaBits.length) {
        const metaLine = document.createElement('div');
        metaLine.className = 'meta-line';
        metaLine.textContent = metaBits.join('  |  ');
        bubble.parentElement.appendChild(metaLine);
      }

      state.history.push({ role: 'assistant', content: bubble.textContent });
      el.chatLog.scrollTop = el.chatLog.scrollHeight;

      await refreshRequests();
    }
  }

  /** Parse one SSE frame; returns the event name. */
  function handleFrame(frame) {
    let name = 'message';
    const dataLines = [];

    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) name = line.slice(6).trim();
      else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim());
    }

    let data = {};
    if (dataLines.length) {
      try {
        data = JSON.parse(dataLines.join('\n'));
      } catch {
        data = {};
      }
    }

    switch (name) {
      case 'token':
        // The `router` node answers directly when no tool is needed and the
        // `respond` node answers after a tool ran - both are reply text.
        if (!data.node || data.node === 'router' || data.node === 'respond') {
          appendToken(data.token);
        }
        break;
      case 'trace':
        if (el.debugToggle.checked) addEvent(data.event);
        break;
      case 'node':
        markNode(data.node, 'done');
        if (data.update && data.update.intent) state.lastIntent = data.update.intent;
        break;
      case 'done':
        state.lastAnswer = data.answer || '';
        finishTrace(data);
        break;
      case 'error':
        state.lastError = data.error || 'Unknown error';
        addEvent({
          node: 'error',
          phase: 'error',
          status: 'error',
          message: data.error,
          duration_ms: null,
          started_at: Date.now() / 1000,
        });
        break;
      default:
        break;
    }
    return name;
  }

  function appendToken(token) {
    const last = el.chatLog.lastElementChild;
    if (!last || !last.classList.contains('bot')) return;
    const bubble = last.querySelector('.bubble');
    if (!bubble) return;
    // Replace the "waiting" placeholder with the first real token.
    if (bubble.dataset.placeholder) {
      bubble.textContent = '';
      delete bubble.dataset.placeholder;
    }
    bubble.textContent += token;
    el.chatLog.scrollTop = el.chatLog.scrollHeight;
  }

  // ----------------------------------------------------------------- trace
  function addEvent(event) {
    if (!event) return;
    state.events.push(event);

    if (event.phase === 'enter') {
      markNode(event.node, 'run');
    }
    if (event.phase === 'exit') {
      markNode(event.node, event.status === 'error' ? 'err' : 'done');
    }
    if (event.phase === 'error') {
      markNode(event.node, 'err');
    }

    const row = document.createElement('div');
    row.className = 'event';

    const status = event.status || 'ok';
    const duration =
      event.duration_ms != null ? `${Number(event.duration_ms).toFixed(1)}ms` : '';

    row.innerHTML = `
      <div class="event-head">
        <span class="event-node">${escapeHtml(event.node)}</span>
        <span class="event-phase">${escapeHtml(event.phase)}</span>
        <span class="badge ${escapeHtml(status)}">${escapeHtml(status)}</span>
        <span class="event-time">${duration}</span>
      </div>
      ${event.message ? `<div class="event-msg">${escapeHtml(event.message)}</div>` : ''}
      ${event.error ? `<div class="event-msg error">${escapeHtml(event.error)}</div>` : ''}
    `;

    const preview = event.output_preview || event.input_preview;
    if (preview && Object.keys(preview).length) {
      const details = document.createElement('details');
      details.className = 'event-detail';
      const summary = document.createElement('summary');
      summary.textContent = 'state snapshot';
      const pre = document.createElement('pre');
      pre.className = 'json';
      pre.textContent = JSON.stringify(preview, null, 2);
      details.appendChild(summary);
      details.appendChild(pre);
      row.appendChild(details);
    }

    el.timeline.appendChild(row);
    el.timeline.scrollTop = el.timeline.scrollHeight;
  }

  function finishTrace(data) {
    const trace = data.trace || {};
    state.lastTraceId = data.trace_id;
    state.lastModel = data.active_model || state.lastModel;
    state.paused = trace.status === 'paused';

    const meta = [
      `trace ${data.trace_id || '-'}`,
      `session ${data.session_id || '-'}`,
      `status ${trace.status || '-'}`,
      data.active_model ? `model ${data.active_model}` : null,
      trace.duration_ms != null ? `${trace.duration_ms}ms` : null,
      trace.error ? `error: ${trace.error}` : null,
    ]
      .filter(Boolean)
      .join('  |  ');

    el.traceMeta.innerHTML = `<span>${escapeHtml(meta)}</span>`;

    if (state.paused) {
      showResumeBar(trace);
    } else {
      el.traceDetail.innerHTML = '';
    }

    renderFlow();
  }

  function showResumeBar(trace) {
    el.traceDetail.innerHTML = `
      <div class="resume-bar">
        <button class="ghost small" id="resumeRun">Resume (run node)</button>
        <button class="ghost small" id="resumeSkip">Resume (skip node)</button>
        <span class="muted" style="font-size:12px">graph paused at a breakpoint</span>
      </div>
    `;
    $('resumeRun').onclick = () => resume(false);
    $('resumeSkip').onclick = () => resume(true);
  }

  async function resume(skip) {
    try {
      const result = await api('/api/chat/resume', {
        method: 'POST',
        body: JSON.stringify({ session_id: state.sessionId, skip }),
      });
      addMessage('bot', result.answer, `resumed - trace ${result.trace_id}`);
      if (result.debug) {
        result.debug.events.forEach((event) => addEvent(event));
      }
      el.traceDetail.innerHTML = '';
      await refreshRequests();
    } catch (err) {
      el.traceDetail.innerHTML = `<div class="event-msg error">Resume failed: ${escapeHtml(err.message)}</div>`;
    }
  }

  // ----------------------------------------------------------------- flow
  function renderFlow() {
    api('/api/debug/graph')
      .then((data) => {
        const nodes = data.nodes || [];
        const edges = data.edges || [];

        el.flow.innerHTML = '';

        const ordered = ['ingest', 'router', 'act', 'respond', 'finalize'];
        const sorted = ordered
          .map((id) => nodes.find((node) => node.id === id))
          .filter(Boolean);

        sorted.forEach((node, index) => {
          const status = state.nodeState[node.id] || 'idle';
          const row = document.createElement('div');
          row.className = 'flow-node';
          row.innerHTML = `
            <span class="state ${status}"></span>
            <span class="name">${escapeHtml(node.id)}</span>
            <span class="desc">${escapeHtml(node.description || '')}</span>
          `;
          el.flow.appendChild(row);

          const next = edges.find((edge) => edge.from === node.id);
          if (next) {
            const arrow = document.createElement('div');
            arrow.className = 'flow-arrow';
            const label = next.label ? ` <span class="flow-cond">${escapeHtml(next.label)}</span>` : '';
            arrow.innerHTML = `&#9662;${label}`;
            el.flow.appendChild(arrow);
          }
        });
      })
      .catch(() => {
        el.flow.innerHTML = '<span class="muted">Could not load the graph structure.</span>';
      });
  }

  function markNode(node, status) {
    if (!node) return;
    state.nodeState[node] = status;
    const rows = el.flow.querySelectorAll('.flow-node');
    rows.forEach((row) => {
      const name = row.querySelector('.name');
      const swatch = row.querySelector('.state');
      if (name && name.textContent === node) {
        swatch.className = `state ${status}`;
      }
    });
  }

  // ----------------------------------------------------------- breakpoints
  async function loadBreakpoints() {
    try {
      const data = await api('/api/debug/breakpoints');
      const breakpoints = data.breakpoints || {};
      const nodes = ['ingest', 'router', 'act', 'respond', 'finalize'];

      el.bpList.innerHTML = '';
      nodes.forEach((node) => {
        const mode = breakpoints[node] || 'off';
        const item = document.createElement('div');
        item.className = 'bp-item';
        item.innerHTML = `
          <span class="name">${escapeHtml(node)}</span>
          <span class="seg">
            <button data-node="${node}" data-mode="off" class="${mode === 'off' ? 'on' : ''}">off</button>
            <button data-node="${node}" data-mode="break" class="${mode === 'break' ? 'on' : ''}">break</button>
            <button data-node="${node}" data-mode="skip" class="${mode === 'skip' ? 'on' : ''}">skip</button>
          </span>
        `;
        el.bpList.appendChild(item);
      });

      el.bpList.querySelectorAll('.seg button').forEach((button) => {
        button.onclick = async () => {
          const { node, mode } = button.dataset;
          await api(`/api/debug/breakpoints/${node}`, {
            method: 'PUT',
            body: JSON.stringify({ mode }),
          });
          loadBreakpoints();
        };
      });
    } catch (err) {
      el.bpList.innerHTML = `<span class="muted">${escapeHtml(err.message)}</span>`;
    }
  }

  // -------------------------------------------------------------- requests
  async function refreshRequests() {
    try {
      const data = await api('/api/debug/requests?limit=40');
      const summary = data.summary || {};
      el.reqSummary.innerHTML = `
        <span>total ${summary.total || 0}</span>
        <span>errors ${summary.errors || 0}</span>
        <span>avg ${summary.avg_duration_ms || 0}ms</span>
        <span>max ${summary.max_duration_ms || 0}ms</span>
      `;

      el.reqBody.innerHTML = (data.requests || [])
        .map(
          (record) => `
            <tr>
              <td>${escapeHtml(record.method)}</td>
              <td>${escapeHtml(record.path)}</td>
              <td class="${String(record.status_code).startsWith('2') ? 'status-2xx' : 'status-5xx'}">
                ${record.status_code}
              </td>
              <td>${record.duration_ms != null ? `${record.duration_ms}ms` : '-'}</td>
            </tr>
          `
        )
        .join('');
    } catch {
      /* the backend may not be up yet */
    }
  }

  // --------------------------------------------------------------- health
  async function checkHealth() {
    try {
      const data = await api('/__health');
      const backend = data.backend || {};
      const reachable = backend.reachable;
      const validation = backend.body && backend.body.model_validation;
      const configured = backend.body && backend.body.ai && backend.body.ai.client;
      const modelBad = validation && validation.exists === false;

      el.healthDot.className = `dot ${reachable && !modelBad ? 'ok' : 'bad'}`;

      if (!reachable) {
        el.healthText.textContent = 'backend offline';
      } else if (modelBad) {
        const suggestion = validation.suggestion ? ` - try ${validation.suggestion}` : '';
        el.healthText.textContent = `model ${validation.model} unavailable${suggestion}`;
      } else if (configured && !configured.configured) {
        el.healthText.textContent = 'backend up - no API key';
      } else {
        const cooldowns = configured.cooldowns || {};
        const cooling = Object.keys(cooldowns).length;
        el.healthText.textContent = cooling
          ? `backend up - ${cooling} model(s) rate limited`
          : `backend up - ${configured.model}`;
      }
    } catch {
      el.healthDot.className = 'dot bad';
      el.healthText.textContent = 'backend offline';
    }
  }

  // ------------------------------------------------------------------ tabs
  function setupTabs() {
    document.querySelectorAll('.tab').forEach((tab) => {
      tab.onclick = () => {
        document.querySelectorAll('.tab').forEach((other) => other.classList.remove('active'));
        document.querySelectorAll('.tab-body').forEach((body) => body.classList.remove('active'));
        tab.classList.add('active');
        $(`tab-${tab.dataset.tab}`).classList.add('active');

        if (tab.dataset.tab === 'requests') refreshRequests();
        if (tab.dataset.tab === 'breakpoints') loadBreakpoints();
        if (tab.dataset.tab === 'graph') renderFlow();
      };
    });
  }

  // ------------------------------------------------------------ live reload
  function setupLiveReload() {
    if (!window.EventSource) return;
    const source = new EventSource('/__reload');
    source.addEventListener('reload', () => window.location.reload());
    source.onerror = () => {
      /* the server restarts often; EventSource retries on its own */
    };
  }

  // ------------------------------------------------------------------ init
  function init() {
    state.sessionId = state.sessionId || newSessionId();
    localStorage.setItem('scb.session', state.sessionId);
    el.sessionLabel.textContent = `session: ${state.sessionId}`;

    setupTabs();
    setupLiveReload();

    el.send.onclick = send;
    el.stop.onclick = () => state.abort && state.abort.abort();
    el.input.addEventListener('input', autoGrow);
    el.input.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        send();
      }
    });

    el.clearSession.onclick = () => {
      state.sessionId = newSessionId();
      localStorage.setItem('scb.session', state.sessionId);
      el.sessionLabel.textContent = `session: ${state.sessionId}`;
      state.history = [];
      el.chatLog.innerHTML = '';
      addMessage('bot', 'New session started. The conversation memory was cleared.');
    };

    el.refreshDebug.onclick = () => {
      refreshRequests();
      loadBreakpoints();
      renderFlow();
    };

    checkHealth();
    setInterval(checkHealth, 15000);
    renderFlow();
    loadBreakpoints();
    refreshRequests();
  }

  document.addEventListener('DOMContentLoaded', init);
})();
