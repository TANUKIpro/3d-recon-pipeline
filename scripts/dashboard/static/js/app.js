/**
 * Main dashboard controller — wires WebSocket, pipeline UI, SAM2 canvas,
 * config panel, log viewer, and 3D preview together.
 */

import { WsManager } from './ws.js';
import { PipelineUI } from './pipeline.js';
import { SAM2Canvas } from './sam2-canvas.js';
import { ConfigPanel } from './config-panel.js';
import { LogViewer } from './log-viewer.js';
import { PreviewPanel } from './preview.js';

// ── Init modules ─────────────────────────────────────────────

const ws = new WsManager();
const pipelineUI = new PipelineUI();
const sam2 = new SAM2Canvas();
const config = new ConfigPanel();
const log = new LogViewer();
const preview = new PreviewPanel();

const statusBadge = document.getElementById('status-badge');
const vramBadge = document.getElementById('vram-badge');

// ── Tab switching ────────────────────────────────────────────

const tabs = document.querySelectorAll('#viewport-tabs .tab');
const tabContents = document.querySelectorAll('.tab-content');

tabs.forEach(tab => {
  tab.addEventListener('click', () => {
    const target = tab.dataset.tab;
    tabs.forEach(t => t.classList.remove('active'));
    tabContents.forEach(tc => tc.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(`tab-${target}`).classList.add('active');

    // Lazy-init 3D scene when tab is first shown
    if (target === 'preview3d') {
      preview.init3D();
      preview.refreshFileList();
    }
    if (target === 'gallery') {
      preview.loadGallery();
    }
  });
});

// ── Config panel callbacks ───────────────────────────────────

config.onStart = async (cfg) => {
  try {
    config.setRunning(true);
    pipelineUI.reset();
    setStatus('running', 'Running');

    const res = await fetch('/api/pipeline/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(cfg),
    });
    const data = await res.json();
    if (data.error) {
      alert('Start failed: ' + data.error);
      config.setRunning(false);
      setStatus('idle', 'Idle');
    }
  } catch (e) {
    alert('Start failed: ' + e.message);
    config.setRunning(false);
    setStatus('idle', 'Idle');
  }
};

config.onCancel = async () => {
  try {
    await fetch('/api/pipeline/cancel', { method: 'POST' });
  } catch (e) {
    console.error('Cancel error:', e);
  }
};

// ── WebSocket handlers ───────────────────────────────────────

ws.on('status', (msg) => {
  pipelineUI.updateAll(msg);
  if (msg.running) {
    config.setRunning(true);
    setStatus('running', 'Running');
  }
});

ws.on('stage_start', (msg) => {
  pipelineUI.stageStart(msg.stage);
  log.append('stdout', `\n=== Stage ${msg.stage}/6: ${msg.label} ===\n`);
});

ws.on('stage_complete', (msg) => {
  pipelineUI.stageComplete(msg.stage, msg.elapsed);
  if (msg.error) {
    pipelineUI.stageFailed(msg.stage);
  }
});

ws.on('sam2_ready', (msg) => {
  pipelineUI.stageInteractive(2);
  sam2.activate(msg.frame_count, msg.width, msg.height);
  // Auto-switch to SAM2 tab
  tabs.forEach(t => t.classList.remove('active'));
  tabContents.forEach(tc => tc.classList.remove('active'));
  document.querySelector('[data-tab="sam2"]').classList.add('active');
  document.getElementById('tab-sam2').classList.add('active');
  log.append('stdout', `SAM2 ready: ${msg.frame_count} frames (${msg.width}x${msg.height})\n`);
  log.append('stdout', 'Click on the object to segment. Right-click for negative points.\n');
});

ws.on('sam2_propagating', () => {
  sam2.deactivate();
  log.append('stdout', 'Propagating masks to all frames...\n');
});

ws.on('sam2_propagate_progress', (msg) => {
  log.append('stdout', `  Propagating frame ${msg.frame}/${msg.total}\n`);
});

ws.on('pipeline_complete', (msg) => {
  config.setRunning(false);
  setStatus('complete', `Done (${formatTime(msg.elapsed)})`);
  log.append('stdout', `\n=== Pipeline Complete! (${formatTime(msg.elapsed)}) ===\n`);
  // Refresh preview file list
  preview.refreshFileList();
  preview.loadGallery();
});

ws.on('pipeline_error', (msg) => {
  config.setRunning(false);
  setStatus('error', 'Error');
  pipelineUI.stageFailed(msg.stage);
  log.append('stderr', `\nPipeline error at stage ${msg.stage}: ${msg.error}\n`);
});

ws.on('log', (msg) => {
  log.append(msg.stream, msg.text);
});

ws.on('_open', () => {
  log.append('stdout', '[Dashboard connected]\n');
});

ws.on('_close', () => {
  log.append('stderr', '[Dashboard disconnected — reconnecting...]\n');
});

// ── VRAM polling ─────────────────────────────────────────────

async function pollVRAM() {
  try {
    const res = await fetch('/api/vram');
    const data = await res.json();
    if (data.free_mb != null) {
      vramBadge.textContent = `VRAM: ${data.free_mb} MB free`;
    } else {
      vramBadge.textContent = 'VRAM: N/A';
    }
  } catch (e) {
    vramBadge.textContent = 'VRAM: --';
  }
}

setInterval(pollVRAM, 5000);
pollVRAM();

// ── Helpers ──────────────────────────────────────────────────

function setStatus(state, text) {
  statusBadge.textContent = text;
  statusBadge.className = 'badge';
  if (state === 'running') statusBadge.classList.add('badge-running');
  else if (state === 'complete') statusBadge.classList.add('badge-complete');
  else if (state === 'error') statusBadge.classList.add('badge-error');
  else statusBadge.classList.add('badge-idle');
}

function formatTime(secs) {
  if (secs < 60) return `${Math.round(secs)}s`;
  const m = Math.floor(secs / 60);
  const s = Math.round(secs % 60);
  return `${m}m${s}s`;
}

// ── Boot ─────────────────────────────────────────────────────

ws.connect();
