/**
 * Main dashboard controller — wires WebSocket, pipeline UI, SAM2 canvas,
 * config panel, log viewer, 3D preview, and stage-panel switching together.
 */

import { WsManager } from './ws.js';
import { PipelineUI } from './pipeline.js';
import { SAM2Canvas } from './sam2-canvas.js';
import { ConfigPanel } from './config-panel.js';
import { LogViewer } from './log-viewer.js';
import { PreviewPanel } from './preview.js';
import { StageController } from './stage-controller.js';
import { SAM2Verification } from './sam2-verification.js';
import { CameraOverlay } from './camera-overlay.js';

// ── Init modules ─────────────────────────────────────────────

const ws = new WsManager();
const pipelineUI = new PipelineUI();
const sam2 = new SAM2Canvas();
const config = new ConfigPanel();
const log = new LogViewer();
const preview = new PreviewPanel();
const stageCtrl = new StageController();
const sam2Verify = new SAM2Verification();
const cameraOverlay = new CameraOverlay();

const statusBadge = document.getElementById('status-badge');
const vramBadge = document.getElementById('vram-badge');

// ── Stage-activated event: lazy-init 3D ─────────────────────

document.addEventListener('stage-activated', async (e) => {
  const stage = e.detail.stage;

  // Config panel: show stage-specific params
  config.setActiveStage(stage);

  // For 3D stages (3-6), activate the renderer if scene is initialized
  if (stage >= 3 && stage <= 6) {
    if (preview._stages?.[stage]?.initialized) {
      preview.activateStage(stage);
    }
  }
});

// ── Config panel callbacks ───────────────────────────────────

config.onStart = async (cfg) => {
  try {
    config.setRunning(true);
    pipelineUI.reset();
    sam2Verify.hide();
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
  stageCtrl.activateStage(msg.stage);
  stageCtrl.setStageState(msg.stage, 'running');
  pipelineUI.stageStart(msg.stage);
  log.append('stdout', `\n=== Stage ${msg.stage}/6: ${msg.label} ===\n`);
});

ws.on('stage_complete', (msg) => {
  pipelineUI.stageComplete(msg.stage, msg.elapsed);
  if (msg.error) {
    pipelineUI.stageFailed(msg.stage);
    stageCtrl.setStageState(msg.stage, 'failed');
    return;
  }
  stageCtrl.setStageState(msg.stage, 'complete');

  // Auto-load results per stage
  if (msg.stage === 1) {
    // Show gallery placeholder hidden, load frames
    const empty = document.querySelector('#stage-panel-1 .stage-panel-empty');
    if (empty) empty.classList.add('hidden');
    preview.loadGallery();
  } else if (msg.stage === 3) {
    preview.loadPi3xResults(cameraOverlay);
  } else if (msg.stage >= 4 && msg.stage <= 6) {
    preview.loadStageResult(msg.stage);
  }
});

ws.on('sam2_ready', (msg) => {
  pipelineUI.stageInteractive(2);
  sam2.activate(msg.frame_count, msg.width, msg.height);
  sam2Verify.hide();
  stageCtrl.activateStage(2);
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

ws.on('sam2_verification_ready', (msg) => {
  sam2Verify.show(msg.frame_count);
  log.append('stdout', 'Mask propagation complete. Please verify the results.\n');
});

ws.on('pipeline_complete', (msg) => {
  config.setRunning(false);
  config.setActiveStage(null);
  setStatus('complete', `Done (${formatTime(msg.elapsed)})`);
  log.append('stdout', `\n=== Pipeline Complete! (${formatTime(msg.elapsed)}) ===\n`);
});

ws.on('pipeline_error', (msg) => {
  config.setRunning(false);
  config.setActiveStage(null);
  setStatus('error', 'Error');
  pipelineUI.stageFailed(msg.stage);
  stageCtrl.setStageState(msg.stage, 'failed');
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
