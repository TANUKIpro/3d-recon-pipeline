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
const overallProgressBadge = document.getElementById('overall-progress-badge');

let _extractedFrameCount = 0;
let _hydratedStatusKey = '';
let _objectLoadRequestId = 0;

// ── Stage-activated event: lazy-init 3D ─────────────────────

document.addEventListener('stage-activated', async (e) => {
  const stage = e.detail.stage;

  // Config panel: show stage-specific params
  config.setActiveStage(stage);

  // For 3D stages (2=Pi3X, 4-6), activate the renderer if scene is initialized
  if (stage === 2 || (stage >= 4 && stage <= 6)) {
    if (preview._stages?.[stage]?.initialized) {
      preview.activateStage(stage);
    }
  }
});

// Align initial config view with default stage tab.
stageCtrl.activateStage(stageCtrl.activeStage);

// ── Config panel callbacks ───────────────────────────────────

config.onObjectSelected = async (objectName) => {
  if (!objectName) {
    _hydratedStatusKey = '';
    cameraOverlay.remove();
    sam2.deactivate();
    sam2Verify.hide();
    preview.reset();
    pipelineUI.reset();
    hidePi3xApproveButton();
    setOverallProgress(0);
    setStatus('idle', 'Idle');
    stageCtrl.activateStage(1);
    return;
  }

  const reqId = ++_objectLoadRequestId;
  try {
    const res = await fetch('/api/pipeline/load-object', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: objectName }),
    });
    const data = await res.json();
    if (reqId !== _objectLoadRequestId) return;
    if (!res.ok || data.error) {
      log.append('stderr', `Failed to load object ${objectName}: ${data.error || res.status}\n`);
      return;
    }

    const status = data.pipeline_status;
    if (status) {
      await applyStatusSnapshot(status, { forceHydrate: true });
      log.append('stdout', `Loaded object: ${objectName}\n`);
    }
  } catch (e) {
    if (reqId !== _objectLoadRequestId) return;
    log.append('stderr', `Failed to load object ${objectName}: ${e.message}\n`);
  }
};

config.onStart = async (cfg) => {
  try {
    config.setRunning(true);
    setStatus('running', 'Running');

    const res = await fetch('/api/pipeline/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(cfg),
    });
    const data = await res.json();
    if (!res.ok || data.error) {
      alert('Start failed: ' + (data.error || res.status));
      config.setRunning(false);
      setStatus('idle', 'Idle');
      return;
    }

    const resumeFromStage = Math.max(1, Math.min(6, Number(data.resume_from_stage) || 1));

    preview.clearFromStage(resumeFromStage);
    cameraOverlay.remove();
    sam2.deactivate();
    sam2Verify.hide();
    hidePi3xApproveButton();
    pipelineUI.resetFromStage(resumeFromStage);
    _extractedFrameCount = 0;
    _hydratedStatusKey = '';
    setOverallProgress(pipelineUI.getOverallProgress());

    if (data.object_name) {
      config.setObjectName(data.object_name);
      log.append('stdout', `Target object: ${data.object_name}\n`);
    }
    config.refreshObjects();
    stageCtrl.activateStage(resumeFromStage);
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

ws.on('status', async (msg) => {
  await applyStatusSnapshot(msg);
});

ws.on('stage_start', (msg) => {
  // Stage 2 is marked "interactive" while waiting for approval.
  // Once Stage 3 starts, Pi3X approval is complete and the pill should return to "complete".
  if (msg.stage >= 3 && pipelineUI.getStageStatus(2) === 'interactive') {
    pipelineUI.stageComplete(2);
    stageCtrl.setStageState(2, 'complete');
    hidePi3xApproveButton();
  }

  stageCtrl.activateStage(msg.stage);
  stageCtrl.setStageState(msg.stage, 'running');
  pipelineUI.stageStart(msg.stage);
  setOverallProgress(msg.overall_progress ?? pipelineUI.getOverallProgress());
  log.append('stdout', `\n=== Stage ${msg.stage}/6: ${msg.label} ===\n`);
});

ws.on('extract_frames_result', (msg) => {
  _extractedFrameCount = msg.frame_count || 0;
});

ws.on('stage_complete', async (msg) => {
  pipelineUI.stageComplete(msg.stage, msg.elapsed);
  setOverallProgress(msg.overall_progress ?? pipelineUI.getOverallProgress());
  if (msg.error) {
    pipelineUI.stageFailed(msg.stage);
    stageCtrl.setStageState(msg.stage, 'failed');
    return;
  }
  stageCtrl.setStageState(msg.stage, 'complete');

  // Auto-load results per stage
  if (msg.stage === 1) {
    const empty = document.querySelector('#stage-panel-1 .stage-panel-empty');
    if (empty) empty.classList.add('hidden');
    await preview.loadGallery(_extractedFrameCount);
  } else if (msg.stage === 2) {
    await preview.loadPi3xResults(cameraOverlay);
  } else if (msg.stage === 3) {
    // SAM2 complete — reload Pi3X viewer with filtered object.ply
    await preview.loadPi3xResults(cameraOverlay, 'object.ply');
  } else if (msg.stage >= 4 && msg.stage <= 6) {
    await preview.loadStageResult(msg.stage);
  }
});

ws.on('stage_progress', (msg) => {
  pipelineUI.stageProgress(msg.stage, msg.progress, msg.detail);
  setOverallProgress(msg.overall_progress ?? pipelineUI.getOverallProgress());
});

ws.on('sam2_ready', (msg) => {
  pipelineUI.stageInteractive(3);
  sam2.activate(msg.frame_count, msg.width, msg.height);
  sam2Verify.hide();
  stageCtrl.activateStage(3);
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

ws.on('pi3x_preview_ready', () => {
  pipelineUI.stageInteractive(2);
  stageCtrl.activateStage(2);
  showPi3xApproveButton();
  log.append('stdout', 'Pi3X complete. Review the 3D preview and click "Approve & Continue".\n');
});

ws.on('pipeline_complete', (msg) => {
  config.setRunning(false);
  config.setActiveStage(null);
  config.refreshObjects();
  setOverallProgress(msg.overall_progress ?? 100);
  setStatus('complete', `Done (${formatTime(msg.elapsed)})`);
  log.append('stdout', `\n=== Pipeline Complete! (${formatTime(msg.elapsed)}) ===\n`);
});

ws.on('pipeline_error', (msg) => {
  config.setRunning(false);
  config.setActiveStage(null);
  config.refreshObjects();
  const cancelled = /cancel/i.test(String(msg.error || ''));
  setStatus(cancelled ? 'idle' : 'error', cancelled ? 'Cancelled' : 'Error');
  if (msg.stage >= 1 && msg.stage <= 6) {
    pipelineUI.stageFailed(msg.stage);
    stageCtrl.setStageState(msg.stage, 'failed');
  }
  setOverallProgress(msg.overall_progress ?? pipelineUI.getOverallProgress());
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

function getStageInfo(statusMsg, stage) {
  return statusMsg?.stages?.[String(stage)] || null;
}

function isStageDone(statusMsg, stage) {
  const info = getStageInfo(statusMsg, stage);
  if (!info) return false;
  if (info.status === 'complete') return true;
  return Number(info.progress) >= 100;
}

function isStageAvailable(statusMsg, stage) {
  const info = getStageInfo(statusMsg, stage);
  if (!info) return false;
  return info.status === 'complete' || info.status === 'interactive' || Number(info.progress) > 0;
}

function resolvePreferredStage(statusMsg) {
  if (!statusMsg) return 1;
  const current = Number(statusMsg.current_stage);
  if (Number.isFinite(current) && current >= 1 && current <= 6) return current;
  for (let stage = 6; stage >= 1; stage--) {
    if (isStageDone(statusMsg, stage)) return stage;
  }
  return 1;
}

function buildStatusKey(statusMsg) {
  const objectName = statusMsg?.object_name || '';
  const stageState = [];
  for (let i = 1; i <= 6; i++) {
    const info = getStageInfo(statusMsg, i);
    stageState.push(`${info?.status || 'pending'}:${Math.round(Number(info?.progress) || 0)}`);
  }
  return `${objectName}|${stageState.join('|')}`;
}

function syncStageStates(statusMsg) {
  for (let i = 1; i <= 6; i++) {
    const info = getStageInfo(statusMsg, i);
    stageCtrl.setStageState(i, info?.status || 'pending');
  }
}

async function hydrateOutputsFromStatus(statusMsg, opts = {}) {
  if (!statusMsg?.object_name) return;

  const key = buildStatusKey(statusMsg);
  if (!opts.force && key === _hydratedStatusKey) {
    return;
  }
  _hydratedStatusKey = key;

  cameraOverlay.remove();
  sam2.deactivate();
  sam2Verify.hide();
  preview.reset();
  hidePi3xApproveButton();

  if (isStageDone(statusMsg, 1)) {
    const empty = document.querySelector('#stage-panel-1 .stage-panel-empty');
    if (empty) empty.classList.add('hidden');
    await preview.loadGallery(Number(statusMsg.frame_count) || 0);
  }

  if (isStageAvailable(statusMsg, 2)) {
    const pi3xFile = isStageDone(statusMsg, 3) ? 'object.ply' : 'object_full.ply';
    await preview.loadPi3xResults(cameraOverlay, pi3xFile);
  }

  if (isStageDone(statusMsg, 4)) await preview.loadStageResult(4);
  if (isStageDone(statusMsg, 5)) await preview.loadStageResult(5);
  if (isStageDone(statusMsg, 6)) await preview.loadStageResult(6);

  if (opts.activate !== false) {
    stageCtrl.activateStage(resolvePreferredStage(statusMsg));
  }
}

async function applyStatusSnapshot(statusMsg, opts = {}) {
  pipelineUI.updateAll(statusMsg);
  syncStageStates(statusMsg);
  setOverallProgress(statusMsg.overall_progress ?? pipelineUI.getOverallProgress());

  if (statusMsg.object_name) {
    config.setObjectName(statusMsg.object_name);
  }

  if (statusMsg.running) {
    config.setRunning(true);
    setStatus('running', 'Running');
    stageCtrl.activateStage(resolvePreferredStage(statusMsg));

    // Reconnect-safe: if already waiting for Pi3X approval, show button again.
    if (getStageInfo(statusMsg, 2)?.status === 'interactive') {
      await preview.loadPi3xResults(cameraOverlay);
      showPi3xApproveButton();
    }
    return;
  }

  config.setRunning(false);
  config.setActiveStage(null);

  const allDone = [1, 2, 3, 4, 5, 6].every((stage) => isStageDone(statusMsg, stage));
  if (allDone) setStatus('complete', 'Done');
  else setStatus('idle', 'Idle');

  if (statusMsg.object_name) {
    await hydrateOutputsFromStatus(statusMsg, {
      force: opts.forceHydrate === true,
      activate: true,
    });
  }
}

function showPi3xApproveButton() {
  const btn = document.getElementById('pi3x-approve');
  if (!btn) return;
  btn.style.display = 'inline-block';
  btn.disabled = false;
  btn.textContent = 'Approve & Continue';
  btn.onclick = async () => {
    btn.disabled = true;
    btn.textContent = 'Proceeding...';
    await fetch('/api/pi3x/approve', { method: 'POST' });
    btn.style.display = 'none';
  };
}

function hidePi3xApproveButton() {
  const btn = document.getElementById('pi3x-approve');
  if (!btn) return;
  btn.style.display = 'none';
}

function setStatus(state, text) {
  statusBadge.textContent = text;
  statusBadge.className = 'badge';
  if (state === 'running') statusBadge.classList.add('badge-running');
  else if (state === 'complete') statusBadge.classList.add('badge-complete');
  else if (state === 'error') statusBadge.classList.add('badge-error');
  else statusBadge.classList.add('badge-idle');
}

function setOverallProgress(value) {
  if (!overallProgressBadge) return;
  const pct = Math.max(0, Math.min(100, Number(value) || 0));
  overallProgressBadge.textContent = `Overall: ${Math.round(pct)}%`;
}

function formatTime(secs) {
  if (secs < 60) return `${Math.round(secs)}s`;
  const m = Math.floor(secs / 60);
  const s = Math.round(secs % 60);
  return `${m}m${s}s`;
}

// ── Boot ─────────────────────────────────────────────────────

ws.connect();
