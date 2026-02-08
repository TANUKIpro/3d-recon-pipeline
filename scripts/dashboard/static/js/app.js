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

const STAGE_COUNT = 6;
const TRANSITION_STAGE_MAX = 5;

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

const _taskConfirmBars = {};
const _taskConfirmMessages = {};
const _taskConfirmButtons = {};

let _extractedFrameCount = 0;
let _hydratedStatusKey = '';
let _objectLoadRequestId = 0;
let _waitingConfirmationStage = null;

for (let stage = 1; stage <= STAGE_COUNT; stage++) {
  _taskConfirmBars[stage] = document.querySelector(`.task-confirm-bar[data-stage="${stage}"]`);
  _taskConfirmMessages[stage] = _taskConfirmBars[stage]?.querySelector('.task-confirm-message') || null;
  _taskConfirmButtons[stage] = _taskConfirmBars[stage]?.querySelector('.task-confirm-btn') || null;

  if (stage <= TRANSITION_STAGE_MAX && _taskConfirmButtons[stage]) {
    _taskConfirmButtons[stage].addEventListener('click', () => confirmNextStage(stage));
  }
}
resetTaskConfirmBars();

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
    resetTaskConfirmBars();
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
    resetTaskConfirmBars(resumeFromStage);
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
  // Stage 2 is marked "interactive" while waiting for confirmation.
  // Once Stage 3 starts, Stage 2 confirmation is complete and the pill should return to "complete".
  if (msg.stage >= 3 && pipelineUI.getStageStatus(2) === 'interactive') {
    pipelineUI.stageComplete(2);
    stageCtrl.setStageState(2, 'complete');
  }

  const prevStage = Number(msg.stage) - 1;
  if (prevStage >= 1 && prevStage <= TRANSITION_STAGE_MAX) {
    if (_waitingConfirmationStage === prevStage) {
      _waitingConfirmationStage = null;
    }
    setTaskConfirmState(prevStage, 'confirmed', defaultTaskConfirmConfirmedMessage(prevStage));
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
  stageCtrl.setStageState(2, 'interactive');
  stageCtrl.activateStage(2);
  log.append('stdout', 'Pi3X complete. Review the 3D preview, then confirm on Stage 2.\n');
});

ws.on('next_stage_confirmation_required', (msg) => {
  const fromStage = Number(msg.from_stage);
  if (fromStage >= 1 && fromStage <= TRANSITION_STAGE_MAX) {
    if (_waitingConfirmationStage !== null && _waitingConfirmationStage !== fromStage) {
      setTaskConfirmState(
        _waitingConfirmationStage,
        'idle',
        defaultTaskConfirmIdleMessage(_waitingConfirmationStage),
      );
    }
    _waitingConfirmationStage = fromStage;
    pipelineUI.stageInteractive(fromStage);
    stageCtrl.setStageState(fromStage, 'interactive');
    stageCtrl.activateStage(fromStage);
    setTaskConfirmState(fromStage, 'waiting', String(msg.message || defaultTaskConfirmWaitingMessage(fromStage)));
  }
  if (msg.message) {
    log.append('stdout', `${msg.message}\n`);
  }
});

ws.on('next_stage_confirmation_cleared', (msg) => {
  const fromStage = Number(msg.from_stage);
  if (fromStage >= 1 && fromStage <= TRANSITION_STAGE_MAX) {
    if (_waitingConfirmationStage === fromStage) {
      _waitingConfirmationStage = null;
    }
    setTaskConfirmState(fromStage, 'confirmed', defaultTaskConfirmConfirmedMessage(fromStage));
  }
});

ws.on('pipeline_complete', (msg) => {
  config.setRunning(false);
  config.setActiveStage(null);
  config.refreshObjects();
  _waitingConfirmationStage = null;
  for (let stage = 1; stage <= TRANSITION_STAGE_MAX; stage++) {
    setTaskConfirmState(stage, 'confirmed', defaultTaskConfirmConfirmedMessage(stage));
  }
  setTaskConfirmState(6, 'final', 'Final stage complete. No next-stage confirmation.');
  setOverallProgress(msg.overall_progress ?? 100);
  setStatus('complete', `Done (${formatTime(msg.elapsed)})`);
  log.append('stdout', `\n=== Pipeline Complete! (${formatTime(msg.elapsed)}) ===\n`);
});

ws.on('pipeline_error', (msg) => {
  config.setRunning(false);
  config.setActiveStage(null);
  config.refreshObjects();
  if (_waitingConfirmationStage !== null) {
    setTaskConfirmState(
      _waitingConfirmationStage,
      'idle',
      `Stage ${_waitingConfirmationStage} confirmation was interrupted.`,
    );
    _waitingConfirmationStage = null;
  }
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
  const next = statusMsg?.next_stage_confirmation || {};
  const stageState = [];
  for (let i = 1; i <= 6; i++) {
    const info = getStageInfo(statusMsg, i);
    stageState.push(`${info?.status || 'pending'}:${Math.round(Number(info?.progress) || 0)}`);
  }
  return [
    objectName,
    stageState.join('|'),
    next.required ? 'wait' : 'idle',
    next.from_stage || '',
    next.to_stage || '',
  ].join('|');
}

function syncStageStates(statusMsg) {
  for (let i = 1; i <= 6; i++) {
    const info = getStageInfo(statusMsg, i);
    stageCtrl.setStageState(i, info?.status || 'pending');
  }
}

function defaultTaskConfirmIdleMessage(stage) {
  return `Stage ${stage} confirmation will appear after completion.`;
}

function defaultTaskConfirmWaitingMessage(stage) {
  return `Stage ${stage} complete. Confirm to continue to Stage ${stage + 1}.`;
}

function defaultTaskConfirmConfirmedMessage(stage) {
  return `Stage ${stage} confirmed. Proceeded to Stage ${stage + 1}.`;
}

function setTaskConfirmState(stage, state, message) {
  const bar = _taskConfirmBars[stage];
  const msg = _taskConfirmMessages[stage];
  const btn = _taskConfirmButtons[stage];
  if (!bar || !msg) return;

  bar.classList.remove('idle', 'waiting', 'sending', 'confirmed');

  if (stage === STAGE_COUNT || state === 'final') {
    bar.classList.add('task-confirm-final');
    msg.textContent = String(message || 'Final stage. No next-stage confirmation.');
    if (btn) {
      btn.disabled = true;
      btn.textContent = 'Done';
    }
    return;
  }

  bar.classList.remove('task-confirm-final');
  bar.classList.add(state);
  msg.textContent = String(message || '');

  if (!btn) return;

  if (state === 'waiting') {
    btn.disabled = false;
    btn.textContent = 'Continue';
  } else if (state === 'sending') {
    btn.disabled = true;
    btn.textContent = 'Waiting...';
  } else if (state === 'confirmed') {
    btn.disabled = true;
    btn.textContent = 'Confirmed';
  } else {
    btn.disabled = true;
    btn.textContent = 'Continue';
  }
}

function resetTaskConfirmBars(resumeFromStage = 1) {
  const resumeStage = Math.max(1, Math.min(STAGE_COUNT, Number(resumeFromStage) || 1));
  _waitingConfirmationStage = null;
  for (let stage = 1; stage <= TRANSITION_STAGE_MAX; stage++) {
    if (stage < resumeStage) {
      setTaskConfirmState(stage, 'confirmed', `Stage ${stage} already completed before resume.`);
    } else {
      setTaskConfirmState(stage, 'idle', defaultTaskConfirmIdleMessage(stage));
    }
  }
  setTaskConfirmState(6, 'final', 'Final stage. No next-stage confirmation.');
}

function isTransitionConfirmed(statusMsg, stage) {
  if (!statusMsg || stage < 1 || stage > TRANSITION_STAGE_MAX) return false;

  const current = Number(statusMsg.current_stage);
  if (Number.isFinite(current) && current > stage) {
    return true;
  }

  const nextInfo = getStageInfo(statusMsg, stage + 1);
  if (!nextInfo) return false;
  if (nextInfo.status !== 'pending') return true;
  return Number(nextInfo.progress) > 0;
}

function syncTaskConfirmBarsFromStatus(statusMsg) {
  if (!statusMsg) return;

  const next = statusMsg.next_stage_confirmation || {};
  const waitingStage = next.required === true ? Number(next.from_stage) : NaN;
  _waitingConfirmationStage = Number.isFinite(waitingStage)
    && waitingStage >= 1
    && waitingStage <= TRANSITION_STAGE_MAX
    ? waitingStage
    : null;

  const resumeStage = Math.max(1, Math.min(STAGE_COUNT, Number(statusMsg.resume_from_stage) || 1));

  for (let stage = 1; stage <= TRANSITION_STAGE_MAX; stage++) {
    if (_waitingConfirmationStage === stage) {
      setTaskConfirmState(stage, 'waiting', String(next.message || defaultTaskConfirmWaitingMessage(stage)));
      continue;
    }

    if (isTransitionConfirmed(statusMsg, stage)) {
      setTaskConfirmState(stage, 'confirmed', defaultTaskConfirmConfirmedMessage(stage));
      continue;
    }

    if (statusMsg.running && stage < resumeStage) {
      setTaskConfirmState(stage, 'confirmed', `Stage ${stage} already completed before resume.`);
      continue;
    }

    setTaskConfirmState(stage, 'idle', defaultTaskConfirmIdleMessage(stage));
  }

  if (isStageDone(statusMsg, 6)) {
    setTaskConfirmState(6, 'final', 'Final stage complete. No next-stage confirmation.');
  } else {
    setTaskConfirmState(6, 'final', 'Final stage. No next-stage confirmation.');
  }
}

async function confirmNextStage(stage) {
  if (_waitingConfirmationStage !== stage) return;

  setTaskConfirmState(stage, 'sending', `Stage ${stage} confirmed. Waiting for Stage ${stage + 1} start...`);

  try {
    const res = await fetch('/api/pipeline/confirm-next', { method: 'POST' });
    const data = await res.json();
    if (!res.ok || data.error) {
      throw new Error(data.error || `HTTP ${res.status}`);
    }

    if (data.status === 'no_waiting_confirmation') {
      _waitingConfirmationStage = null;
      setTaskConfirmState(stage, 'confirmed', defaultTaskConfirmConfirmedMessage(stage));
      return;
    }

    // Keep "sending" state until backend emits stage start or cleared event.
  } catch (e) {
    setTaskConfirmState(stage, 'waiting', `${defaultTaskConfirmWaitingMessage(stage)} (${e.message})`);
    log.append('stderr', `Next-stage confirmation failed at stage ${stage}: ${e.message}\n`);
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
  syncTaskConfirmBarsFromStatus(statusMsg);
  setOverallProgress(statusMsg.overall_progress ?? pipelineUI.getOverallProgress());

  if (statusMsg.object_name) {
    config.setObjectName(statusMsg.object_name);
  }

  if (statusMsg.running) {
    config.setRunning(true);
    setStatus('running', 'Running');
    const waiting = statusMsg?.next_stage_confirmation?.required === true;
    const waitingFromStage = Number(statusMsg?.next_stage_confirmation?.from_stage);
    if (waiting) {
      await hydrateOutputsFromStatus(statusMsg, {
        force: opts.forceHydrate === true,
        activate: false,
      });
      if (Number.isFinite(waitingFromStage) && waitingFromStage >= 1 && waitingFromStage <= 6) {
        stageCtrl.activateStage(waitingFromStage);
      } else {
        stageCtrl.activateStage(resolvePreferredStage(statusMsg));
      }
    } else {
      stageCtrl.activateStage(resolvePreferredStage(statusMsg));
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
