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
import { CheckpointPanel } from './checkpoint-panel.js';

const STAGE_COUNT = 8;
const TRANSITION_STAGE_MAX = 7;
const DEFAULT_TAUBIN_NU = -0.53;
const MESH_METHOD_DEFAULT = 'poisson';
const MESH_METHOD_SET = new Set(['diffcd', 'poisson']);
const CLASSICAL_PREVIEW_TITLE = 'Classical Mesh result';

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
const checkpoints = new CheckpointPanel();

function appendLog(stream, text, opts = {}) {
  const stage = Number(opts.stage);
  log.append(stream, text, {
    ...opts,
    stage: Number.isFinite(stage) ? stage : stageCtrl.activeStage,
  });
}

const statusBadge = document.getElementById('status-badge');
const vramBadge = document.getElementById('vram-badge');
const overallProgressBadge = document.getElementById('overall-progress-badge');
const meshPostToolbar = document.getElementById('mesh-post-toolbar');
const meshPostMethodInput = document.getElementById('mesh-post-method');
const meshPostIterationsInput = document.getElementById('mesh-post-iterations');
const meshPostLambdaInput = document.getElementById('mesh-post-lambda');
const meshPostDownsampleEnabledInput = document.getElementById('mesh-post-downsample-enabled');
const meshPostTargetFacesInput = document.getElementById('mesh-post-target-faces');
const meshPostTriggerFacesInput = document.getElementById('mesh-post-trigger-faces');
const meshPostApplyBtn = document.getElementById('mesh-post-apply');
const meshPostResetBtn = document.getElementById('mesh-post-reset');
const meshPostStatus = document.getElementById('mesh-post-status');
const meshPhaseStatusBar = document.getElementById('mesh-phase-status');
const meshPhaseStatusText = document.getElementById('mesh-phase-status-text');
const meshRepairToolbar = document.getElementById('mesh-repair-toolbar');
const meshRepairStatus = document.getElementById('mesh-repair-status');
const meshRepairClearBtn = document.getElementById('mesh-repair-clear');
const meshRepairApplyBtn = document.getElementById('mesh-repair-apply');
const meshMethodPills = {
  diffcd: document.getElementById('mesh-pill-diffcd'),
  poisson: document.getElementById('mesh-pill-poisson'),
};

const _taskConfirmBars = {};
const _taskConfirmMessages = {};
const _taskConfirmButtons = {};

let _extractedFrameCount = 0;
let _hydratedStatusKey = '';
let _objectLoadRequestId = 0;
let _waitingConfirmationStage = null;
let _waitingConfirmationToStage = null;
let _latestStatusSnapshot = null;
let _meshPostInFlight = false;
let _meshMethod = MESH_METHOD_DEFAULT;
let _meshRepairCandidateCount = 0;
let _meshRepairInFlight = false;

for (let stage = 1; stage <= STAGE_COUNT; stage++) {
  _taskConfirmBars[stage] = document.querySelector(`.task-confirm-bar[data-stage="${stage}"]`);
  _taskConfirmMessages[stage] = _taskConfirmBars[stage]?.querySelector('.task-confirm-message') || null;
  _taskConfirmButtons[stage] = _taskConfirmBars[stage]?.querySelector('.task-confirm-btn') || null;

  if (stage <= TRANSITION_STAGE_MAX && _taskConfirmButtons[stage]) {
    _taskConfirmButtons[stage].addEventListener('click', () => confirmNextStage(stage));
  }
}
resetTaskConfirmBars();
initMeshPostToolbar();
initMeshRepairToolbar();
bindMeshMethodPills();
applyMeshMethod(config.getMeshMethod?.() || MESH_METHOD_DEFAULT, { announce: false });
preview.onMeshRepairSelectionChanged = (selectedIds) => {
  updateMeshRepairStatus(selectedIds);
};

config.onCropScaleChanged = (cropScale) => {
  preview.updateCropBbox(cropScale);
};

// ── Stage-activated event: lazy-init 3D ─────────────────────

document.addEventListener('stage-activated', async (e) => {
  const stage = e.detail.stage;

  // Config panel: show stage-specific params
  config.setActiveStage(stage);
  checkpoints.setActiveStage(stage);

  // For 3D stages (2=Pi3X, 4-8), activate the renderer if scene is initialized
  if (stage === 2 || (stage >= 4 && stage <= 8)) {
    if (preview._stages?.[stage]?.initialized) {
      preview.activateStage(stage);
    }
  }

  // Show crop bbox preview when Stage 6 is activated with Stage 5 done and Stage 6 not yet done
  if (stage === 6) {
    const s5 = stageCtrl.getStageState(5);
    const s6 = stageCtrl.getStageState(6);
    if (s5 === 'complete' && s6 !== 'complete' && s6 !== 'interactive') {
      await preview.showCropBbox(config.getCropScale(), { preferPreview: _meshMethod === 'poisson' });
    }
  }

  setTaskConfirmVisibleStage(stage);
});

// Align initial config view with default stage tab.
stageCtrl.activateStage(stageCtrl.activeStage);

// ── Config panel callbacks ───────────────────────────────────

config.onObjectSelected = async (objectName) => {
  const reqId = ++_objectLoadRequestId;
  if (!objectName) {
    _hydratedStatusKey = '';
    _latestStatusSnapshot = null;
    cameraOverlay.remove();
    sam2.deactivate();
    sam2Verify.hide();
    preview.reset();
    pipelineUI.reset();
    resetTaskConfirmBars();
    setMeshPostToolbarVisible(false);
    setMeshPostStatus('');
    setMeshRepairToolbarVisible(false);
    setMeshRepairStatus('');
    _meshRepairCandidateCount = 0;
    _meshRepairInFlight = false;
    setMeshPhaseStatus('', '');
    setOverallProgress(0);
    setStatus('idle', 'Idle');
    pipelineUI.setMeshMethodEnabled(true);
    applyMeshMethod(MESH_METHOD_DEFAULT, { announce: false });
    checkpoints.reset(1);
    stageCtrl.activateStage(1);
    return;
  }

  try {
    const res = await fetch('/api/pipeline/load-object', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: objectName }),
    });
    const data = await res.json();
    if (reqId !== _objectLoadRequestId) return;
    if (!res.ok || data.error) {
      appendLog('stderr', `Failed to load object ${objectName}: ${data.error || res.status}\n`);
      return;
    }

    const status = data.pipeline_status;
    if (status) {
      await applyStatusSnapshot(status, { forceHydrate: true });
      appendLog('stdout', `Loaded object: ${objectName}\n`);
    }
  } catch (e) {
    if (reqId !== _objectLoadRequestId) return;
    appendLog('stderr', `Failed to load object ${objectName}: ${e.message}\n`);
  }
};

config.onStart = async (cfg) => {
  try {
    // Ignore late responses from previous object-load requests while starting a new run.
    _objectLoadRequestId += 1;
    applyMeshMethod(cfg.mesh_method || _meshMethod, { announce: false });
    config.setRunning(true);
    pipelineUI.setMeshMethodEnabled(false);
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
      pipelineUI.setMeshMethodEnabled(true);
      setStatus('idle', 'Idle');
      return;
    }

    const resumeFromStage = Math.max(1, Math.min(STAGE_COUNT, Number(data.resume_from_stage) || 1));

    preview.clearFromStage(resumeFromStage);
    cameraOverlay.remove();
    sam2.deactivate();
    sam2Verify.hide();
    resetTaskConfirmBars(resumeFromStage);
    pipelineUI.resetFromStage(resumeFromStage);
    checkpoints.reset(resumeFromStage);
    _extractedFrameCount = 0;
    _hydratedStatusKey = '';
    _latestStatusSnapshot = null;
    setMeshPostToolbarVisible(false);
    setMeshPostStatus('');
    setMeshRepairToolbarVisible(false);
    setMeshRepairStatus('');
    _meshRepairCandidateCount = 0;
    _meshRepairInFlight = false;
    setMeshPhaseStatus('', '');
    setOverallProgress(pipelineUI.getOverallProgress());

    if (data.object_name) {
      config.setObjectName(data.object_name);
      appendLog('stdout', `Target object: ${data.object_name}\n`);
    }
    config.refreshObjects();
    stageCtrl.activateStage(resumeFromStage);
  } catch (e) {
    alert('Start failed: ' + e.message);
    config.setRunning(false);
    pipelineUI.setMeshMethodEnabled(true);
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
  checkpoints.onStageStart(msg.stage);
  const prevStage = Number(msg.stage) - 1;
  if (prevStage >= 1 && prevStage <= TRANSITION_STAGE_MAX) {
    const prevStatus = pipelineUI.getStageStatus(prevStage);
    if (prevStatus === 'interactive' || prevStatus === 'running') {
      pipelineUI.stageComplete(prevStage);
    }
    stageCtrl.setStageState(prevStage, 'complete');

    const waitingToStage = _waitingConfirmationToStage;
    if (_waitingConfirmationStage === prevStage) {
      _waitingConfirmationStage = null;
      _waitingConfirmationToStage = null;
    }
    setTaskConfirmState(
      prevStage,
      'confirmed',
      defaultTaskConfirmConfirmedMessage(prevStage, waitingToStage),
    );
  }

  stageCtrl.activateStage(msg.stage);
  stageCtrl.setStageState(msg.stage, 'running');
  pipelineUI.stageStart(msg.stage);
  setOverallProgress(msg.overall_progress ?? pipelineUI.getOverallProgress());
  appendLog('stdout', `\n=== Stage ${msg.stage}/8: ${msg.label} ===\n`, { stage: msg.stage });
  setMeshPostToolbarVisible(false);
  if (!_meshPostInFlight) {
    setMeshPostStatus('');
  }
  if (Number(msg.stage) !== 7 || !_meshRepairInFlight) {
    setMeshRepairToolbarVisible(false);
    if (!_meshRepairInFlight) {
      setMeshRepairStatus('');
    }
  }
  if (Number(msg.stage) === 7) {
    _meshRepairCandidateCount = 0;
    _meshRepairInFlight = false;
  }
  if (Number(msg.stage) === 5 && _meshMethod === 'poisson') {
    setMeshPhaseStatus('Classical Mesh started.', 'live');
  } else if (Number(msg.stage) === 5 && _meshMethod !== 'poisson') {
    setMeshPhaseStatus('', '');
  }
});

ws.on('extract_frames_result', (msg) => {
  _extractedFrameCount = msg.frame_count || 0;
});

ws.on('stage_complete', async (msg) => {
  pipelineUI.stageComplete(msg.stage, msg.elapsed);
  setOverallProgress(msg.overall_progress ?? pipelineUI.getOverallProgress());
  if (msg.error) {
    checkpoints.onStageFailed(msg.stage, msg.error, msg.checkpoint_id);
    pipelineUI.stageFailed(msg.stage);
    stageCtrl.setStageState(msg.stage, 'failed');
    if (Number(msg.stage) === 5 && _meshMethod === 'poisson') {
      setMeshPhaseStatus(`Stage 5 failed: ${msg.error}`, 'error');
    }
    if (Number(msg.stage) === 7) {
      setMeshRepairToolbarVisible(false);
      setMeshRepairStatus(`Stage 7 failed: ${msg.error}`, 'error');
      _meshRepairInFlight = false;
    }
    return;
  }
  checkpoints.onStageComplete(msg.stage);
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
  } else if (msg.stage === 5) {
    await preview.loadStageResult(5, { preferPreview: _meshMethod === 'poisson' });
  } else if (msg.stage >= 4 && msg.stage <= 8) {
    await preview.loadStageResult(msg.stage);
  }

  if (msg.stage === 5) {
    if (_meshMethod === 'poisson') {
      setMeshPhaseStatus(`Showing: ${CLASSICAL_PREVIEW_TITLE}`, 'ready');
    } else {
      setMeshPhaseStatus('', '');
    }
    setMeshPostToolbarVisible(true);
    setMeshPostEnabled(true);
    if (!_meshPostInFlight) {
      setMeshPostStatus('');
    }
  }
  if (msg.stage === 7) {
    _meshRepairInFlight = false;
    setMeshRepairToolbarVisible(false);
    if (!_meshRepairInFlight) {
      setMeshRepairStatus('');
    }
  }
});

ws.on('stage_progress', (msg) => {
  checkpoints.onStageProgress(msg.stage, msg.detail, 'running', msg.checkpoint_id);
  pipelineUI.stageProgress(msg.stage, msg.progress, msg.detail);
  setOverallProgress(msg.overall_progress ?? pipelineUI.getOverallProgress());
  if (Number(msg.stage) === 5 && _meshMethod === 'poisson') {
    setMeshPhaseStatus('Running: Classical Mesh', 'live');
  }
});

ws.on('sam2_ready', (msg) => {
  checkpoints.onStageInteractive(3, 'Waiting for interactive clicks');
  pipelineUI.stageInteractive(3);
  sam2.activate(msg.frame_count, msg.width, msg.height);
  sam2Verify.hide();
  stageCtrl.activateStage(3);
  appendLog('stdout', `SAM2 ready: ${msg.frame_count} frames (${msg.width}x${msg.height})\n`, { stage: 3 });
  appendLog('stdout', 'Click on the object to segment. Right-click for negative points.\n', { stage: 3 });
});

ws.on('sam2_propagating', () => {
  sam2.deactivate();
  appendLog('stdout', 'Propagating masks to all frames...\n', { stage: 3 });
});

ws.on('sam2_propagate_progress', (msg) => {
  appendLog('stdout', `Propagating frame ${msg.frame}/${msg.total}`, { stage: 3, progress: true });
});

ws.on('sam2_verification_ready', (msg) => {
  sam2Verify.show(msg.frame_count);
  appendLog('stdout', 'Mask propagation complete. Please verify the results.\n', { stage: 3 });
});

ws.on('pi3x_preview_ready', () => {
  checkpoints.onStageInteractive(2, 'Waiting for next-stage confirmation');
  pipelineUI.stageInteractive(2);
  stageCtrl.setStageState(2, 'interactive');
  stageCtrl.activateStage(2);
  appendLog('stdout', 'Pi3X complete. Review the 3D preview, then confirm on Stage 2.\n', { stage: 2 });
});

ws.on('mesh_repair_ready', async (msg) => {
  checkpoints.onStageInteractive(7, 'Waiting for repair-loop selection');
  pipelineUI.stageInteractive(7);
  stageCtrl.setStageState(7, 'interactive');
  try {
    await activateMeshRepairSelectionFromApi();
    stageCtrl.activateStage(7);
    const loopCount = Number(msg.candidate_count) || _meshRepairCandidateCount;
    appendLog(
      'stdout',
      `Mesh Repair ready: ${loopCount} selectable loops. Click closed surfaces, then confirm selection (0 selected = skip).\n`,
      { stage: 7 },
    );
  } catch (e) {
    setMeshRepairToolbarVisible(false);
    setMeshRepairStatus(`Mesh Repair UI failed: ${e.message}`, 'error');
    appendLog('stderr', `Mesh Repair setup failed: ${e.message}\n`, { stage: 7 });
  }
});

ws.on('next_stage_confirmation_required', (msg) => {
  const fromStage = Number(msg.from_stage);
  const toStage = Number(msg.to_stage);
  if (fromStage >= 1 && fromStage <= TRANSITION_STAGE_MAX) {
    // Keep checkpoint detail machine-readable for deterministic completion mapping.
    checkpoints.onStageInteractive(fromStage, 'Waiting for next-stage confirmation');
    if (_waitingConfirmationStage !== null && _waitingConfirmationStage !== fromStage) {
      setTaskConfirmState(
        _waitingConfirmationStage,
        'confirmed',
        defaultTaskConfirmConfirmedMessage(_waitingConfirmationStage, _waitingConfirmationToStage),
      );
    }
    _waitingConfirmationStage = fromStage;
    _waitingConfirmationToStage = Number.isFinite(toStage) ? toStage : null;
    pipelineUI.stageInteractive(fromStage);
    stageCtrl.setStageState(fromStage, 'interactive');
    stageCtrl.activateStage(fromStage);
    setTaskConfirmState(
      fromStage,
      'waiting',
      String(msg.message || defaultTaskConfirmWaitingMessage(fromStage, _waitingConfirmationToStage)),
    );
    setTaskConfirmVisibleStage(fromStage);
  }
  if (msg.message) {
    appendLog('stdout', `${msg.message}\n`, { stage: fromStage });
  }
});

ws.on('next_stage_confirmation_cleared', (msg) => {
  const fromStage = Number(msg.from_stage);
  const toStage = Number(msg.to_stage);
  if (fromStage >= 1 && fromStage <= TRANSITION_STAGE_MAX) {
    if (_waitingConfirmationStage === fromStage) {
      _waitingConfirmationStage = null;
      _waitingConfirmationToStage = null;
    }
    setTaskConfirmState(fromStage, 'confirmed', defaultTaskConfirmConfirmedMessage(fromStage, toStage));
    setTaskConfirmVisibleStage(stageCtrl.activeStage);
  }
});

ws.on('pipeline_complete', (msg) => {
  checkpoints.onStageComplete(8);
  config.setRunning(false);
  pipelineUI.setMeshMethodEnabled(true);
  config.setActiveStage(null);
  config.refreshObjects();
  _waitingConfirmationStage = null;
  _waitingConfirmationToStage = null;
  for (let stage = 1; stage <= TRANSITION_STAGE_MAX; stage++) {
    setTaskConfirmState(stage, 'confirmed', defaultTaskConfirmConfirmedMessage(stage));
  }
  setTaskConfirmState(8, 'final', 'Final stage complete. No next-stage confirmation.');
  setTaskConfirmVisibleStage(8);
  setOverallProgress(msg.overall_progress ?? 100);
  setStatus('complete', `Done (${formatTime(msg.elapsed)})`);
  setMeshPostToolbarVisible(true);
  setMeshPostEnabled(true);
  setMeshRepairToolbarVisible(false);
  setMeshRepairStatus('');
  _meshRepairCandidateCount = 0;
  _meshRepairInFlight = false;
  if (_meshMethod === 'poisson') {
    setMeshPhaseStatus(`Showing: ${CLASSICAL_PREVIEW_TITLE}`, 'ready');
  } else {
    setMeshPhaseStatus('', '');
  }
  appendLog('stdout', `\n=== Pipeline Complete! (${formatTime(msg.elapsed)}) ===\n`, { stage: 8 });
});

ws.on('pipeline_error', (msg) => {
  config.setRunning(false);
  pipelineUI.setMeshMethodEnabled(true);
  config.setActiveStage(null);
  config.refreshObjects();
  if (_waitingConfirmationStage !== null) {
    setTaskConfirmState(
      _waitingConfirmationStage,
      'idle',
      `Stage ${_waitingConfirmationStage} confirmation was interrupted.`,
    );
    _waitingConfirmationStage = null;
    _waitingConfirmationToStage = null;
  }
  setTaskConfirmVisibleStage(stageCtrl.activeStage);
  const cancelled = String(msg.reason_code || '').startsWith('cancelled')
    || /cancel/i.test(String(msg.error || ''));
  setStatus(cancelled ? 'idle' : 'error', cancelled ? 'Cancelled' : 'Error');
  if (msg.stage >= 1 && msg.stage <= 8) {
    checkpoints.onStageFailed(msg.stage, msg.error, msg.checkpoint_id);
    pipelineUI.stageFailed(msg.stage);
    stageCtrl.setStageState(msg.stage, 'failed');
  }
  setOverallProgress(msg.overall_progress ?? pipelineUI.getOverallProgress());
  setMeshPostToolbarVisible(false);
  if (!_meshPostInFlight) {
    setMeshPostStatus('');
  }
  setMeshRepairToolbarVisible(false);
  if (!_meshRepairInFlight) {
    setMeshRepairStatus('');
  }
  _meshRepairCandidateCount = 0;
  _meshRepairInFlight = false;
  if (Number(msg.stage) === 5 && _meshMethod === 'poisson') {
    setMeshPhaseStatus(`Stage 5 error: ${msg.error}`, 'error');
  }
  appendLog('stderr', `\nPipeline error at stage ${msg.stage}: ${msg.error}\n`, { stage: msg.stage });
});

ws.on('log', (msg) => {
  appendLog(msg.stream, msg.text, { stage: msg.stage });
});

ws.on('_open', () => {
  appendLog('stdout', '[Dashboard connected]\n');
});

ws.on('_close', () => {
  appendLog('stderr', '[Dashboard disconnected — reconnecting...]\n');
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

function normalizeMeshMethod(value) {
  const method = String(value || '').trim().toLowerCase();
  return MESH_METHOD_SET.has(method) ? method : MESH_METHOD_DEFAULT;
}

function isPipelineRunning() {
  return _latestStatusSnapshot?.running === true || statusBadge?.classList.contains('badge-running');
}

function setMeshPhaseStatus(message, tone = '') {
  if (!meshPhaseStatusBar || !meshPhaseStatusText) return;
  const text = String(message || '').trim();
  meshPhaseStatusBar.classList.remove('live', 'ready', 'error', 'hidden');
  if (!text) {
    meshPhaseStatusText.textContent = '';
    meshPhaseStatusBar.classList.add('hidden');
    return;
  }
  meshPhaseStatusText.textContent = text;
  if (tone === 'live' || tone === 'ready' || tone === 'error') {
    meshPhaseStatusBar.classList.add(tone);
  }
}

function bindMeshMethodPills() {
  const diffcdPill = meshMethodPills.diffcd;
  const poissonPill = meshMethodPills.poisson;

  diffcdPill?.addEventListener('click', (event) => {
    if (isPipelineRunning()) {
      event.preventDefault();
      return;
    }
    applyMeshMethod('diffcd');
    stageCtrl.activateStage(5);
  });

  const choosePoisson = (event) => {
    if (isPipelineRunning()) {
      if (_meshMethod !== 'poisson') {
        event.preventDefault();
        return;
      }
      stageCtrl.activateStage(5);
      return;
    }

    applyMeshMethod('poisson');
    stageCtrl.activateStage(5);
  };
  poissonPill?.addEventListener('click', (event) => {
    choosePoisson(event);
  });
  poissonPill?.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      choosePoisson(event);
    }
  });
}

function applyMeshMethod(method, opts = {}) {
  const resolved = normalizeMeshMethod(method);
  const changed = resolved !== _meshMethod;
  _meshMethod = resolved;

  checkpoints.setMeshMethod(resolved);
  pipelineUI.setMeshMethod(resolved);
  config.setMeshMethod?.(resolved);
  if (resolved !== 'poisson') {
    setMeshPhaseStatus('', '');
  }

  if (changed && opts.announce !== false) {
    const label = resolved === 'diffcd'
      ? 'Learning Mesh (DiffCD)'
      : 'Classical Mesh';
    appendLog('stdout', `Mesh method switched to: ${label}\n`, { stage: 5 });
  }
}

function initMeshPostToolbar() {
  if (!meshPostToolbar) return;
  setMeshPostToolbarVisible(false);
  setMeshPostEnabled(false);
  setMeshPostStatus('');

  meshPostApplyBtn?.addEventListener('click', () => {
    applyMeshPostprocess({ resetToRaw: false });
  });
  meshPostResetBtn?.addEventListener('click', () => {
    applyMeshPostprocess({ resetToRaw: true });
  });
}

function setMeshPostToolbarVisible(visible) {
  if (!meshPostToolbar) return;
  meshPostToolbar.style.display = visible ? 'flex' : 'none';
}

function setMeshPostEnabled(enabled) {
  const disabled = !enabled;
  if (meshPostMethodInput) meshPostMethodInput.disabled = disabled;
  if (meshPostIterationsInput) meshPostIterationsInput.disabled = disabled;
  if (meshPostLambdaInput) meshPostLambdaInput.disabled = disabled;
  if (meshPostDownsampleEnabledInput) meshPostDownsampleEnabledInput.disabled = disabled;
  if (meshPostTargetFacesInput) meshPostTargetFacesInput.disabled = disabled;
  if (meshPostTriggerFacesInput) meshPostTriggerFacesInput.disabled = disabled;
  if (meshPostApplyBtn) meshPostApplyBtn.disabled = disabled;
  if (meshPostResetBtn) meshPostResetBtn.disabled = disabled;
}

function setMeshPostStatus(message, tone = '') {
  if (!meshPostStatus) return;
  meshPostStatus.textContent = String(message || '');
  meshPostStatus.classList.remove('error', 'success');
  if (tone === 'error' || tone === 'success') {
    meshPostStatus.classList.add(tone);
  }
}

function initMeshRepairToolbar() {
  if (!meshRepairToolbar) return;
  setMeshRepairToolbarVisible(false);
  setMeshRepairEnabled(false);
  setMeshRepairStatus('');
  meshRepairClearBtn?.addEventListener('click', () => {
    preview.clearMeshRepairSelection();
    updateMeshRepairStatus();
  });
  meshRepairApplyBtn?.addEventListener('click', () => {
    applyMeshRepairSelection();
  });
}

function setMeshRepairToolbarVisible(visible) {
  if (!meshRepairToolbar) return;
  meshRepairToolbar.style.display = visible ? 'flex' : 'none';
}

function setMeshRepairEnabled(enabled) {
  const disabled = !enabled;
  if (meshRepairClearBtn) meshRepairClearBtn.disabled = disabled;
  if (meshRepairApplyBtn) meshRepairApplyBtn.disabled = disabled;
}

function setMeshRepairStatus(message, tone = '') {
  if (!meshRepairStatus) return;
  meshRepairStatus.textContent = String(message || '');
  meshRepairStatus.classList.remove('error', 'success');
  if (tone === 'error' || tone === 'success') {
    meshRepairStatus.classList.add(tone);
  }
}

function meshRepairCountLabel(selectedCount, candidateCount = _meshRepairCandidateCount) {
  const selected = Math.max(0, Number(selectedCount) || 0);
  const candidates = Math.max(0, Number(candidateCount) || 0);
  return `Closed surfaces: ${selected}/${candidates} selected.`;
}

function updateMeshRepairStatus(selectedIds = null) {
  if (!meshRepairToolbar || meshRepairToolbar.style.display === 'none') return;
  const selected = Array.isArray(selectedIds)
    ? selectedIds.length
    : preview.getMeshRepairSelectedLoopIds().length;
  setMeshRepairStatus(meshRepairCountLabel(selected, _meshRepairCandidateCount));
}

async function activateMeshRepairSelectionFromApi() {
  const res = await fetch('/api/mesh-repair/candidates', { cache: 'no-store' });
  const data = await res.json();
  if (!res.ok || data.error) {
    throw new Error(data.error || `HTTP ${res.status}`);
  }
  await preview.beginMeshRepairSelection(data);
  _meshRepairCandidateCount = Number(data.loop_count) || 0;
  _meshRepairInFlight = false;
  setMeshRepairToolbarVisible(true);
  setMeshRepairEnabled(true);
  updateMeshRepairStatus([]);
}

async function applyMeshRepairSelection() {
  if (_meshRepairInFlight) return;
  const selectedLoopIds = preview.getMeshRepairSelectedLoopIds();
  const selectedCount = Array.isArray(selectedLoopIds) ? selectedLoopIds.length : 0;
  const candidateCount = _meshRepairCandidateCount;

  _meshRepairInFlight = true;
  setMeshRepairEnabled(false);
  setMeshRepairStatus(
    selectedCount > 0
      ? `${meshRepairCountLabel(selectedCount, candidateCount)} Confirming repair selection...`
      : `${meshRepairCountLabel(0, candidateCount)} No surface selected. Confirming skip...`,
  );
  try {
    const res = await fetch('/api/mesh-repair/confirm', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ selected_loop_ids: selectedLoopIds || [] }),
    });
    const data = await res.json();
    if (!res.ok || data.error) {
      throw new Error(data.error || `HTTP ${res.status}`);
    }
    const confirmedCountRaw = Number(data.selected_count);
    const confirmedCount = Number.isFinite(confirmedCountRaw) ? confirmedCountRaw : selectedCount;
    preview.setMeshRepairConfirmed();
    if (confirmedCount > 0) {
      setMeshRepairStatus(
        `${meshRepairCountLabel(confirmedCount, candidateCount)} Repairing mesh...`,
        'success',
      );
      appendLog('stdout', `Mesh Repair confirmed with ${confirmedCount} selected loops.\n`, { stage: 7 });
    } else {
      setMeshRepairStatus(
        `${meshRepairCountLabel(0, candidateCount)} Skipping repair and continuing...`,
        'success',
      );
      appendLog('stdout', 'Mesh Repair confirmed with 0 selected loops. Skipping repair.\n', { stage: 7 });
    }
  } catch (e) {
    _meshRepairInFlight = false;
    setMeshRepairEnabled(true);
    setMeshRepairStatus(`Failed to confirm repair selection: ${e.message}`, 'error');
    appendLog('stderr', `Mesh Repair confirm failed: ${e.message}\n`, { stage: 7 });
  }
}

function canRunMeshPostprocess(statusMsg) {
  if (!statusMsg) return true;
  if (!statusMsg.running) return true;
  const next = statusMsg.next_stage_confirmation || {};
  return next.required === true && Number(next.from_stage) === 5;
}

function syncMeshPostToolbarFromStatus(statusMsg) {
  if (!statusMsg) return;
  const ready = Boolean(statusMsg?.object_name) && isStageDone(statusMsg, 5);
  setMeshPostToolbarVisible(ready);
  if (!ready) {
    setMeshPostEnabled(false);
    if (!_meshPostInFlight) {
      setMeshPostStatus('');
    }
    return;
  }

  const allowed = canRunMeshPostprocess(statusMsg);
  setMeshPostEnabled(allowed && !_meshPostInFlight);
  if (_meshPostInFlight) return;

  if (!allowed) {
    setMeshPostStatus('Mesh post-process is disabled while pipeline is running.');
    return;
  }

  if (meshPostStatus?.textContent === 'Mesh post-process is disabled while pipeline is running.') {
    setMeshPostStatus('');
  }
}

function syncMeshRepairToolbarFromStatus(statusMsg) {
  const stage7 = getStageInfo(statusMsg, 7);
  const ready = statusMsg?.running === true
    && stage7?.status === 'interactive'
    && statusMsg?.mesh_repair?.ready === true;
  if (!ready) {
    if (!_meshRepairInFlight) {
      setMeshRepairToolbarVisible(false);
      setMeshRepairStatus('');
    }
    return;
  }
  if (_meshRepairInFlight) return;
  setMeshRepairToolbarVisible(true);
  setMeshRepairEnabled(true);
  const selected = Number(statusMsg?.mesh_repair?.selected_loop_count) || 0;
  const candidateCount = Number(statusMsg?.mesh_repair?.candidate_count) || _meshRepairCandidateCount;
  _meshRepairCandidateCount = candidateCount;
  setMeshRepairStatus(meshRepairCountLabel(selected, candidateCount));
}

async function applyMeshPostprocess({ resetToRaw = false } = {}) {
  if (_meshPostInFlight) return;

  const method = resetToRaw ? 'laplacian' : (meshPostMethodInput?.value || 'laplacian');
  const iterationsRaw = Number.parseInt(meshPostIterationsInput?.value || '8', 10);
  const iterations = resetToRaw ? 0 : Math.max(0, Math.min(100, Number.isFinite(iterationsRaw) ? iterationsRaw : 8));
  const lambRaw = Number.parseFloat(meshPostLambdaInput?.value || '0.5');
  const lamb = Math.max(0.01, Math.min(1.5, Number.isFinite(lambRaw) ? lambRaw : 0.5));
  const downsampleEnabled = meshPostDownsampleEnabledInput?.checked ?? true;
  const targetRaw = Number.parseInt(meshPostTargetFacesInput?.value || '120000', 10);
  const downsampleTargetFaces = Math.max(
    1000,
    Math.min(5_000_000, Number.isFinite(targetRaw) ? targetRaw : 120000),
  );
  const triggerRaw = Number.parseInt(meshPostTriggerFacesInput?.value || '170000', 10);
  const downsampleTriggerFaces = Math.max(
    downsampleTargetFaces,
    Math.min(5_000_000, Number.isFinite(triggerRaw) ? triggerRaw : 170000),
  );

  if (meshPostTargetFacesInput) meshPostTargetFacesInput.value = String(downsampleTargetFaces);
  if (meshPostTriggerFacesInput) meshPostTriggerFacesInput.value = String(downsampleTriggerFaces);

  _meshPostInFlight = true;
  setMeshPostEnabled(false);
  setMeshPostStatus(resetToRaw ? 'Resetting to raw mesh...' : 'Applying smoothing...');

  try {
    const res = await fetch('/api/mesh/postprocess', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        method,
        iterations,
        lamb,
        taubin_nu: DEFAULT_TAUBIN_NU,
        downsample_enabled: downsampleEnabled,
        downsample_target_faces: downsampleTargetFaces,
        downsample_trigger_faces: downsampleTriggerFaces,
        source: 'raw',
        invalidate_texture: true,
      }),
    });
    const data = await res.json();
    if (!res.ok || data.error) {
      throw new Error(data.error || `HTTP ${res.status}`);
    }

    await preview.loadStageResult(5, { preferPreview: _meshMethod === 'poisson' });
    if (_meshMethod === 'poisson') {
      setMeshPhaseStatus(`Showing: ${CLASSICAL_PREVIEW_TITLE}`, 'ready');
    }
    stageCtrl.activateStage(5);

    const vertices = Number(data.vertices) || 0;
    const faces = Number(data.faces) || 0;
    const apiDownsampleEnabled = data.downsample_enabled !== false;
    const apiTargetFaces = Number(data.downsample_target_faces) || downsampleTargetFaces;
    const apiTriggerFaces = Number(data.downsample_trigger_faces) || downsampleTriggerFaces;
    if (meshPostDownsampleEnabledInput) {
      meshPostDownsampleEnabledInput.checked = apiDownsampleEnabled;
    }
    if (meshPostTargetFacesInput) {
      meshPostTargetFacesInput.value = String(apiTargetFaces);
    }
    if (meshPostTriggerFacesInput) {
      meshPostTriggerFacesInput.value = String(apiTriggerFaces);
    }
    let msg = resetToRaw
      ? `Reset complete (${vertices.toLocaleString()}v / ${faces.toLocaleString()}f).`
      : `Smoothing complete (${vertices.toLocaleString()}v / ${faces.toLocaleString()}f).`;
    if (!apiDownsampleEnabled) {
      msg += ' Downsample disabled.';
    } else if (data.downsample_applied) {
      msg += ' Downsample applied.';
    } else {
      msg += ` Downsample skipped (target=${apiTargetFaces.toLocaleString()}, trigger=${apiTriggerFaces.toLocaleString()}).`;
    }
    if (data.texture_invalidated) {
      msg += ' Downstream Stage 6/7/8 artifacts were cleared.';
    }
    setMeshPostStatus(msg, 'success');

    appendLog(
      'stdout',
      `Mesh post-process: method=${data.method}, iterations=${data.iterations}, lambda=${Number(data.lamb).toFixed(2)}, source=${data.source}, downsample_enabled=${apiDownsampleEnabled ? 'yes' : 'no'}, target_faces=${apiTargetFaces}, trigger_faces=${apiTriggerFaces}, applied=${data.downsample_applied ? 'yes' : 'no'}\n`,
      { stage: 5 },
    );
    if (data.texture_invalidated) {
      appendLog('stdout', 'Wrap/repair/texture outputs removed. Re-run from Stage 6 for updated texturing.\n', { stage: 6 });
    }

    if (!isPipelineRunning()) {
      _hydratedStatusKey = '';
      try {
        const statusRes = await fetch('/api/pipeline/status', { cache: 'no-store' });
        if (statusRes.ok) {
          const status = await statusRes.json();
          await applyStatusSnapshot(status, { forceHydrate: true });
          stageCtrl.activateStage(5);
        }
      } catch (statusErr) {
        console.warn('Failed to refresh status after mesh post-process:', statusErr);
      }
    }
  } catch (e) {
    setMeshPostStatus(`Failed: ${e.message}`, 'error');
    appendLog('stderr', `Mesh post-process failed: ${e.message}\n`, { stage: 5 });
  } finally {
    _meshPostInFlight = false;
    syncMeshPostToolbarFromStatus(_latestStatusSnapshot);
  }
}

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
  if (Number.isFinite(current) && current >= 1 && current <= STAGE_COUNT) return current;
  for (let stage = STAGE_COUNT; stage >= 1; stage--) {
    if (isStageDone(statusMsg, stage)) return stage;
  }
  return 1;
}

function buildStatusKey(statusMsg) {
  const objectName = statusMsg?.object_name || '';
  const next = statusMsg?.next_stage_confirmation || {};
  const stageState = [];
  for (let i = 1; i <= STAGE_COUNT; i++) {
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
  for (let i = 1; i <= STAGE_COUNT; i++) {
    const info = getStageInfo(statusMsg, i);
    stageCtrl.setStageState(i, info?.status || 'pending');
  }
}

function resolveTransitionTarget(stage, toStage = null) {
  const parsed = Number(toStage);
  if (Number.isFinite(parsed) && parsed >= 1 && parsed <= STAGE_COUNT) {
    return parsed;
  }
  return Math.max(1, Math.min(STAGE_COUNT, Number(stage) + 1));
}

function defaultTaskConfirmIdleMessage(stage) {
  const target = resolveTransitionTarget(stage);
  return `Stage ${stage} confirmation will appear after completion (next: Stage ${target}).`;
}

function defaultTaskConfirmWaitingMessage(stage, toStage = null) {
  const target = resolveTransitionTarget(stage, toStage);
  if (target === Number(stage)) {
    return `Stage ${stage} step complete. Confirm to continue within Stage ${stage}.`;
  }
  return `Stage ${stage} complete. Confirm to continue to Stage ${target}.`;
}

function defaultTaskConfirmConfirmedMessage(stage, toStage = null) {
  const target = resolveTransitionTarget(stage, toStage);
  if (target === Number(stage)) {
    return `Stage ${stage} step confirmed. Continuing Stage ${stage} workflow.`;
  }
  return `Stage ${stage} confirmed. Proceeded to Stage ${target}.`;
}

function defaultTaskConfirmStandbyMessage(stage, toStage = null) {
  const target = resolveTransitionTarget(stage, toStage);
  if (target === Number(stage)) {
    return `Stage ${stage} is complete. Start the pipeline to continue remaining Stage ${stage} tasks.`;
  }
  return `Stage ${stage} is complete. Start the pipeline to continue to Stage ${target}.`;
}

function resolveTaskConfirmVisibleStage(preferredStage) {
  const waiting = Number(_waitingConfirmationStage);
  if (Number.isFinite(waiting) && waiting >= 1 && waiting <= TRANSITION_STAGE_MAX) {
    return waiting;
  }

  const preferred = Number(preferredStage);
  if (Number.isFinite(preferred) && preferred >= 1 && preferred <= STAGE_COUNT) {
    return preferred;
  }

  const active = Number(stageCtrl.activeStage);
  if (Number.isFinite(active) && active >= 1 && active <= STAGE_COUNT) {
    return active;
  }

  return 1;
}

function setTaskConfirmVisibleStage(preferredStage = null) {
  const visibleStage = resolveTaskConfirmVisibleStage(preferredStage);
  for (let stage = 1; stage <= STAGE_COUNT; stage++) {
    const bar = _taskConfirmBars[stage];
    if (!bar) continue;
    bar.classList.toggle('is-visible', stage === visibleStage);
  }
}

function resolveTaskConfirmIdleMessage(statusMsg, stage) {
  const info = getStageInfo(statusMsg, stage);
  if (!info) return defaultTaskConfirmIdleMessage(stage);

  const status = String(info.status || 'pending');
  if (status === 'running') {
    return `Stage ${stage} is running. Confirmation will appear after completion.`;
  }
  if (status === 'failed') {
    return `Stage ${stage} failed. Resolve the error and rerun this stage.`;
  }
  if (status === 'interactive') {
    return `Stage ${stage} is waiting for required interaction.`;
  }
  return defaultTaskConfirmIdleMessage(stage);
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
  _waitingConfirmationToStage = null;
  for (let stage = 1; stage <= TRANSITION_STAGE_MAX; stage++) {
    if (stage < resumeStage) {
      setTaskConfirmState(stage, 'confirmed', `Stage ${stage} already completed before resume.`);
    } else {
      setTaskConfirmState(stage, 'idle', defaultTaskConfirmIdleMessage(stage));
    }
  }
  setTaskConfirmState(8, 'final', 'Final stage. No next-stage confirmation.');
  setTaskConfirmVisibleStage(resumeStage);
}

function isTransitionConfirmed(statusMsg, stage) {
  if (!statusMsg || stage < 1 || stage > TRANSITION_STAGE_MAX) return false;
  if (!isStageDone(statusMsg, stage)) return false;

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
  const waitingToStage = next.required === true ? Number(next.to_stage) : NaN;
  const waitingStageIsValid = Number.isFinite(waitingStage)
    && waitingStage >= 1
    && waitingStage <= TRANSITION_STAGE_MAX
    && statusMsg.running === true
    && isStageDone(statusMsg, waitingStage)
    && !isTransitionConfirmed(statusMsg, waitingStage);
  _waitingConfirmationStage = waitingStageIsValid ? waitingStage : null;
  _waitingConfirmationToStage = waitingStageIsValid && Number.isFinite(waitingToStage)
    ? waitingToStage
    : null;

  const resumeStage = Math.max(1, Math.min(STAGE_COUNT, Number(statusMsg.resume_from_stage) || 1));

  for (let stage = 1; stage <= TRANSITION_STAGE_MAX; stage++) {
    const stageDone = isStageDone(statusMsg, stage);

    if (_waitingConfirmationStage === stage) {
      setTaskConfirmState(
        stage,
        'waiting',
        String(next.message || defaultTaskConfirmWaitingMessage(stage, _waitingConfirmationToStage)),
      );
      continue;
    }

    if (isTransitionConfirmed(statusMsg, stage)) {
      setTaskConfirmState(stage, 'confirmed', defaultTaskConfirmConfirmedMessage(stage));
      continue;
    }

    if (stage < resumeStage && stageDone) {
      setTaskConfirmState(stage, 'confirmed', `Stage ${stage} already completed before resume.`);
      continue;
    }

    if (!statusMsg.running && stageDone) {
      setTaskConfirmState(stage, 'idle', defaultTaskConfirmStandbyMessage(stage));
      continue;
    }

    setTaskConfirmState(stage, 'idle', resolveTaskConfirmIdleMessage(statusMsg, stage));
  }

  if (isStageDone(statusMsg, 8)) {
    setTaskConfirmState(8, 'final', 'Final stage complete. No next-stage confirmation.');
  } else {
    setTaskConfirmState(8, 'final', 'Final stage. No next-stage confirmation.');
  }
  setTaskConfirmVisibleStage(resolvePreferredStage(statusMsg));
}

async function confirmNextStage(stage) {
  if (_waitingConfirmationStage !== stage) return;

  const toStage = _waitingConfirmationToStage;
  const target = resolveTransitionTarget(stage, toStage);
  const waitingMsg = target === Number(stage)
    ? `Stage ${stage} confirmed. Waiting for the next Stage ${stage} step...`
    : `Stage ${stage} confirmed. Waiting for Stage ${target} start...`;
  setTaskConfirmState(stage, 'sending', waitingMsg);

  try {
    const res = await fetch('/api/pipeline/confirm-next', { method: 'POST' });
    const data = await res.json();
    if (!res.ok || data.error) {
      throw new Error(data.error || `HTTP ${res.status}`);
    }

    if (data.status === 'no_waiting_confirmation') {
      _waitingConfirmationStage = null;
      _waitingConfirmationToStage = null;
      setTaskConfirmState(stage, 'confirmed', defaultTaskConfirmConfirmedMessage(stage, toStage));
      setTaskConfirmVisibleStage(stageCtrl.activeStage);
      return;
    }

    // Keep "sending" state until backend emits stage start or cleared event.
  } catch (e) {
    setTaskConfirmState(stage, 'waiting', `${defaultTaskConfirmWaitingMessage(stage, toStage)} (${e.message})`);
    appendLog('stderr', `Next-stage confirmation failed at stage ${stage}: ${e.message}\n`, { stage });
  }
}

async function hydrateOutputsFromStatus(statusMsg, opts = {}) {
  if (!statusMsg?.object_name) return;
  const statusMeshMethod = normalizeMeshMethod(statusMsg?.mesh_method || _meshMethod);
  const isPoissonMesh = statusMeshMethod === 'poisson';

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
  if (isStageDone(statusMsg, 5)) {
    await preview.loadStageResult(5, { preferPreview: isPoissonMesh });
    if (isPoissonMesh) {
      setMeshPhaseStatus(`Showing: ${CLASSICAL_PREVIEW_TITLE}`, 'ready');
    } else {
      setMeshPhaseStatus('', '');
    }
  } else if (statusMsg.running && Number(statusMsg.current_stage) === 5 && isPoissonMesh) {
    setMeshPhaseStatus('Running: Classical Mesh', 'live');
  }
  if (isStageDone(statusMsg, 5) && !isStageDone(statusMsg, 6)) {
    await preview.showCropBbox(config.getCropScale(), { preferPreview: isPoissonMesh });
  }
  if (isStageDone(statusMsg, 6)) await preview.loadStageResult(6);
  if (isStageDone(statusMsg, 7)) await preview.loadStageResult(7);
  if (isStageDone(statusMsg, 8)) await preview.loadStageResult(8);

  if (opts.activate !== false) {
    stageCtrl.activateStage(resolvePreferredStage(statusMsg));
  }
}

async function applyStatusSnapshot(statusMsg, opts = {}) {
  _latestStatusSnapshot = statusMsg;
  checkpoints.applyStatusSnapshot(statusMsg);
  applyMeshMethod(statusMsg?.mesh_method || _meshMethod, { announce: false });
  if (normalizeMeshMethod(statusMsg?.mesh_method || _meshMethod) !== 'poisson') {
    setMeshPhaseStatus('', '');
  }
  pipelineUI.setMeshMethodEnabled(statusMsg?.running !== true);
  pipelineUI.updateAll(statusMsg);
  syncStageStates(statusMsg);
  syncTaskConfirmBarsFromStatus(statusMsg);
  syncMeshPostToolbarFromStatus(statusMsg);
  syncMeshRepairToolbarFromStatus(statusMsg);
  setOverallProgress(statusMsg.overall_progress ?? pipelineUI.getOverallProgress());

  if (statusMsg.object_name) {
    config.setObjectName(statusMsg.object_name);
  }

  if (statusMsg.running) {
    config.setRunning(true);
    setStatus('running', 'Running');
    const waiting = statusMsg?.next_stage_confirmation?.required === true;
    const waitingFromStage = Number(statusMsg?.next_stage_confirmation?.from_stage);
    const meshRepairInteractive = statusMsg?.mesh_repair?.ready === true
      && getStageInfo(statusMsg, 7)?.status === 'interactive';
    if (waiting) {
      await hydrateOutputsFromStatus(statusMsg, {
        force: opts.forceHydrate === true,
        activate: false,
      });
      if (Number.isFinite(waitingFromStage) && waitingFromStage >= 1 && waitingFromStage <= STAGE_COUNT) {
        stageCtrl.activateStage(waitingFromStage);
      } else {
        stageCtrl.activateStage(resolvePreferredStage(statusMsg));
      }
    } else if (meshRepairInteractive) {
      await hydrateOutputsFromStatus(statusMsg, {
        force: opts.forceHydrate === true,
        activate: false,
      });
      try {
        await activateMeshRepairSelectionFromApi();
      } catch (e) {
        setMeshRepairToolbarVisible(false);
        setMeshRepairStatus(`Mesh Repair UI failed: ${e.message}`, 'error');
      }
      stageCtrl.activateStage(7);
    } else {
      stageCtrl.activateStage(resolvePreferredStage(statusMsg));
    }
    return;
  }

  config.setRunning(false);
  config.setActiveStage(null);
  _meshRepairInFlight = false;
  setMeshRepairToolbarVisible(false);
  setMeshRepairStatus('');

  const allDone = [1, 2, 3, 4, 5, 6, 7, 8].every((stage) => isStageDone(statusMsg, stage));
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
