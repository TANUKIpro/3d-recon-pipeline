/**
 * Main dashboard controller — wires WebSocket, pipeline UI, SAM2 canvas,
 * config panel, log viewer, 3D preview, and stage-panel switching together.
 */

import { I18n } from './i18n.js';
import { SettingsPanel } from './settings-panel.js';
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
import { TaskConfirmController } from './task-confirm-controller.js';
import { MeshRepairController } from './mesh-repair-controller.js';
import { MeshPostController } from './mesh-post-controller.js';
import { StatusHydrator } from './status-hydrator.js';
import {
  STAGE_COUNT,
  TRANSITION_STAGE_MAX,
  MESH_METHOD_DEFAULT,
  CLASSICAL_PREVIEW_TITLE,
} from './constants.js';
import {
  formatTime,
  normalizeMeshMethod,
} from './utils.js';
import {
  defaultTaskConfirmConfirmedMessage,
  defaultTaskConfirmWaitingMessage,
} from './pipeline-status.js';

// ── Settings & i18n (before other modules so theme/lang apply early) ────

const i18n = new I18n();
const settings = new SettingsPanel(i18n);
i18n.apply();

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

// Wire settings callbacks
settings.onLogSettingsChanged = ({ autoScroll, maxLines }) => {
  log.setAutoScroll(autoScroll);
  log.setMaxLines(maxLines);
};
settings.onThemeChanged = (theme) => { preview.applyTheme(theme); };
preview.applyTheme(settings.theme);

// Apply stored log settings at startup
log.setAutoScroll(settings.autoScroll);
log.setMaxLines(settings.maxLines);

function appendLog(stream, text, opts = {}) {
  const stage = Number(opts.stage);
  log.append(stream, text, {
    ...opts,
    stage: Number.isFinite(stage) ? stage : stageCtrl.activeStage,
  });
}

// ── DOM references ───────────────────────────────────────────

const statusBadge = document.getElementById('status-badge');
const vramBadge = document.getElementById('vram-badge');
const overallProgressBadge = document.getElementById('overall-progress-badge');
const meshPhaseStatusBar = document.getElementById('mesh-phase-status');
const meshPhaseStatusText = document.getElementById('mesh-phase-status-text');
const meshMethodPills = {
  diffcd: document.getElementById('mesh-pill-diffcd'),
  poisson: document.getElementById('mesh-pill-poisson'),
};

// ── State ────────────────────────────────────────────────────

let _extractedFrameCount = 0;
let _objectLoadRequestId = 0;
let _meshMethod = MESH_METHOD_DEFAULT;

// ── Init controllers ─────────────────────────────────────────

const taskConfirm = new TaskConfirmController({ stageCtrl, appendLog });

const meshRepair = new MeshRepairController({ preview, appendLog });
meshRepair.init();
preview.onMeshRepairSelectionChanged = (selectedIds) => {
  meshRepair.updateStatus(selectedIds);
};

const meshPost = new MeshPostController({
  preview,
  stageCtrl,
  appendLog,
  getMeshMethod: () => _meshMethod,
  isPipelineRunning,
  setMeshPhaseStatus,
  applyStatusSnapshot: (status, opts) => statusHydrator.applySnapshot(status, opts),
});
meshPost.init();
meshPost.onPostprocessComplete = () => {
  meshPost.syncFromStatus(statusHydrator.latestSnapshot);
};

const statusHydrator = new StatusHydrator({
  preview, cameraOverlay, sam2, sam2Verify, config, stageCtrl,
  checkpoints, pipelineUI, taskConfirm, meshPost, meshRepair,
  getMeshMethod: () => _meshMethod,
  applyMeshMethod,
  setMeshPhaseStatus,
  setStatus,
  setOverallProgress,
});

bindMeshMethodPills();
applyMeshMethod(config.getMeshMethod?.() || MESH_METHOD_DEFAULT, { announce: false });

config.onCropScaleChanged = (cropScale) => {
  preview.updateCropBbox(cropScale);
};

// ── Stage-activated event: lazy-init 3D ─────────────────────

document.addEventListener('stage-activated', async (e) => {
  const stage = e.detail.stage;

  config.setActiveStage(stage);
  checkpoints.setActiveStage(stage);

  if (stage === 2 || (stage >= 4 && stage <= 8)) {
    if (preview._stages?.[stage]?.initialized) {
      preview.activateStage(stage);
    }
  }

  if (stage === 6) {
    const s5 = stageCtrl.getStageState(5);
    const s6 = stageCtrl.getStageState(6);
    if (s5 === 'complete' && s6 !== 'complete' && s6 !== 'interactive') {
      await preview.showCropBbox(config.getCropScale(), { preferPreview: _meshMethod === 'poisson' });
    }
  }

  taskConfirm.setVisibleStage(stage);
});

stageCtrl.activateStage(stageCtrl.activeStage);

// ── Helpers ──────────────────────────────────────────────────

function isPipelineRunning() {
  return statusHydrator.latestSnapshot?.running === true || statusBadge?.classList.contains('badge-running');
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

// ── Config panel callbacks ───────────────────────────────────

config.onObjectSelected = async (objectName) => {
  const reqId = ++_objectLoadRequestId;
  if (!objectName) {
    statusHydrator.reset();
    cameraOverlay.remove();
    sam2.deactivate();
    sam2Verify.hide();
    preview.reset();
    pipelineUI.reset();
    taskConfirm.resetBars();
    meshPost.setToolbarVisible(false);
    meshPost.setStatus('');
    meshRepair.setToolbarVisible(false);
    meshRepair.setStatus('');
    meshRepair.resetState({ resetThreshold: true });
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
      await statusHydrator.applySnapshot(status, { forceHydrate: true });
      appendLog('stdout', `Loaded object: ${objectName}\n`);
    }
  } catch (e) {
    if (reqId !== _objectLoadRequestId) return;
    appendLog('stderr', `Failed to load object ${objectName}: ${e.message}\n`);
  }
};

config.onStart = async (cfg) => {
  try {
    _objectLoadRequestId += 1;
    cfg.auto_accept = settings.autoAccept;
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
    taskConfirm.resetBars(resumeFromStage);
    pipelineUI.resetFromStage(resumeFromStage);
    checkpoints.reset(resumeFromStage);
    _extractedFrameCount = 0;
    statusHydrator.reset();
    meshPost.setToolbarVisible(false);
    meshPost.setStatus('');
    meshRepair.setToolbarVisible(false);
    meshRepair.setStatus('');
    meshRepair.resetState({ resetThreshold: true });
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
  await statusHydrator.applySnapshot(msg);
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

    const waitingToStage = taskConfirm.waitingToStage;
    if (taskConfirm.waitingStage === prevStage) {
      taskConfirm.waitingStage = null;
      taskConfirm.waitingToStage = null;
    }
    taskConfirm.setState(
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
  meshPost.setToolbarVisible(false);
  if (!meshPost.inFlight) {
    meshPost.setStatus('');
  }
  if (Number(msg.stage) === 7) {
    meshRepair.resetState({ resetThreshold: true });
  }
  if (Number(msg.stage) !== 7 || !meshRepair.inFlight) {
    meshRepair.setToolbarVisible(false);
    if (!meshRepair.inFlight) {
      meshRepair.setStatus('');
    }
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
      meshRepair.setToolbarVisible(false);
      meshRepair.setStatus(`Stage 7 failed: ${msg.error}`, 'error');
      meshRepair.resetState({ resetThreshold: true });
    }
    return;
  }
  checkpoints.onStageComplete(msg.stage);
  stageCtrl.setStageState(msg.stage, 'complete');

  if (msg.stage === 1) {
    const empty = document.querySelector('#stage-panel-1 .stage-panel-empty');
    if (empty) empty.classList.add('hidden');
    await preview.loadGallery(_extractedFrameCount);
  } else if (msg.stage === 2) {
    await preview.loadPi3xResults(cameraOverlay);
  } else if (msg.stage === 3) {
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
    meshPost.setToolbarVisible(true);
    meshPost.setEnabled(true);
    if (!meshPost.inFlight) {
      meshPost.setStatus('');
    }
  }
  if (msg.stage === 7) {
    meshRepair.resetState({ resetThreshold: true });
    meshRepair.setToolbarVisible(false);
    meshRepair.setStatus('');
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
  const autoAccepted = msg.auto_accepted === true;
  const loopCount = Number(msg.candidate_count) || 0;
  if (loopCount === 0) {
    const prefix = autoAccepted ? '[Auto] ' : '';
    appendLog(
      'warn',
      `${prefix}Mesh Repair: no closed surfaces found — skipped.\n`,
      { stage: 7 },
    );
    return;
  }
  if (autoAccepted) {
    appendLog(
      'stdout',
      `[Auto] Mesh Repair: auto-selected all ${loopCount} closed surfaces.\n`,
      { stage: 7 },
    );
    return;
  }
  checkpoints.onStageInteractive(7, 'Waiting for repair-loop selection');
  pipelineUI.stageInteractive(7);
  stageCtrl.setStageState(7, 'interactive');
  try {
    await meshRepair.activateFromApi();
    stageCtrl.activateStage(7);
    const loopCount = Number(msg.candidate_count) || meshRepair.candidateCount;
    appendLog(
      'stdout',
      `Mesh Repair ready: ${loopCount} selectable loops. Click closed surfaces, then confirm selection (0 selected = skip).\n`,
      { stage: 7 },
    );
  } catch (e) {
    meshRepair.setToolbarVisible(false);
    meshRepair.setStatus(`Mesh Repair UI failed: ${e.message}`, 'error');
    appendLog('stderr', `Mesh Repair setup failed: ${e.message}\n`, { stage: 7 });
  }
});

ws.on('next_stage_confirmation_required', (msg) => {
  const fromStage = Number(msg.from_stage);
  const toStage = Number(msg.to_stage);
  const autoAccepted = msg.auto_accepted === true;
  if (fromStage >= 1 && fromStage <= TRANSITION_STAGE_MAX) {
    checkpoints.onStageInteractive(fromStage, 'Waiting for next-stage confirmation');
    if (taskConfirm.waitingStage !== null && taskConfirm.waitingStage !== fromStage) {
      taskConfirm.setState(
        taskConfirm.waitingStage,
        'confirmed',
        defaultTaskConfirmConfirmedMessage(taskConfirm.waitingStage, taskConfirm.waitingToStage),
      );
    }
    if (autoAccepted) {
      taskConfirm.waitingStage = null;
      taskConfirm.waitingToStage = null;
      taskConfirm.setState(fromStage, 'sending', 'Auto-accepted');
    } else {
      taskConfirm.waitingStage = fromStage;
      taskConfirm.waitingToStage = Number.isFinite(toStage) ? toStage : null;
      pipelineUI.stageInteractive(fromStage);
      stageCtrl.setStageState(fromStage, 'interactive');
      stageCtrl.activateStage(fromStage);
      taskConfirm.setState(
        fromStage,
        'waiting',
        String(msg.message || defaultTaskConfirmWaitingMessage(fromStage, taskConfirm.waitingToStage)),
      );
      taskConfirm.setVisibleStage(fromStage);
    }
  }
  if (msg.message) {
    const prefix = autoAccepted ? '[Auto] ' : '';
    appendLog('stdout', `${prefix}${msg.message}\n`, { stage: fromStage });
  }
});

ws.on('next_stage_confirmation_cleared', (msg) => {
  const fromStage = Number(msg.from_stage);
  const toStage = Number(msg.to_stage);
  if (fromStage >= 1 && fromStage <= TRANSITION_STAGE_MAX) {
    if (taskConfirm.waitingStage === fromStage) {
      taskConfirm.waitingStage = null;
      taskConfirm.waitingToStage = null;
    }
    taskConfirm.setState(fromStage, 'confirmed', defaultTaskConfirmConfirmedMessage(fromStage, toStage));
    taskConfirm.setVisibleStage(stageCtrl.activeStage);
  }
});

ws.on('pipeline_complete', (msg) => {
  checkpoints.onStageComplete(8);
  config.setRunning(false);
  pipelineUI.setMeshMethodEnabled(true);
  config.setActiveStage(null);
  config.refreshObjects();
  taskConfirm.waitingStage = null;
  taskConfirm.waitingToStage = null;
  for (let stage = 1; stage <= TRANSITION_STAGE_MAX; stage++) {
    taskConfirm.setState(stage, 'confirmed', defaultTaskConfirmConfirmedMessage(stage));
  }
  taskConfirm.setState(8, 'final', 'Final stage complete. No next-stage confirmation.');
  taskConfirm.setVisibleStage(8);
  setOverallProgress(msg.overall_progress ?? 100);
  setStatus('complete', `Done (${formatTime(msg.elapsed)})`);
  meshPost.setToolbarVisible(true);
  meshPost.setEnabled(true);
  meshRepair.setToolbarVisible(false);
  meshRepair.setStatus('');
  meshRepair.resetState({ resetThreshold: true });
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
  if (taskConfirm.waitingStage !== null) {
    taskConfirm.setState(
      taskConfirm.waitingStage,
      'idle',
      `Stage ${taskConfirm.waitingStage} confirmation was interrupted.`,
    );
    taskConfirm.waitingStage = null;
    taskConfirm.waitingToStage = null;
  }
  taskConfirm.setVisibleStage(stageCtrl.activeStage);
  const cancelled = String(msg.reason_code || '').startsWith('cancelled')
    || /cancel/i.test(String(msg.error || ''));
  setStatus(cancelled ? 'idle' : 'error', cancelled ? 'Cancelled' : 'Error');
  if (msg.stage >= 1 && msg.stage <= 8) {
    checkpoints.onStageFailed(msg.stage, msg.error, msg.checkpoint_id);
    pipelineUI.stageFailed(msg.stage);
    stageCtrl.setStageState(msg.stage, 'failed');
  }
  setOverallProgress(msg.overall_progress ?? pipelineUI.getOverallProgress());
  meshPost.setToolbarVisible(false);
  if (!meshPost.inFlight) {
    meshPost.setStatus('');
  }
  meshRepair.setToolbarVisible(false);
  if (!meshRepair.inFlight) {
    meshRepair.setStatus('');
  }
  meshRepair.resetState({ resetThreshold: true });
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

// ── Boot ─────────────────────────────────────────────────────

ws.connect();
