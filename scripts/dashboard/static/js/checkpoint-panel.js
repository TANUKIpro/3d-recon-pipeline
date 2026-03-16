/**
 * Stage checkpoint viewer for the right panel.
 *
 * Shows fixed checkpoint templates per stage and maps stage detail text
 * to one running checkpoint.
 */

import { STAGE_COUNT } from './constants.js';

const STAGE_MIN = 1;
const STAGE_MAX = STAGE_COUNT;

const STAGE_LABELS = {
  1: 'Extract Frames',
  2: 'COLMAP SfM',
  3: 'SAM2',
  4: 'gs2mesh',
  5: 'Texture Bake',
  6: 'Post Cleanup',
};

const CHECKPOINT_TEMPLATES = {
  1: [
    'Inspect video metadata',
    'Extract frames',
    'Finalize frame set',
  ],
  2: [
    'Extract features (COLMAP)',
    'Match features',
    'Sparse reconstruction',
    'Export camera poses',
  ],
  3: [
    'Initialize SAM2 model',
    'Wait for interactive clicks',
    'Propagate masks',
    'Wait for mask verification',
  ],
  4: [
    'Train 3D Gaussian Splatting',
    'Stereo depth estimation',
    'TSDF fusion + mesh extraction',
    'Save output mesh',
  ],
  5: [
    'Load mesh',
    'Estimate camera intrinsics',
    'Generate UV atlas / texel mapping',
    'Score and project views',
    'Secondary fill and seam padding',
    'Export textured mesh',
  ],
  6: [
    'Generate cleanup proposal',
    'Wait for cleanup review',
    'Bottom hole-fill (skirt + cap)',
    'Post-holefill noise removal',
    'General hole-fill (watertight)',
    'Final component cleanup',
    'Write cleaned mesh',
  ],
};

const CHECKPOINT_IDS = {
  1: ['s1.inspect', 's1.extract', 's1.finalize'],
  2: ['s2.features', 's2.match', 's2.reconstruct', 's2.export'],
  3: ['s3.initialize', 's3.interact', 's3.propagate', 's3.verify'],
  4: ['s4.train_gs', 's4.stereo', 's4.tsdf', 's4.save'],
  5: ['s5.load', 's5.intrinsics', 's5.uv', 's5.score', 's5.fill', 's5.export'],
  6: ['s6.proposal', 's6.review', 's6.holefill', 's6.noise', 's6.watertight', 's6.finalclean', 's6.apply'],
};

const DETAIL_MATCHERS = {
  1: [
    { re: /inspect|metadata/i, idx: 0 },
    { re: /extracting frames|extract frames/i, idx: 1 },
    { re: /extracted\s+\d+\s+frames|finalize|complete/i, idx: 2 },
  ],
  2: [
    { re: /feature extraction|feature_extractor/i, idx: 0 },
    { re: /matcher|matching/i, idx: 1 },
    { re: /sparse reconstruction|mapper/i, idx: 2 },
    { re: /exporting camera poses|colmap sfm complete/i, idx: 3 },
  ],
  3: [
    { re: /initializing sam2 model|sam2 ready/i, idx: 0 },
    { re: /waiting for interactive clicks|redoing sam2 interaction/i, idx: 1 },
    { re: /propagating masks/i, idx: 2 },
    { re: /waiting for mask verification|verification/i, idx: 3 },
  ],
  4: [
    { re: /training 3d gaussian|3dgs training|iteration/i, idx: 0 },
    { re: /stereo depth|dlnr|running gs2mesh/i, idx: 1 },
    { re: /tsdf|fusion|mesh extraction|collecting output/i, idx: 2 },
    { re: /gs2mesh reconstruction complete|gs2mesh complete/i, idx: 3 },
  ],
  5: [
    { re: /loading mesh/i, idx: 0 },
    { re: /estimating camera intrinsics|intrinsics optimization complete|camera intrinsics estimated/i, idx: 1 },
    { re: /generating uv atlas|building texel mapping/i, idx: 2 },
    { re: /scoring camera views|scoring views|applying primary views|projecting primary chart textures/i, idx: 3 },
    { re: /secondary view search|padding uv seams|padding seams/i, idx: 4 },
    { re: /exporting textured mesh|texture stage complete/i, idx: 5 },
  ],
  6: [
    { re: /loading textured mesh|scoring cleanup proposal|cleanup proposal ready/i, idx: 0 },
    { re: /waiting for cleanup review decision|cleanup review ready/i, idx: 1 },
    { re: /bottom hole-fill/i, idx: 2 },
    { re: /post-holefill noise removal/i, idx: 3 },
    { re: /general hole-fill/i, idx: 4 },
    { re: /final cleanup/i, idx: 5 },
    { re: /writing cleaned obj\/mtl|post-texture cleanup complete|post-texture cleanup skipped|applying cleanup decision|preparing cleanup geometry/i, idx: 6 },
  ],
};

function clampStage(stage) {
  const n = Number(stage);
  if (!Number.isFinite(n)) return STAGE_MIN;
  return Math.max(STAGE_MIN, Math.min(STAGE_MAX, Math.round(n)));
}

function normalizeStatus(status) {
  const s = String(status || 'pending').toLowerCase();
  if (s === 'running' || s === 'complete' || s === 'failed' || s === 'interactive') {
    return s;
  }
  return 'pending';
}

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function isTransitionConfirmationDetail(detail, stage) {
  const text = String(detail || '').trim();
  if (!text) return false;

  if (/continue to|confirm to continue/i.test(text)) {
    return true;
  }

  const stageNum = clampStage(stage);
  if (new RegExp(`\\bstage\\s+${stageNum}\\s+(?:step\\s+)?complete\\b`, 'i').test(text)) {
    return true;
  }

  const stageLabel = STAGE_LABELS[stageNum];
  if (!stageLabel) return false;
  return new RegExp(`\\b${escapeRegExp(stageLabel)}\\s+complete\\b`, 'i').test(text);
}

function stateToken(state) {
  if (state === 'running') return '[o]';
  if (state === 'complete') return '[x]';
  if (state === 'failed') return '[!]';
  return '[ ]';
}

export class CheckpointPanel {
  constructor() {
    this._panel = document.getElementById('checkpoint-panel');
    this._stageLabel = document.getElementById('checkpoint-stage-label');
    this._list = document.getElementById('checkpoint-list');
    this._activeStage = STAGE_MIN;
    this._stageInfo = {};
    this._lastRunningIndex = {};

    this.reset(1);
  }

  setActiveStage(stage) {
    const next = clampStage(stage);
    if (this._activeStage === next) return;
    this._activeStage = next;
    this._render();
  }

  reset(resumeFromStage = 1) {
    const resume = clampStage(resumeFromStage);
    for (let stage = STAGE_MIN; stage <= STAGE_MAX; stage++) {
      this._stageInfo[stage] = {
        status: stage < resume ? 'complete' : 'pending',
        detail: null,
        checkpointId: null,
      };
      this._lastRunningIndex[stage] = 0;
    }
    this._render();
  }

  applyStatusSnapshot(statusMsg) {
    const stages = statusMsg?.stages || {};
    for (let stage = STAGE_MIN; stage <= STAGE_MAX; stage++) {
      const info = stages[String(stage)] || {};
      this._stageInfo[stage] = {
        status: normalizeStatus(info.status),
        detail: typeof info.detail === 'string' ? info.detail : null,
        checkpointId: typeof info.checkpoint_id === 'string' ? info.checkpoint_id : null,
      };
    }
    this._render();
  }

  onStageStart(stage) {
    const s = clampStage(stage);
    const ids = CHECKPOINT_IDS[s] || [];
    this._stageInfo[s] = {
      status: 'running',
      detail: 'Starting',
      checkpointId: ids.length > 0 ? ids[0] : null,
    };
    this._renderStageIfActive(s);
  }

  onStageProgress(stage, detail = null, statusHint = null, checkpointId = null) {
    const s = clampStage(stage);
    const current = this._stageInfo[s] || { status: 'pending', detail: null, checkpointId: null };
    const nextStatus = normalizeStatus(statusHint || current.status || 'running');
    const status = nextStatus === 'pending' ? 'running' : nextStatus;
    this._stageInfo[s] = {
      status,
      detail: typeof detail === 'string' ? detail : current.detail,
      checkpointId: typeof checkpointId === 'string' && checkpointId
        ? checkpointId
        : current.checkpointId,
    };
    this._renderStageIfActive(s);
  }

  onStageInteractive(stage, detail = null, checkpointId = null) {
    const s = clampStage(stage);
    const current = this._stageInfo[s] || { status: 'pending', detail: null, checkpointId: null };
    this._stageInfo[s] = {
      status: 'interactive',
      detail: typeof detail === 'string' ? detail : current.detail,
      checkpointId: typeof checkpointId === 'string' && checkpointId
        ? checkpointId
        : current.checkpointId,
    };
    this._renderStageIfActive(s);
  }

  onStageComplete(stage) {
    const s = clampStage(stage);
    this._stageInfo[s] = {
      status: 'complete',
      detail: null,
    };
    this._renderStageIfActive(s);
  }

  onStageFailed(stage, error = null, checkpointId = null) {
    const s = clampStage(stage);
    const current = this._stageInfo[s] || { status: 'pending', detail: null, checkpointId: null };
    this._stageInfo[s] = {
      status: 'failed',
      detail: typeof error === 'string' && error ? error : current.detail,
      checkpointId: typeof checkpointId === 'string' && checkpointId
        ? checkpointId
        : current.checkpointId,
    };
    this._renderStageIfActive(s);
  }

  _renderStageIfActive(stage) {
    if (clampStage(stage) !== this._activeStage) return;
    this._render();
  }

  _resolveCurrentIndex(stage, detail, checkpointCount, checkpointId = null) {
    const text = String(detail || '').trim();
    const maxIndex = Math.max(0, checkpointCount - 1);
    const ids = CHECKPOINT_IDS[stage] || [];

    if (typeof checkpointId === 'string' && checkpointId) {
      const idx = ids.indexOf(checkpointId);
      if (idx >= 0) {
        return Math.max(0, Math.min(maxIndex, idx));
      }
    }

    if (text) {
      const matchers = DETAIL_MATCHERS[stage] || [];
      for (const matcher of matchers) {
        if (matcher.re.test(text)) {
          return Math.max(0, Math.min(maxIndex, Number(matcher.idx) || 0));
        }
      }
      if (/complete/i.test(text)) {
        return maxIndex;
      }
    }

    const previous = Number(this._lastRunningIndex[stage]);
    if (Number.isFinite(previous)) {
      return Math.max(0, Math.min(maxIndex, previous));
    }
    return 0;
  }

  _resolveCheckpointStates(stage, info, checkpointCount) {
    const states = new Array(checkpointCount).fill('pending');
    const status = normalizeStatus(info?.status);
    const detail = info?.detail;
    const checkpointId = info?.checkpointId;

    if (checkpointCount === 0) return states;

    if (status === 'complete') {
      states.fill('complete');
      this._lastRunningIndex[stage] = checkpointCount - 1;
      return states;
    }

    if (status === 'pending') {
      this._lastRunningIndex[stage] = 0;
      return states;
    }

    if (/waiting for next-stage confirmation/i.test(String(detail || ''))) {
      states.fill('complete');
      this._lastRunningIndex[stage] = checkpointCount - 1;
      return states;
    }
    if (status === 'interactive' && isTransitionConfirmationDetail(detail, stage)) {
      states.fill('complete');
      this._lastRunningIndex[stage] = checkpointCount - 1;
      return states;
    }

    const current = this._resolveCurrentIndex(stage, detail, checkpointCount, checkpointId);
    for (let i = 0; i < current; i++) {
      states[i] = 'complete';
    }

    if (status === 'failed') {
      states[current] = 'failed';
    } else {
      states[current] = 'running';
    }

    this._lastRunningIndex[stage] = current;
    return states;
  }

  _render() {
    if (!this._panel || !this._stageLabel || !this._list) return;

    const stage = this._activeStage;
    const stageLabel = STAGE_LABELS[stage] || `Stage ${stage}`;
    this._stageLabel.textContent = `Stage ${stage} | ${stageLabel}`;

    const checkpoints = CHECKPOINT_TEMPLATES[stage] || [];
    const info = this._stageInfo[stage] || { status: 'pending', detail: null };
    const states = this._resolveCheckpointStates(stage, info, checkpoints.length);

    this._list.innerHTML = '';

    for (let i = 0; i < checkpoints.length; i++) {
      const state = states[i] || 'pending';
      const item = document.createElement('li');
      item.className = `checkpoint-item checkpoint-${state}`;

      const indicator = document.createElement('span');
      indicator.className = 'checkpoint-indicator';
      indicator.textContent = stateToken(state);

      const text = document.createElement('span');
      text.className = 'checkpoint-text';
      text.textContent = checkpoints[i];

      item.appendChild(indicator);
      item.appendChild(text);
      this._list.appendChild(item);
    }
  }
}
