/**
 * Stage checkpoint viewer for the right panel.
 *
 * Shows fixed checkpoint templates per stage and maps stage detail text
 * to one running checkpoint.
 */

const STAGE_MIN = 1;
const STAGE_MAX = 8;

const STAGE_LABELS = {
  1: 'Extract Frames',
  2: 'Pi3X',
  3: 'SAM2',
  4: 'Denoise',
  5: 'Mesh Reconstruction',
  6: 'Mesh Wrap',
  7: 'Mesh Repair',
  8: 'Texture Bake',
};

const CHECKPOINT_TEMPLATES = {
  1: [
    'Inspect video metadata',
    'Extract frames',
    'Finalize frame set',
  ],
  2: [
    'Load Pi3X model',
    'Run Pi3X inference',
    'Apply confidence/edge filters',
    'Save point cloud and camera poses',
    'Save Pi3X cache',
  ],
  3: [
    'Initialize SAM2 model',
    'Wait for interactive clicks',
    'Propagate masks',
    'Wait for mask verification',
    'Apply SAM2 masks to point cloud',
  ],
  4: [
    'Load point cloud',
    'Run denoise algorithm',
    'Save denoised point cloud',
  ],
  5: {
    poisson: [
      'Preprocess point cloud',
      'Reconstruct mesh (Poisson)',
      'Postprocess mesh',
      'Downsample mesh',
    ],
    diffcd: [
      'Prepare point cloud for DiffCD',
      'Run DiffCD fitting',
      'Collect DiffCD outputs',
      'Apply mesh smoothing',
    ],
  },
  6: [
    'Load mesh',
    'Iterative mesh wrapping',
    'Adjust face count',
    'Save wrapped mesh',
  ],
  7: [
    'Load wrapped mesh',
    'Find and evaluate boundary loops',
    'Save repaired mesh',
  ],
  8: [
    'Load mesh',
    'Estimate camera intrinsics',
    'Generate UV atlas / texel mapping',
    'Score and project views',
    'Secondary fill and seam padding',
    'Export textured mesh',
  ],
};

const CHECKPOINT_IDS = {
  1: [
    's1.inspect',
    's1.extract',
    's1.finalize',
  ],
  2: [
    's2.load_model',
    's2.infer',
    's2.filter',
    's2.save_outputs',
    's2.save_cache',
  ],
  3: [
    's3.initialize',
    's3.interact',
    's3.propagate',
    's3.verify',
    's3.apply',
  ],
  4: [
    's4.load',
    's4.denoise',
    's4.save',
  ],
  5: {
    poisson: [
      's5.poisson.preprocess',
      's5.poisson.reconstruct',
      's5.poisson.postprocess',
      's5.poisson.downsample',
    ],
    diffcd: [
      's5.diffcd.prepare',
      's5.diffcd.fit',
      's5.diffcd.collect',
      's5.diffcd.smooth',
    ],
  },
  6: [
    's6.load',
    's6.wrap',
    's6.adjust',
    's6.save',
  ],
  7: [
    's7.load',
    's7.find_loops',
    's7.save',
  ],
  8: [
    's8.load',
    's8.intrinsics',
    's8.uv',
    's8.score',
    's8.fill',
    's8.export',
  ],
};

const DETAIL_MATCHERS = {
  1: [
    { re: /inspect|metadata/i, idx: 0 },
    { re: /extracting frames|extract frames/i, idx: 1 },
    { re: /extracted\s+\d+\s+frames|finalize|complete/i, idx: 2 },
  ],
  2: [
    { re: /loading pi3x model/i, idx: 0 },
    { re: /running pi3x fitting|pi3x chunk inference|running pi3x fitting/i, idx: 1 },
    { re: /confidence|edge filters/i, idx: 2 },
    { re: /saving point cloud|saving camera poses/i, idx: 3 },
    { re: /saving pi3x cache|collecting pi3x outputs|pi3x complete/i, idx: 4 },
  ],
  3: [
    { re: /initializing sam2 model|sam2 ready/i, idx: 0 },
    { re: /waiting for interactive clicks|redoing sam2 interaction/i, idx: 1 },
    { re: /propagating masks/i, idx: 2 },
    { re: /waiting for mask verification|verification/i, idx: 3 },
    { re: /loading sam2 masks|applying sam2 mask filter|applying sam2 masks/i, idx: 4 },
  ],
  4: [
    { re: /loading point cloud|loaded\s+\d+/i, idx: 0 },
    { re: /running dbscan|running statistical outlier removal|running radius outlier removal|dbscan complete|sor complete|radius outlier complete/i, idx: 1 },
    { re: /saving denoised point cloud|denoise complete/i, idx: 2 },
  ],
  '5:poisson': [
    { re: /classical\/preprocess/i, idx: 0 },
    { re: /classical\/main|poisson reconstruction|raw poisson mesh saved/i, idx: 1 },
    { re: /classical\/postprocess/i, idx: 2 },
    { re: /classical\/downsample|classical mesh stage complete/i, idx: 3 },
  ],
  '5:diffcd': [
    { re: /preparing point cloud for diffcd|point cloud prepared/i, idx: 0 },
    { re: /running diffcd fitting|diffcd attempt|diffcd training/i, idx: 1 },
    { re: /collecting diffcd outputs|diffcd complete/i, idx: 2 },
    { re: /applying mesh smoothing|diffcd stage complete/i, idx: 3 },
  ],
  6: [
    { re: /mesh wrap: loading mesh/i, idx: 0 },
    { re: /mesh wrap: sampling surface points|mesh wrap: estimating normals|mesh wrap: running poisson reconstruction|mesh wrap: iteration/i, idx: 1 },
    { re: /mesh wrap: adjusting final face count/i, idx: 2 },
    { re: /mesh wrap: saving wrapped mesh|mesh wrap: complete/i, idx: 3 },
  ],
  7: [
    { re: /contact hole repair: loading mesh/i, idx: 0 },
    { re: /contact hole repair: extracting boundary loops|contact hole repair: evaluating boundary loop/i, idx: 1 },
    { re: /contact hole repair: writing repaired mesh|saved repaired mesh|contact hole repair: complete|contact hole repair: skipped/i, idx: 2 },
  ],
  8: [
    { re: /loading mesh/i, idx: 0 },
    { re: /estimating camera intrinsics|intrinsics optimization complete|camera intrinsics estimated/i, idx: 1 },
    { re: /generating uv atlas|building texel mapping/i, idx: 2 },
    { re: /scoring camera views|scoring views|applying primary views|projecting primary chart textures/i, idx: 3 },
    { re: /secondary view search|padding uv seams|padding seams/i, idx: 4 },
    { re: /exporting textured mesh|texture stage complete/i, idx: 5 },
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

function normalizeMeshMethod(method) {
  return String(method || '').trim().toLowerCase() === 'diffcd' ? 'diffcd' : 'poisson';
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
    this._meshMethod = 'poisson';
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

  setMeshMethod(method) {
    const next = normalizeMeshMethod(method);
    if (this._meshMethod === next) return;
    this._meshMethod = next;
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
    this._meshMethod = normalizeMeshMethod(statusMsg?.mesh_method || this._meshMethod);
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
    const ids = this._idTemplateFor(s);
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

  _templateFor(stage) {
    if (stage === 5) {
      return CHECKPOINT_TEMPLATES[5][this._meshMethod] || CHECKPOINT_TEMPLATES[5].poisson;
    }
    return CHECKPOINT_TEMPLATES[stage] || [];
  }

  _idTemplateFor(stage) {
    if (stage === 5) {
      return CHECKPOINT_IDS[5][this._meshMethod] || CHECKPOINT_IDS[5].poisson;
    }
    return CHECKPOINT_IDS[stage] || [];
  }

  _matchersFor(stage) {
    if (stage === 5) {
      return DETAIL_MATCHERS[`5:${this._meshMethod}`] || [];
    }
    return DETAIL_MATCHERS[stage] || [];
  }

  _resolveCurrentIndex(stage, detail, checkpointCount, checkpointId = null) {
    const text = String(detail || '').trim();
    const maxIndex = Math.max(0, checkpointCount - 1);
    const ids = this._idTemplateFor(stage);

    if (typeof checkpointId === 'string' && checkpointId) {
      const idx = ids.indexOf(checkpointId);
      if (idx >= 0) {
        return Math.max(0, Math.min(maxIndex, idx));
      }
    }

    if (text) {
      for (const matcher of this._matchersFor(stage)) {
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
      // Treat interactive checkpoints as running from the stage-task perspective.
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

    const checkpoints = this._templateFor(stage);
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
