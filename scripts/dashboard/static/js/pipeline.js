/**
 * Stage progress bar UI controller.
 */

const STAGE_COUNT = 6;
const MESH_METHODS = new Set(['diffcd', 'poisson']);
const POISSON_STEP_ORDER = ['preprocess', 'main', 'postprocess', 'downsample'];

function _isDone(info = {}) {
  const status = String(info.status || 'pending');
  if (status === 'complete' || status === 'interactive') return true;
  return Number(info.progress) >= 100;
}

export class PipelineUI {
  constructor() {
    this._pills = {};
    this._mainConnectors = [];
    this._timers = {}; // stage → interval id
    this._stageMeta = {};
    this._meshMethod = 'poisson';
    this._poissonFlowState = { activeIndex: -1, done: false };

    for (let i = 1; i <= STAGE_COUNT; i++) {
      this._pills[i] = document.querySelector(`.stage-pill[data-stage="${i}"]`);
      this._stageMeta[i] = {
        status: 'pending',
        progress: 0,
        detail: null,
        elapsed: null,
      };
    }

    this._meshPills = {
      diffcd: document.getElementById('mesh-pill-diffcd'),
      poisson: document.getElementById('mesh-pill-poisson'),
    };
    this._meshBranchSlot = document.getElementById('mesh-branch-slot');

    this._poissonStepPills = {
      preprocess: document.getElementById('mesh-pill-poisson-pre'),
      main: document.getElementById('mesh-pill-poisson'),
      postprocess: document.getElementById('mesh-pill-poisson-post'),
      downsample: document.getElementById('mesh-pill-poisson-downsample'),
    };

    this._mainConnectors = Array.from(document.querySelectorAll('.stage-connector-main'));
    this._poissonConnectors = {
      left: document.getElementById('mesh-connector-poisson-left'),
      preMain: document.getElementById('mesh-connector-poisson-pre-main'),
      mainPost: document.getElementById('mesh-connector-poisson-main-post'),
      postDown: document.getElementById('mesh-connector-poisson-post-down'),
      right: document.getElementById('mesh-connector-poisson-right'),
    };

    this.setMeshMethod(this._meshMethod);
  }

  /** Update all stages from a status dict. */
  updateAll(statusData) {
    const stages = statusData.stages || {};
    for (let i = 1; i <= STAGE_COUNT; i++) {
      const info = stages[String(i)];
      if (!info) continue;
      this._stageMeta[i] = {
        status: info.status || 'pending',
        progress: this._normalizeProgress(info.progress),
        detail: info.detail || null,
        elapsed: info.elapsed ?? null,
      };
      this._setStageState(i, info.status, info.elapsed, info.progress, info.detail);
    }
    this._refreshConnectors();
  }

  setMeshMethod(method) {
    const next = MESH_METHODS.has(String(method)) ? String(method) : 'poisson';
    this._meshMethod = next;
    for (const [key, pill] of Object.entries(this._meshPills)) {
      if (!pill) continue;
      pill.classList.toggle('active', key === next);
    }
    if (this._meshBranchSlot) {
      this._meshBranchSlot.classList.toggle('poisson-inactive', next !== 'poisson');
    }

    // If Stage 5 tab is currently selected, move selected outline to active method.
    const stage5Focused = Object.values(this._meshPills).some((pill) => pill?.classList.contains('selected'));
    if (stage5Focused) {
      for (const [key, pill] of Object.entries(this._meshPills)) {
        if (!pill) continue;
        pill.classList.toggle('selected', key === next);
      }
    }

    this._setStageState(
      5,
      this._stageMeta[5]?.status,
      this._stageMeta[5]?.elapsed,
      this._stageMeta[5]?.progress,
      this._stageMeta[5]?.detail,
    );
    this._refreshConnectors();
  }

  getMeshMethod() {
    return this._meshMethod;
  }

  setMeshMethodEnabled(enabled) {
    const disabled = !enabled;
    for (const pill of Object.values(this._meshPills)) {
      if (!pill) continue;
      pill.classList.toggle('method-disabled', disabled);
    }
    const poissonPill = this._meshPills.poisson;
    if (poissonPill) {
      poissonPill.setAttribute('tabindex', disabled ? '-1' : '0');
      poissonPill.setAttribute('aria-disabled', disabled ? 'true' : 'false');
    }

    for (const pill of Object.values(this._poissonStepPills)) {
      if (!pill || pill === poissonPill) continue;
      pill.classList.toggle('method-disabled', disabled);
      pill.setAttribute('tabindex', disabled ? '-1' : '0');
      pill.setAttribute('aria-disabled', disabled ? 'true' : 'false');
    }
  }

  stageStart(stage) {
    this._stageMeta[stage].status = 'running';
    this._stageMeta[stage].progress = 0;
    this._stageMeta[stage].detail = 'Starting';
    this._stageMeta[stage].elapsed = null;
    this._setStageState(stage, 'running', null, 0, 'Starting');
    this._startTimer(stage);
  }

  stageComplete(stage, elapsed) {
    this._stopTimer(stage);
    this._stageMeta[stage].status = 'complete';
    this._stageMeta[stage].progress = 100;
    this._stageMeta[stage].detail = null;
    this._stageMeta[stage].elapsed = elapsed ?? null;
    this._setStageState(stage, 'complete', elapsed, 100, null);
    this._refreshConnectors();
  }

  stageProgress(stage, progress, detail) {
    const normalized = this._normalizeProgress(progress);
    const current = this._stageMeta[stage] || { status: 'running', detail: null };
    let status = current.status;
    if (status === 'pending' || status === 'interactive') status = 'running';
    this._stageMeta[stage] = {
      ...current,
      status,
      progress: normalized,
      detail: detail ?? current.detail,
    };
    this._setStageState(stage, status, null, normalized, detail ?? current.detail);
  }

  stageFailed(stage) {
    this._stopTimer(stage);
    this._stageMeta[stage].status = 'failed';
    this._setStageState(
      stage,
      'failed',
      null,
      this._stageMeta[stage].progress,
      this._stageMeta[stage].detail,
    );
  }

  stageInteractive(stage) {
    this._stageMeta[stage].status = 'interactive';
    this._setStageState(
      stage,
      'interactive',
      null,
      this._stageMeta[stage].progress,
      this._stageMeta[stage].detail,
    );
  }

  reset() {
    for (let i = 1; i <= STAGE_COUNT; i++) {
      this._stopTimer(i);
      this._stageMeta[i] = {
        status: 'pending',
        progress: 0,
        detail: null,
        elapsed: null,
      };
      this._setStageState(i, 'pending', null, 0, null);
    }
    this._refreshConnectors();
  }

  resetFromStage(startStage = 1) {
    const start = Math.max(1, Math.min(STAGE_COUNT, Number(startStage) || 1));
    for (let i = 1; i < start; i++) {
      this._stopTimer(i);
    }
    for (let i = start; i <= STAGE_COUNT; i++) {
      this._stopTimer(i);
      this._stageMeta[i] = {
        status: 'pending',
        progress: 0,
        detail: null,
        elapsed: null,
      };
      this._setStageState(i, 'pending', null, 0, null);
    }
    this._refreshConnectors();
  }

  getOverallProgress() {
    let sum = 0;
    for (let i = 1; i <= STAGE_COUNT; i++) {
      sum += this._normalizeProgress(this._stageMeta[i]?.progress);
    }
    return Math.round((sum / STAGE_COUNT) * 10) / 10;
  }

  getStageStatus(stage) {
    return this._stageMeta[stage]?.status || 'pending';
  }

  _pillForStage(stage) {
    if (stage === 5) {
      return this._meshMethod === 'diffcd'
        ? this._meshPills.diffcd
        : this._meshPills.poisson;
    }
    return this._pills[stage];
  }

  _inactiveMeshPill() {
    return this._meshMethod === 'diffcd'
      ? this._meshPills.poisson
      : this._meshPills.diffcd;
  }

  _setStageState(stage, status, elapsed, progress, detail) {
    const pill = this._pillForStage(stage);
    if (!pill) return;

    this._applyPillState(pill, status, elapsed, progress, detail);

    // Stage 5 has dual mesh method pills. Keep inactive path visually pending.
    if (stage === 5) {
      const inactivePill = this._inactiveMeshPill();
      if (inactivePill) {
        this._applyPillState(inactivePill, 'pending', null, 0, null);
      }
      this._updatePoissonStepStates(status, progress, detail);
    }
  }

  _updatePoissonStepStates(status, progress, detail) {
    const normalized = this._normalizeProgress(progress);
    const stageStatus = String(status || 'pending');
    const isPoisson = this._meshMethod === 'poisson';
    const done = stageStatus === 'complete';

    if (!isPoisson) {
      for (const step of POISSON_STEP_ORDER) {
        const pill = this._poissonStepPills[step];
        if (!pill) continue;
        this._applyPillState(pill, 'pending', null, 0, null);
      }
      this._poissonFlowState = { activeIndex: -1, done: false };
      return;
    }

    if (stageStatus === 'pending' && normalized <= 0) {
      for (const step of POISSON_STEP_ORDER) {
        const pill = this._poissonStepPills[step];
        if (!pill) continue;
        this._applyPillState(pill, 'pending', null, 0, null);
      }
      this._poissonFlowState = { activeIndex: -1, done: false };
      return;
    }

    if (done) {
      let completedIndex = this._inferPoissonStepIndex(detail, normalized);
      if (completedIndex < 0) {
        completedIndex = POISSON_STEP_ORDER.length - 1;
      }
      completedIndex = Math.max(0, Math.min(POISSON_STEP_ORDER.length - 1, completedIndex));
      for (const step of POISSON_STEP_ORDER) {
        const pill = this._poissonStepPills[step];
        if (!pill) continue;
        const idx = POISSON_STEP_ORDER.indexOf(step);
        if (idx <= completedIndex) {
          this._applyPillState(pill, 'complete', null, 100, detail);
        } else {
          this._applyPillState(pill, 'pending', null, 0, null);
        }
      }
      this._poissonFlowState = {
        activeIndex: completedIndex,
        done: completedIndex >= POISSON_STEP_ORDER.length - 1,
      };
      return;
    }

    let activeIndex = this._inferPoissonStepIndex(detail, normalized);
    if (
      stageStatus === 'interactive'
      && String(detail || '').toLowerCase().includes('waiting for next-stage confirmation')
    ) {
      activeIndex = this._poissonFlowState.activeIndex;
    }
    if (activeIndex < 0) activeIndex = this._progressStepIndex(normalized);
    activeIndex = Math.max(0, Math.min(POISSON_STEP_ORDER.length - 1, activeIndex));

    for (let i = 0; i < POISSON_STEP_ORDER.length; i++) {
      const step = POISSON_STEP_ORDER[i];
      const pill = this._poissonStepPills[step];
      if (!pill) continue;

      if (i < activeIndex) {
        this._applyPillState(pill, 'complete', null, 100, detail);
        continue;
      }
      if (i > activeIndex) {
        this._applyPillState(pill, 'pending', null, 0, null);
        continue;
      }

      const currentStatus = stageStatus === 'failed'
        ? 'failed'
        : stageStatus === 'interactive'
          ? 'interactive'
          : 'running';
      this._applyPillState(pill, currentStatus, null, normalized, detail);
    }

    this._poissonFlowState = { activeIndex, done: false };
  }

  _inferPoissonStepIndex(detail, progress) {
    const lower = String(detail || '').toLowerCase();
    if (!lower) return this._progressStepIndex(progress);

    if (
      lower.includes('classical/downsample')
      || lower.includes('downsample')
      || lower.includes('decimation')
      || lower.includes('reducing face count')
    ) {
      return 3;
    }
    if (
      lower.includes('classical/postprocess')
      || lower.includes('postprocess')
      || lower.includes('smoothing')
      || lower.includes('cleaning mesh')
    ) {
      return 2;
    }
    if (
      lower.includes('classical/main')
      || lower.includes('screened poisson')
      || lower.includes('poisson')
      || lower.includes('normal')
    ) {
      return 1;
    }
    if (
      lower.includes('classical/preprocess')
      || lower.includes('preprocess')
      || lower.includes('resampling points')
    ) {
      return 0;
    }
    return this._progressStepIndex(progress);
  }

  _progressStepIndex(progress) {
    const p = this._normalizeProgress(progress);
    if (p >= 94) return 3;
    if (p >= 76) return 2;
    if (p >= 28) return 1;
    return 0;
  }

  _applyPillState(pill, status, elapsed, progress, detail) {
    pill.classList.remove('running', 'complete', 'failed', 'interactive');
    if (status && status !== 'pending') {
      pill.classList.add(status);
    }

    const normalized = this._normalizeProgress(progress);
    pill.style.setProperty('--progress', `${normalized.toFixed(1)}`);

    const progressEl = pill.querySelector('.stage-progress');
    if (progressEl) {
      progressEl.textContent = `${Math.round(normalized)}%`;
    }

    const timeEl = pill.querySelector('.stage-time');
    if (timeEl && elapsed != null) {
      timeEl.textContent = this._formatTime(elapsed);
    } else if (timeEl && status === 'pending') {
      timeEl.textContent = '';
    }

    pill.title = detail || '';
  }

  _refreshConnectors() {
    // Linear connectors for stages 1→4.
    for (let i = 0; i < 3; i++) {
      const stage = i + 2;
      const conn = this._mainConnectors[i];
      if (!conn) continue;
      const done = _isDone(this._stageMeta[stage]);
      conn.classList.toggle('active', false);
      conn.classList.toggle('done', done);
    }

    const stage5Done = _isDone(this._stageMeta[5]);
    const stage6Done = _isDone(this._stageMeta[6]);
    const poissonActive = this._meshMethod === 'poisson';

    // Connectors around the mesh branch slot are shared trunk lines.
    // Keep them active regardless of selected mesh method so branch lines connect visually.
    const meshLeftTrunk = this._mainConnectors[3];
    const meshRightTrunk = this._mainConnectors[4];
    if (meshLeftTrunk) {
      meshLeftTrunk.classList.toggle('active', true);
      meshLeftTrunk.classList.toggle('done', stage5Done);
    }
    if (meshRightTrunk) {
      meshRightTrunk.classList.toggle('active', true);
      meshRightTrunk.classList.toggle('done', stage6Done);
    }

    const activeIndex = this._poissonFlowState.activeIndex;
    const flowDone = this._poissonFlowState.done;
    const leftDone = poissonActive && (flowDone || activeIndex >= 0);
    const preMainDone = poissonActive && (flowDone || activeIndex > 0);
    const mainPostDone = poissonActive && (flowDone || activeIndex > 1);
    const postDownDone = poissonActive && (flowDone || activeIndex > 2);

    this._setConnectorState(this._poissonConnectors.left, poissonActive, leftDone);
    this._setConnectorState(
      this._poissonConnectors.preMain,
      poissonActive && activeIndex >= 0,
      preMainDone,
    );
    this._setConnectorState(
      this._poissonConnectors.mainPost,
      poissonActive && activeIndex >= 1,
      mainPostDone,
    );
    this._setConnectorState(
      this._poissonConnectors.postDown,
      poissonActive && activeIndex >= 2,
      postDownDone,
    );
    this._setConnectorState(this._poissonConnectors.right, poissonActive, poissonActive && stage6Done);
  }

  _setConnectorState(connector, active, done) {
    if (!connector) return;
    connector.classList.toggle('active', Boolean(active));
    connector.classList.toggle('done', Boolean(done));
  }

  _startTimer(stage) {
    this._stopTimer(stage);
    const pill = this._pillForStage(stage);
    if (!pill) return;
    const timeEl = pill.querySelector('.stage-time');
    if (!timeEl) return;
    const start = Date.now();
    this._timers[stage] = setInterval(() => {
      const secs = (Date.now() - start) / 1000;
      timeEl.textContent = this._formatTime(secs);
    }, 1000);
  }

  _stopTimer(stage) {
    if (this._timers[stage]) {
      clearInterval(this._timers[stage]);
      delete this._timers[stage];
    }
  }

  _formatTime(secs) {
    if (secs == null) return '';
    if (secs < 60) return `${Math.round(secs)}s`;
    const m = Math.floor(secs / 60);
    const s = Math.round(secs % 60);
    return `${m}m${s}s`;
  }

  _normalizeProgress(progress) {
    const n = Number(progress);
    if (!Number.isFinite(n)) return 0;
    if (n < 0) return 0;
    if (n > 100) return 100;
    return n;
  }
}
