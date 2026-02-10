/**
 * Stage progress bar UI controller.
 */

const STAGE_COUNT = 7;
const MESH_METHODS = new Set(['diffcd', 'poisson']);

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

    this._mainConnectors = Array.from(document.querySelectorAll('.stage-connector-main'));
    this._poissonConnectors = {
      left: document.getElementById('mesh-connector-poisson-left'),
      right: document.getElementById('mesh-connector-poisson-right'),
    };
    this._diffcdConnectors = {
      left: document.getElementById('mesh-connector-diffcd-left'),
      right: document.getElementById('mesh-connector-diffcd-right'),
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
    }
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
    const stage7Done = _isDone(this._stageMeta[7]);
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
    const textureTrunk = this._mainConnectors[5];
    if (textureTrunk) {
      textureTrunk.classList.toggle('active', true);
      textureTrunk.classList.toggle('done', stage7Done);
    }

    this._setConnectorState(this._poissonConnectors.left, poissonActive, poissonActive && stage5Done);
    this._setConnectorState(this._poissonConnectors.right, poissonActive, poissonActive && stage6Done);

    // DiffCD path connectors
    const diffcdActive = !poissonActive;
    this._setConnectorState(this._diffcdConnectors.left, diffcdActive, diffcdActive && stage5Done);
    this._setConnectorState(this._diffcdConnectors.right, diffcdActive, diffcdActive && stage6Done);
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
