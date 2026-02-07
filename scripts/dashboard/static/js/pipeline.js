/**
 * Stage progress bar UI controller.
 */

const STAGE_COUNT = 6;

export class PipelineUI {
  constructor() {
    this._pills = [];
    this._connectors = [];
    this._timers = {}; // stage → interval id
    this._stageMeta = {};

    for (let i = 1; i <= STAGE_COUNT; i++) {
      this._pills.push(document.querySelector(`.stage-pill[data-stage="${i}"]`));
      this._stageMeta[i] = {
        status: 'pending',
        progress: 0,
        detail: null,
      };
    }
    this._connectors = Array.from(document.querySelectorAll('.stage-connector'));
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
      };
      this._setStageState(i, info.status, info.elapsed, info.progress, info.detail);
    }
    this._refreshConnectors();
  }

  stageStart(stage) {
    this._stageMeta[stage].status = 'running';
    this._stageMeta[stage].progress = 0;
    this._stageMeta[stage].detail = 'Starting';
    this._setStageState(stage, 'running', null, 0, 'Starting');
    this._startTimer(stage);
  }

  stageComplete(stage, elapsed) {
    this._stopTimer(stage);
    this._stageMeta[stage].status = 'complete';
    this._stageMeta[stage].progress = 100;
    this._stageMeta[stage].detail = null;
    this._setStageState(stage, 'complete', elapsed, 100, null);
    this._refreshConnectors();
  }

  stageProgress(stage, progress, detail) {
    const normalized = this._normalizeProgress(progress);
    const current = this._stageMeta[stage] || { status: 'running', detail: null };
    let status = current.status;
    if (status === 'pending') status = 'running';
    this._stageMeta[stage] = {
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

  _setStageState(stage, status, elapsed, progress, detail) {
    const pill = this._pills[stage - 1];
    if (!pill) return;
    const isSelected = pill.classList.contains('selected');
    pill.className = 'stage-pill';
    if (isSelected) pill.classList.add('selected');
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
    for (let i = 0; i < this._connectors.length; i++) {
      const stage = i + 2;
      const conn = this._connectors[i];
      if (!conn) continue;
      const info = this._stageMeta[stage];
      const done = info.status === 'complete'
        || info.status === 'interactive'
        || this._normalizeProgress(info.progress) >= 100;
      if (done) conn.classList.add('done');
      else conn.classList.remove('done');
    }
  }

  _startTimer(stage) {
    this._stopTimer(stage);
    const pill = this._pills[stage - 1];
    if (!pill) return;
    const timeEl = pill.querySelector('.stage-time');
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
