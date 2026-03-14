/**
 * Stage progress bar UI controller.
 */

import { formatTime } from './utils.js';
import { STAGE_COUNT } from './constants.js';

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

    for (let i = 1; i <= STAGE_COUNT; i++) {
      this._pills[i] = document.querySelector(`.stage-pill[data-stage="${i}"]`);
      this._stageMeta[i] = {
        status: 'pending',
        progress: 0,
        detail: null,
        elapsed: null,
      };
    }

    this._mainConnectors = Array.from(document.querySelectorAll('.stage-connector-main'));
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

  _setStageState(stage, status, elapsed, progress, detail) {
    const pill = this._pills[stage];
    if (!pill) return;
    this._applyPillState(pill, status, elapsed, progress, detail);
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
    // Linear connectors between stages
    for (let i = 0; i < this._mainConnectors.length; i++) {
      const stage = i + 2;
      const conn = this._mainConnectors[i];
      if (!conn) continue;
      const done = _isDone(this._stageMeta[stage]);
      conn.classList.toggle('done', done);
    }
  }

  _startTimer(stage) {
    this._stopTimer(stage);
    const pill = this._pills[stage];
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
    return formatTime(secs);
  }

  _normalizeProgress(progress) {
    const n = Number(progress);
    if (!Number.isFinite(n)) return 0;
    if (n < 0) return 0;
    if (n > 100) return 100;
    return n;
  }
}
