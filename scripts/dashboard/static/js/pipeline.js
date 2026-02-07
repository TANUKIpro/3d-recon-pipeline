/**
 * Stage progress bar UI controller.
 */

const STAGE_COUNT = 6;

export class PipelineUI {
  constructor() {
    this._pills = [];
    this._connectors = [];
    this._timers = {};  // stage → interval id

    for (let i = 1; i <= STAGE_COUNT; i++) {
      this._pills.push(document.querySelector(`.stage-pill[data-stage="${i}"]`));
    }
    this._connectors = Array.from(document.querySelectorAll('.stage-connector'));
  }

  /** Update all stages from a status dict. */
  updateAll(statusData) {
    const stages = statusData.stages || {};
    for (let i = 1; i <= STAGE_COUNT; i++) {
      const info = stages[String(i)];
      if (!info) continue;
      this._setStageState(i, info.status, info.elapsed);
    }
  }

  stageStart(stage) {
    this._setStageState(stage, 'running');
    this._startTimer(stage);
  }

  stageComplete(stage, elapsed) {
    this._stopTimer(stage);
    this._setStageState(stage, 'complete', elapsed);
    // Mark connector before this stage as done
    if (stage >= 2) {
      const conn = this._connectors[stage - 2];
      if (conn) conn.classList.add('done');
    }
  }

  stageFailed(stage) {
    this._stopTimer(stage);
    this._setStageState(stage, 'failed');
  }

  stageInteractive(stage) {
    this._setStageState(stage, 'interactive');
  }

  reset() {
    for (let i = 1; i <= STAGE_COUNT; i++) {
      this._stopTimer(i);
      this._setStageState(i, 'pending');
    }
    for (const conn of this._connectors) {
      conn.classList.remove('done');
    }
  }

  _setStageState(stage, status, elapsed) {
    const pill = this._pills[stage - 1];
    if (!pill) return;
    const isSelected = pill.classList.contains('selected');
    pill.className = 'stage-pill';
    if (isSelected) pill.classList.add('selected');
    if (status && status !== 'pending') {
      pill.classList.add(status);
    }
    const timeEl = pill.querySelector('.stage-time');
    if (timeEl && elapsed != null) {
      timeEl.textContent = this._formatTime(elapsed);
    } else if (timeEl && status === 'pending') {
      timeEl.textContent = '';
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
}
