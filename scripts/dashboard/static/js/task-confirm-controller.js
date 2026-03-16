/**
 * Task confirmation bar UI controller.
 *
 * Manages the per-stage confirmation bars that gate transitions between stages.
 */

import { STAGE_COUNT, TRANSITION_STAGE_MAX } from './constants.js';
import {
  getStageInfo,
  isStageDone,
  isTransitionConfirmed,
  resolvePreferredStage,
  resolveTransitionTarget,
  defaultTaskConfirmIdleMessage,
  defaultTaskConfirmWaitingMessage,
  defaultTaskConfirmConfirmedMessage,
  defaultTaskConfirmStandbyMessage,
} from './pipeline-status.js';

export class TaskConfirmController {
  /**
   * @param {object} deps
   * @param {object} deps.stageCtrl - StageController instance (for activeStage)
   * @param {function} deps.appendLog - appendLog(stream, text, opts)
   */
  constructor({ stageCtrl, appendLog }) {
    this._stageCtrl = stageCtrl;
    this._appendLog = appendLog;
    this._waitingConfirmationStage = null;
    this._waitingConfirmationToStage = null;
    this._bars = {};
    this._messages = {};
    this._buttons = {};

    for (let stage = 1; stage <= STAGE_COUNT; stage++) {
      this._bars[stage] = document.querySelector(`.task-confirm-bar[data-stage="${stage}"]`);
      this._messages[stage] = this._bars[stage]?.querySelector('.task-confirm-message') || null;
      this._buttons[stage] = this._bars[stage]?.querySelector('.task-confirm-btn') || null;

      if (stage <= TRANSITION_STAGE_MAX && this._buttons[stage]) {
        this._buttons[stage].addEventListener('click', () => this.confirmNextStage(stage));
      }
    }
    this.resetBars();
  }

  get waitingStage() { return this._waitingConfirmationStage; }
  set waitingStage(v) { this._waitingConfirmationStage = v; }

  get waitingToStage() { return this._waitingConfirmationToStage; }
  set waitingToStage(v) { this._waitingConfirmationToStage = v; }

  resolveVisibleStage(preferredStage) {
    const waiting = Number(this._waitingConfirmationStage);
    if (Number.isFinite(waiting) && waiting >= 1 && waiting <= TRANSITION_STAGE_MAX) {
      return waiting;
    }

    const preferred = Number(preferredStage);
    if (Number.isFinite(preferred) && preferred >= 1 && preferred <= STAGE_COUNT) {
      return preferred;
    }

    const active = Number(this._stageCtrl.activeStage);
    if (Number.isFinite(active) && active >= 1 && active <= STAGE_COUNT) {
      return active;
    }

    return 1;
  }

  setVisibleStage(preferredStage = null) {
    const visibleStage = this.resolveVisibleStage(preferredStage);
    for (let stage = 1; stage <= STAGE_COUNT; stage++) {
      const bar = this._bars[stage];
      if (!bar) continue;
      bar.classList.toggle('is-visible', stage === visibleStage);
    }
  }

  resolveIdleMessage(statusMsg, stage) {
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

  setState(stage, state, message) {
    const bar = this._bars[stage];
    const msg = this._messages[stage];
    const btn = this._buttons[stage];
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

  resetBars(resumeFromStage = 1) {
    const resumeStage = Math.max(1, Math.min(STAGE_COUNT, Number(resumeFromStage) || 1));
    this._waitingConfirmationStage = null;
    this._waitingConfirmationToStage = null;
    for (let stage = 1; stage <= TRANSITION_STAGE_MAX; stage++) {
      if (stage < resumeStage) {
        this.setState(stage, 'confirmed', `Stage ${stage} already completed before resume.`);
      } else {
        this.setState(stage, 'idle', defaultTaskConfirmIdleMessage(stage));
      }
    }
    this.setState(STAGE_COUNT, 'final', 'Final stage. No next-stage confirmation.');
    this.setVisibleStage(resumeStage);
  }

  syncFromStatus(statusMsg) {
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
    this._waitingConfirmationStage = waitingStageIsValid ? waitingStage : null;
    this._waitingConfirmationToStage = waitingStageIsValid && Number.isFinite(waitingToStage)
      ? waitingToStage
      : null;

    const resumeStage = Math.max(1, Math.min(STAGE_COUNT, Number(statusMsg.resume_from_stage) || 1));

    for (let stage = 1; stage <= TRANSITION_STAGE_MAX; stage++) {
      const stageDone = isStageDone(statusMsg, stage);

      if (this._waitingConfirmationStage === stage) {
        this.setState(
          stage,
          'waiting',
          String(next.message || defaultTaskConfirmWaitingMessage(stage, this._waitingConfirmationToStage)),
        );
        continue;
      }

      if (isTransitionConfirmed(statusMsg, stage)) {
        this.setState(stage, 'confirmed', defaultTaskConfirmConfirmedMessage(stage));
        continue;
      }

      if (stage < resumeStage && stageDone) {
        this.setState(stage, 'confirmed', `Stage ${stage} already completed before resume.`);
        continue;
      }

      if (!statusMsg.running && stageDone) {
        this.setState(stage, 'idle', defaultTaskConfirmStandbyMessage(stage));
        continue;
      }

      this.setState(stage, 'idle', this.resolveIdleMessage(statusMsg, stage));
    }

    if (isStageDone(statusMsg, STAGE_COUNT)) {
      this.setState(STAGE_COUNT, 'final', 'Final stage complete. No next-stage confirmation.');
    } else {
      this.setState(STAGE_COUNT, 'final', 'Final stage. No next-stage confirmation.');
    }
    this.setVisibleStage(resolvePreferredStage(statusMsg));
  }

  async confirmNextStage(stage) {
    if (this._waitingConfirmationStage !== stage) return;

    const toStage = this._waitingConfirmationToStage;
    const target = resolveTransitionTarget(stage, toStage);
    const waitingMsg = target === Number(stage)
      ? `Stage ${stage} confirmed. Waiting for the next Stage ${stage} step...`
      : `Stage ${stage} confirmed. Waiting for Stage ${target} start...`;
    this.setState(stage, 'sending', waitingMsg);

    try {
      const res = await fetch('/api/pipeline/confirm-next', { method: 'POST' });
      const data = await res.json();
      if (!res.ok || data.error) {
        throw new Error(data.error || `HTTP ${res.status}`);
      }

      if (data.status === 'no_waiting_confirmation') {
        this._waitingConfirmationStage = null;
        this._waitingConfirmationToStage = null;
        this.setState(stage, 'confirmed', defaultTaskConfirmConfirmedMessage(stage, toStage));
        this.setVisibleStage(this._stageCtrl.activeStage);
        return;
      }

      // Keep "sending" state until backend emits stage start or cleared event.
    } catch (e) {
      this.setState(stage, 'waiting', `${defaultTaskConfirmWaitingMessage(stage, toStage)} (${e.message})`);
      this._appendLog('stderr', `Next-stage confirmation failed at stage ${stage}: ${e.message}\n`, { stage });
    }
  }
}
