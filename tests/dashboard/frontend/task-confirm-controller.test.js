import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { useFetchMock } from './helpers/fetch-mock.js';
import { TaskConfirmController } from '../../../scripts/dashboard/static/js/task-confirm-controller.js';
import { STAGE_COUNT, TRANSITION_STAGE_MAX } from '../../../scripts/dashboard/static/js/constants.js';

// ── DOM builder (local to this test file) ────────────────────

function buildTaskConfirmDOM() {
  let bars = '';
  for (let i = 1; i <= 8; i++) {
    bars += `
      <div class="task-confirm-bar" data-stage="${i}">
        <span class="task-confirm-message"></span>
        <button class="task-confirm-btn"></button>
      </div>
    `;
  }
  document.body.innerHTML = bars;
}

// ── Helpers ──────────────────────────────────────────────────

function mkStageCtrl(activeStage = 1) {
  return { activeStage };
}

function mkStatus(overrides = {}) {
  return {
    object_name: 'test-obj',
    running: false,
    current_stage: null,
    next_stage_confirmation: {},
    resume_from_stage: 1,
    stages: {},
    ...overrides,
  };
}

function mkStage(status = 'pending', progress = 0) {
  return { status, progress };
}

// ── Tests ────────────────────────────────────────────────────

describe('TaskConfirmController', () => {
  let ctrl;
  let stageCtrl;
  let appendLog;
  const fetchMock = useFetchMock();

  beforeEach(() => {
    buildTaskConfirmDOM();
    stageCtrl = mkStageCtrl(1);
    appendLog = vi.fn();
    ctrl = new TaskConfirmController({ stageCtrl, appendLog });
  });

  afterEach(() => {
    fetchMock.clearRoutes();
    fetchMock.restoreFetch();
  });

  // ── constructor ──────────────────────────────────────────────

  describe('constructor', () => {
    it('sets up bars for all 8 stages', () => {
      for (let i = 1; i <= STAGE_COUNT; i++) {
        expect(ctrl._bars[i]).toBeTruthy();
      }
    });

    it('sets up message spans for all 8 stages', () => {
      for (let i = 1; i <= STAGE_COUNT; i++) {
        expect(ctrl._messages[i]).toBeTruthy();
      }
    });

    it('sets up buttons for all 8 stages', () => {
      for (let i = 1; i <= STAGE_COUNT; i++) {
        expect(ctrl._buttons[i]).toBeTruthy();
      }
    });

    it('resets bars on construction (stage 8 is final)', () => {
      const bar8 = ctrl._bars[8];
      expect(bar8.classList.contains('task-confirm-final')).toBe(true);
    });

    it('resets bars on construction (stages 1-7 are idle)', () => {
      for (let i = 1; i <= TRANSITION_STAGE_MAX; i++) {
        const bar = ctrl._bars[i];
        expect(bar.classList.contains('idle')).toBe(true);
      }
    });
  });

  // ── waitingStage / waitingToStage ─────────────────────────────

  describe('waitingStage / waitingToStage', () => {
    it('waitingStage getter returns null initially', () => {
      expect(ctrl.waitingStage).toBe(null);
    });

    it('waitingStage setter updates the value', () => {
      ctrl.waitingStage = 3;
      expect(ctrl.waitingStage).toBe(3);
    });

    it('waitingToStage getter returns null initially', () => {
      expect(ctrl.waitingToStage).toBe(null);
    });

    it('waitingToStage setter updates the value', () => {
      ctrl.waitingToStage = 5;
      expect(ctrl.waitingToStage).toBe(5);
    });
  });

  // ── resolveVisibleStage ───────────────────────────────────────

  describe('resolveVisibleStage', () => {
    it('returns waiting stage if valid and within TRANSITION_STAGE_MAX', () => {
      ctrl.waitingStage = 3;
      expect(ctrl.resolveVisibleStage(null)).toBe(3);
    });

    it('falls back to preferred stage when waiting stage is null', () => {
      ctrl.waitingStage = null;
      expect(ctrl.resolveVisibleStage(5)).toBe(5);
    });

    it('falls back to activeStage when both waiting and preferred are invalid', () => {
      ctrl.waitingStage = null;
      stageCtrl.activeStage = 4;
      expect(ctrl.resolveVisibleStage(null)).toBe(4);
    });

    it('defaults to 1 when all sources are invalid', () => {
      ctrl.waitingStage = null;
      stageCtrl.activeStage = NaN;
      expect(ctrl.resolveVisibleStage(null)).toBe(1);
    });

    it('ignores waiting stage if it exceeds TRANSITION_STAGE_MAX', () => {
      ctrl.waitingStage = 8;
      expect(ctrl.resolveVisibleStage(2)).toBe(2);
    });

    it('ignores preferred stage if it exceeds STAGE_COUNT', () => {
      ctrl.waitingStage = null;
      stageCtrl.activeStage = 6;
      expect(ctrl.resolveVisibleStage(99)).toBe(6);
    });
  });

  // ── setVisibleStage ───────────────────────────────────────────

  describe('setVisibleStage', () => {
    it('toggles is-visible class on the correct bar', () => {
      ctrl.setVisibleStage(3);
      expect(ctrl._bars[3].classList.contains('is-visible')).toBe(true);
    });

    it('removes is-visible from all other bars', () => {
      ctrl.setVisibleStage(3);
      for (let i = 1; i <= STAGE_COUNT; i++) {
        if (i === 3) continue;
        expect(ctrl._bars[i].classList.contains('is-visible')).toBe(false);
      }
    });

    it('resolves stage when called with null', () => {
      stageCtrl.activeStage = 5;
      ctrl.setVisibleStage(null);
      expect(ctrl._bars[5].classList.contains('is-visible')).toBe(true);
    });
  });

  // ── setState ──────────────────────────────────────────────────

  describe('setState', () => {
    it('sets idle state with correct class and disabled button', () => {
      ctrl.setState(2, 'idle', 'Idle message');
      const bar = ctrl._bars[2];
      expect(bar.classList.contains('idle')).toBe(true);
      expect(bar.classList.contains('waiting')).toBe(false);
      expect(ctrl._messages[2].textContent).toBe('Idle message');
      expect(ctrl._buttons[2].disabled).toBe(true);
      expect(ctrl._buttons[2].textContent).toBe('Continue');
    });

    it('sets waiting state with enabled button showing Continue', () => {
      ctrl.setState(3, 'waiting', 'Please confirm');
      const bar = ctrl._bars[3];
      expect(bar.classList.contains('waiting')).toBe(true);
      expect(ctrl._buttons[3].disabled).toBe(false);
      expect(ctrl._buttons[3].textContent).toBe('Continue');
    });

    it('sets sending state with disabled button showing Waiting...', () => {
      ctrl.setState(4, 'sending', 'Sending...');
      const bar = ctrl._bars[4];
      expect(bar.classList.contains('sending')).toBe(true);
      expect(ctrl._buttons[4].disabled).toBe(true);
      expect(ctrl._buttons[4].textContent).toBe('Waiting...');
    });

    it('sets confirmed state with disabled button showing Confirmed', () => {
      ctrl.setState(1, 'confirmed', 'Done');
      const bar = ctrl._bars[1];
      expect(bar.classList.contains('confirmed')).toBe(true);
      expect(ctrl._buttons[1].disabled).toBe(true);
      expect(ctrl._buttons[1].textContent).toBe('Confirmed');
    });

    it('removes previous state classes when setting a new state', () => {
      ctrl.setState(2, 'waiting', 'Wait');
      ctrl.setState(2, 'confirmed', 'Done');
      const bar = ctrl._bars[2];
      expect(bar.classList.contains('waiting')).toBe(false);
      expect(bar.classList.contains('confirmed')).toBe(true);
    });

    it('sets message text content', () => {
      ctrl.setState(1, 'idle', 'Custom message');
      expect(ctrl._messages[1].textContent).toBe('Custom message');
    });
  });

  // ── setState for final stage ──────────────────────────────────

  describe('setState for final stage', () => {
    it('adds task-confirm-final class for stage 8', () => {
      ctrl.setState(8, 'idle', 'Final');
      const bar = ctrl._bars[8];
      expect(bar.classList.contains('task-confirm-final')).toBe(true);
    });

    it('sets button to disabled with Done text for final stage', () => {
      ctrl.setState(8, 'idle', 'Final');
      expect(ctrl._buttons[8].disabled).toBe(true);
      expect(ctrl._buttons[8].textContent).toBe('Done');
    });

    it('uses default message when message is falsy for final stage', () => {
      ctrl.setState(8, 'idle', null);
      expect(ctrl._messages[8].textContent).toBe('Final stage. No next-stage confirmation.');
    });

    it('adds task-confirm-final class when state is explicitly "final"', () => {
      ctrl.setState(3, 'final', 'Pipeline done');
      const bar = ctrl._bars[3];
      expect(bar.classList.contains('task-confirm-final')).toBe(true);
      expect(ctrl._buttons[3].disabled).toBe(true);
      expect(ctrl._buttons[3].textContent).toBe('Done');
    });
  });

  // ── resetBars ─────────────────────────────────────────────────

  describe('resetBars', () => {
    it('resets all transition bars (1-7) to idle', () => {
      // First set some states
      ctrl.setState(1, 'confirmed', 'ok');
      ctrl.setState(2, 'waiting', 'wait');
      ctrl.resetBars();
      for (let i = 1; i <= TRANSITION_STAGE_MAX; i++) {
        expect(ctrl._bars[i].classList.contains('idle')).toBe(true);
      }
    });

    it('clears waitingStage and waitingToStage', () => {
      ctrl.waitingStage = 3;
      ctrl.waitingToStage = 4;
      ctrl.resetBars();
      expect(ctrl.waitingStage).toBe(null);
      expect(ctrl.waitingToStage).toBe(null);
    });

    it('sets stage 8 as final', () => {
      ctrl.resetBars();
      const bar8 = ctrl._bars[8];
      expect(bar8.classList.contains('task-confirm-final')).toBe(true);
      expect(ctrl._messages[8].textContent).toBe('Final stage. No next-stage confirmation.');
    });

    it('marks stages before resumeFromStage as confirmed', () => {
      ctrl.resetBars(4);
      for (let i = 1; i < 4; i++) {
        expect(ctrl._bars[i].classList.contains('confirmed')).toBe(true);
        expect(ctrl._messages[i].textContent).toContain('already completed before resume');
      }
    });

    it('marks stages at or after resumeFromStage as idle', () => {
      ctrl.resetBars(4);
      for (let i = 4; i <= TRANSITION_STAGE_MAX; i++) {
        expect(ctrl._bars[i].classList.contains('idle')).toBe(true);
      }
    });

    it('defaults to resumeFromStage=1 when argument is omitted', () => {
      ctrl.resetBars();
      expect(ctrl._bars[1].classList.contains('idle')).toBe(true);
    });
  });

  // ── syncFromStatus ────────────────────────────────────────────

  describe('syncFromStatus', () => {
    it('does nothing when status is null', () => {
      ctrl.syncFromStatus(null);
      // Should not throw
      expect(ctrl.waitingStage).toBe(null);
    });

    it('sets waiting state for a stage awaiting confirmation', () => {
      const status = mkStatus({
        running: true,
        current_stage: 2,
        stages: {
          '2': mkStage('complete', 100),
          '3': mkStage('pending', 0),
        },
        next_stage_confirmation: { required: true, from_stage: 2, to_stage: 3 },
      });
      ctrl.syncFromStatus(status);
      expect(ctrl.waitingStage).toBe(2);
      expect(ctrl.waitingToStage).toBe(3);
      expect(ctrl._bars[2].classList.contains('waiting')).toBe(true);
    });

    it('sets confirmed state for a transition-confirmed stage', () => {
      const status = mkStatus({
        running: true,
        current_stage: 3,
        stages: {
          '1': mkStage('complete', 100),
          '2': mkStage('complete', 100),
          '3': mkStage('running', 50),
        },
      });
      ctrl.syncFromStatus(status);
      expect(ctrl._bars[1].classList.contains('confirmed')).toBe(true);
      expect(ctrl._bars[2].classList.contains('confirmed')).toBe(true);
    });

    it('sets idle state for stages not yet complete', () => {
      const status = mkStatus({
        running: true,
        current_stage: 2,
        stages: {
          '1': mkStage('complete', 100),
          '2': mkStage('running', 50),
          '3': mkStage('pending', 0),
        },
      });
      ctrl.syncFromStatus(status);
      expect(ctrl._bars[3].classList.contains('idle')).toBe(true);
    });

    it('handles running stage idle message variant', () => {
      const status = mkStatus({
        running: true,
        current_stage: 3,
        stages: {
          '1': mkStage('complete', 100),
          '2': mkStage('complete', 100),
          '3': mkStage('running', 40),
        },
      });
      ctrl.syncFromStatus(status);
      expect(ctrl._messages[3].textContent).toContain('is running');
    });

    it('handles failed stage idle message variant', () => {
      const status = mkStatus({
        running: true,
        current_stage: 4,
        stages: {
          '4': mkStage('failed', 50),
        },
      });
      ctrl.syncFromStatus(status);
      expect(ctrl._messages[4].textContent).toContain('failed');
    });

    it('handles interactive stage idle message variant', () => {
      const status = mkStatus({
        running: true,
        current_stage: 2,
        stages: {
          '2': mkStage('interactive', 50),
        },
      });
      ctrl.syncFromStatus(status);
      expect(ctrl._messages[2].textContent).toContain('waiting for required interaction');
    });

    it('uses standby message when pipeline is not running and stage is done', () => {
      const status = mkStatus({
        running: false,
        stages: {
          '1': mkStage('complete', 100),
          '2': mkStage('pending', 0),
        },
      });
      ctrl.syncFromStatus(status);
      expect(ctrl._bars[1].classList.contains('idle')).toBe(true);
      expect(ctrl._messages[1].textContent).toContain('Start the pipeline');
    });

    it('handles final stage (8) done', () => {
      const stages = {};
      for (let i = 1; i <= 8; i++) {
        stages[String(i)] = mkStage('complete', 100);
      }
      const status = mkStatus({ running: false, stages });
      ctrl.syncFromStatus(status);
      expect(ctrl._bars[8].classList.contains('task-confirm-final')).toBe(true);
      expect(ctrl._messages[8].textContent).toContain('Final stage complete');
    });

    it('handles final stage (8) not done', () => {
      const status = mkStatus({ running: true, current_stage: 5, stages: {} });
      ctrl.syncFromStatus(status);
      expect(ctrl._messages[8].textContent).toBe('Final stage. No next-stage confirmation.');
    });

    it('marks earlier stages as confirmed when resume_from_stage is set', () => {
      // For the resume branch to fire, isTransitionConfirmed must return false
      // while stage < resumeStage and stageDone. isTransitionConfirmed returns
      // false when current_stage is not higher AND the next stage info is either
      // missing or still pending with 0 progress.
      // Here stage 2 is done, current_stage is null, and stage 3 is pending/0,
      // so isTransitionConfirmed(status, 2) returns false.
      const status = mkStatus({
        running: true,
        current_stage: null,
        resume_from_stage: 3,
        stages: {
          '2': mkStage('complete', 100),
          '3': mkStage('pending', 0),
        },
      });
      ctrl.syncFromStatus(status);
      // Stage 2 is done and < resumeStage(3), so confirmed with resume message
      expect(ctrl._bars[2].classList.contains('confirmed')).toBe(true);
      expect(ctrl._messages[2].textContent).toContain('already completed before resume');
    });

    it('clears waitingStage when confirmation is not required', () => {
      ctrl.waitingStage = 3;
      const status = mkStatus({
        running: true,
        current_stage: 4,
        stages: { '3': mkStage('complete', 100), '4': mkStage('running', 10) },
        next_stage_confirmation: { required: false },
      });
      ctrl.syncFromStatus(status);
      expect(ctrl.waitingStage).toBe(null);
    });

    it('uses custom confirmation message from status', () => {
      const status = mkStatus({
        running: true,
        current_stage: 2,
        stages: {
          '2': mkStage('complete', 100),
          '3': mkStage('pending', 0),
        },
        next_stage_confirmation: {
          required: true,
          from_stage: 2,
          to_stage: 3,
          message: 'Custom waiting message',
        },
      });
      ctrl.syncFromStatus(status);
      expect(ctrl._messages[2].textContent).toBe('Custom waiting message');
    });
  });

  // ── confirmNextStage ──────────────────────────────────────────

  describe('confirmNextStage', () => {
    beforeEach(() => {
      fetchMock.installFetch();
    });

    it('does nothing if waitingStage does not match the given stage', async () => {
      ctrl.waitingStage = 3;
      await ctrl.confirmNextStage(5);
      expect(globalThis.fetch).not.toHaveBeenCalled();
    });

    it('calls fetch on /api/pipeline/confirm-next with POST', async () => {
      ctrl.waitingStage = 2;
      ctrl.waitingToStage = 3;
      fetchMock.addRoute('/api/pipeline/confirm-next', fetchMock.jsonResponse({ status: 'confirmed' }));

      await ctrl.confirmNextStage(2);

      expect(globalThis.fetch).toHaveBeenCalledWith('/api/pipeline/confirm-next', { method: 'POST' });
    });

    it('sets sending state before fetch completes', async () => {
      ctrl.waitingStage = 2;
      ctrl.waitingToStage = 3;
      fetchMock.addRoute('/api/pipeline/confirm-next', fetchMock.jsonResponse({ status: 'confirmed' }));

      const promise = ctrl.confirmNextStage(2);
      // The sending state should be set synchronously before await
      expect(ctrl._bars[2].classList.contains('sending')).toBe(true);
      await promise;
    });

    it('handles success response (keeps sending state)', async () => {
      ctrl.waitingStage = 2;
      ctrl.waitingToStage = 3;
      fetchMock.addRoute('/api/pipeline/confirm-next', fetchMock.jsonResponse({ status: 'confirmed' }));

      await ctrl.confirmNextStage(2);

      // On normal success, stays in sending state waiting for backend event
      expect(ctrl._bars[2].classList.contains('sending')).toBe(true);
    });

    it('handles no_waiting_confirmation response', async () => {
      ctrl.waitingStage = 4;
      ctrl.waitingToStage = 5;
      stageCtrl.activeStage = 5;
      fetchMock.addRoute('/api/pipeline/confirm-next', fetchMock.jsonResponse({ status: 'no_waiting_confirmation' }));

      await ctrl.confirmNextStage(4);

      expect(ctrl.waitingStage).toBe(null);
      expect(ctrl.waitingToStage).toBe(null);
      expect(ctrl._bars[4].classList.contains('confirmed')).toBe(true);
    });

    it('handles error response (restores waiting state)', async () => {
      ctrl.waitingStage = 2;
      ctrl.waitingToStage = 3;
      fetchMock.addRoute('/api/pipeline/confirm-next', fetchMock.jsonResponse({ error: 'Server error' }, 500));

      await ctrl.confirmNextStage(2);

      expect(ctrl._bars[2].classList.contains('waiting')).toBe(true);
      expect(ctrl._messages[2].textContent).toContain('Server error');
    });

    it('calls appendLog on error', async () => {
      ctrl.waitingStage = 2;
      ctrl.waitingToStage = 3;
      fetchMock.addRoute('/api/pipeline/confirm-next', fetchMock.jsonResponse({ error: 'Boom' }, 500));

      await ctrl.confirmNextStage(2);

      expect(appendLog).toHaveBeenCalledWith(
        'stderr',
        expect.stringContaining('stage 2'),
        expect.objectContaining({ stage: 2 }),
      );
    });

    it('sending message says "Waiting for the next Stage N step" for same-stage transition', async () => {
      ctrl.waitingStage = 3;
      ctrl.waitingToStage = 3;
      fetchMock.addRoute('/api/pipeline/confirm-next', fetchMock.jsonResponse({ status: 'confirmed' }));

      const promise = ctrl.confirmNextStage(3);
      expect(ctrl._messages[3].textContent).toContain('Waiting for the next Stage 3 step');
      await promise;
    });

    it('sending message says "Waiting for Stage N start" for cross-stage transition', async () => {
      ctrl.waitingStage = 2;
      ctrl.waitingToStage = 3;
      fetchMock.addRoute('/api/pipeline/confirm-next', fetchMock.jsonResponse({ status: 'confirmed' }));

      const promise = ctrl.confirmNextStage(2);
      expect(ctrl._messages[2].textContent).toContain('Waiting for Stage 3 start');
      await promise;
    });
  });

  // ── resolveIdleMessage ────────────────────────────────────────

  describe('resolveIdleMessage', () => {
    it('returns running message when stage status is running', () => {
      const status = mkStatus({ stages: { '2': mkStage('running', 30) } });
      const msg = ctrl.resolveIdleMessage(status, 2);
      expect(msg).toContain('Stage 2 is running');
      expect(msg).toContain('after completion');
    });

    it('returns failed message when stage status is failed', () => {
      const status = mkStatus({ stages: { '3': mkStage('failed', 50) } });
      const msg = ctrl.resolveIdleMessage(status, 3);
      expect(msg).toContain('Stage 3 failed');
      expect(msg).toContain('Resolve the error');
    });

    it('returns interactive message when stage status is interactive', () => {
      const status = mkStatus({ stages: { '2': mkStage('interactive', 40) } });
      const msg = ctrl.resolveIdleMessage(status, 2);
      expect(msg).toContain('Stage 2 is waiting for required interaction');
    });

    it('returns default idle message for pending status', () => {
      const status = mkStatus({ stages: { '5': mkStage('pending', 0) } });
      const msg = ctrl.resolveIdleMessage(status, 5);
      expect(msg).toContain('Stage 5');
      expect(msg).toContain('next: Stage 6');
    });

    it('returns default idle message when stage info is missing', () => {
      const status = mkStatus({ stages: {} });
      const msg = ctrl.resolveIdleMessage(status, 4);
      expect(msg).toContain('Stage 4');
      expect(msg).toContain('next: Stage 5');
    });
  });
});
