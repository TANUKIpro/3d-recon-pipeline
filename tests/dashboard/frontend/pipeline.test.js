import { describe, it, expect, vi, beforeEach } from 'vitest';
import { buildPipelineDOM } from './helpers/dom-factory.js';
import { PipelineUI } from '../../../scripts/dashboard/static/js/pipeline.js';

describe('PipelineUI', () => {
  let ui;

  beforeEach(() => {
    buildPipelineDOM();
    ui = new PipelineUI();
  });

  // ── _formatTime ──────────────────────────────────────────────

  describe('_formatTime', () => {
    const fmt = (secs) => PipelineUI.prototype._formatTime.call(ui, secs);

    it('returns empty string for null', () => {
      expect(fmt(null)).toBe('');
    });

    it('returns "45s" for 45', () => {
      expect(fmt(45)).toBe('45s');
    });

    it('returns "2m5s" for 125', () => {
      expect(fmt(125)).toBe('2m5s');
    });

    it('returns "0s" for 0', () => {
      expect(fmt(0)).toBe('0s');
    });
  });

  // ── _normalizeProgress ───────────────────────────────────────

  describe('_normalizeProgress', () => {
    const norm = (v) => PipelineUI.prototype._normalizeProgress.call(ui, v);

    it('returns 0 for NaN', () => {
      expect(norm(NaN)).toBe(0);
    });

    it('returns 0 for negative values', () => {
      expect(norm(-5)).toBe(0);
    });

    it('returns 100 for values above 100', () => {
      expect(norm(150)).toBe(100);
    });

    it('passes through valid values', () => {
      expect(norm(50)).toBe(50);
    });
  });

  // ── setMeshMethod / getMeshMethod ────────────────────────────

  describe('setMeshMethod', () => {
    it('activates poisson pill and deactivates diffcd', () => {
      ui.setMeshMethod('poisson');
      const poisson = document.getElementById('mesh-pill-poisson');
      const diffcd = document.getElementById('mesh-pill-diffcd');
      expect(poisson.classList.contains('active')).toBe(true);
      expect(diffcd.classList.contains('active')).toBe(false);
      expect(ui.getMeshMethod()).toBe('poisson');
    });

    it('activates diffcd pill and deactivates poisson', () => {
      ui.setMeshMethod('diffcd');
      const poisson = document.getElementById('mesh-pill-poisson');
      const diffcd = document.getElementById('mesh-pill-diffcd');
      expect(diffcd.classList.contains('active')).toBe(true);
      expect(poisson.classList.contains('active')).toBe(false);
      expect(ui.getMeshMethod()).toBe('diffcd');
    });

    it('defaults to poisson for invalid method', () => {
      ui.setMeshMethod('invalid');
      expect(ui.getMeshMethod()).toBe('poisson');
      const poisson = document.getElementById('mesh-pill-poisson');
      expect(poisson.classList.contains('active')).toBe(true);
    });
  });

  // ── stageStart ───────────────────────────────────────────────

  describe('stageStart', () => {
    it('adds running class to pill', () => {
      ui.stageStart(1);
      const pill = document.querySelector('.stage-pill[data-stage="1"]');
      expect(pill.classList.contains('running')).toBe(true);
    });

    it('starts a timer that updates time element after 1s', () => {
      vi.useFakeTimers();
      try {
        ui.stageStart(2);
        const pill = document.querySelector('.stage-pill[data-stage="2"]');
        const timeEl = pill.querySelector('.stage-time');

        vi.advanceTimersByTime(1000);
        expect(timeEl.textContent).not.toBe('');
      } finally {
        vi.useRealTimers();
      }
    });
  });

  // ── stageComplete ────────────────────────────────────────────

  describe('stageComplete', () => {
    it('adds complete class and sets progress to 100%', () => {
      ui.stageStart(1);
      ui.stageComplete(1, 10);
      const pill = document.querySelector('.stage-pill[data-stage="1"]');
      expect(pill.classList.contains('complete')).toBe(true);
      expect(pill.classList.contains('running')).toBe(false);
      const progressEl = pill.querySelector('.stage-progress');
      expect(progressEl.textContent).toBe('100%');
    });

    it('stops the timer', () => {
      vi.useFakeTimers();
      try {
        ui.stageStart(3);
        ui.stageComplete(3, 5);

        const pill = document.querySelector('.stage-pill[data-stage="3"]');
        const timeEl = pill.querySelector('.stage-time');
        const textAfterComplete = timeEl.textContent;

        vi.advanceTimersByTime(2000);
        expect(timeEl.textContent).toBe(textAfterComplete);
      } finally {
        vi.useRealTimers();
      }
    });
  });

  // ── stageProgress ────────────────────────────────────────────

  describe('stageProgress', () => {
    it('updates progress percentage', () => {
      ui.stageStart(1);
      ui.stageProgress(1, 42, 'processing');
      const pill = document.querySelector('.stage-pill[data-stage="1"]');
      const progressEl = pill.querySelector('.stage-progress');
      expect(progressEl.textContent).toBe('42%');
    });

    it('updates detail as tooltip', () => {
      ui.stageStart(1);
      ui.stageProgress(1, 50, 'Encoding frame 5/10');
      const pill = document.querySelector('.stage-pill[data-stage="1"]');
      expect(pill.title).toBe('Encoding frame 5/10');
    });
  });

  // ── stageFailed ──────────────────────────────────────────────

  describe('stageFailed', () => {
    it('adds failed class to pill', () => {
      ui.stageStart(4);
      ui.stageFailed(4);
      const pill = document.querySelector('.stage-pill[data-stage="4"]');
      expect(pill.classList.contains('failed')).toBe(true);
      expect(pill.classList.contains('running')).toBe(false);
    });
  });

  // ── stageInteractive ────────────────────────────────────────

  describe('stageInteractive', () => {
    it('adds interactive class to pill', () => {
      ui.stageInteractive(2);
      const pill = document.querySelector('.stage-pill[data-stage="2"]');
      expect(pill.classList.contains('interactive')).toBe(true);
    });
  });

  // ── reset ────────────────────────────────────────────────────

  describe('reset', () => {
    it('resets all stages to pending and clears timers', () => {
      vi.useFakeTimers();
      try {
        ui.stageStart(1);
        ui.stageComplete(2, 5);
        ui.stageProgress(3, 60, 'halfway');

        ui.reset();

        for (let i = 1; i <= 8; i++) {
          expect(ui.getStageStatus(i)).toBe('pending');
          const pill = document.querySelector(`.stage-pill[data-stage="${i}"]`);
          expect(pill.classList.contains('running')).toBe(false);
          expect(pill.classList.contains('complete')).toBe(false);
          expect(pill.classList.contains('failed')).toBe(false);
        }

        // Timer for stage 1 should have been cleared
        const pill1 = document.querySelector('.stage-pill[data-stage="1"]');
        const timeEl1 = pill1.querySelector('.stage-time');
        const textAfterReset = timeEl1.textContent;
        vi.advanceTimersByTime(2000);
        expect(timeEl1.textContent).toBe(textAfterReset);
      } finally {
        vi.useRealTimers();
      }
    });
  });

  // ── resetFromStage ───────────────────────────────────────────

  describe('resetFromStage', () => {
    it('keeps stages before startStage unchanged and resets stages from startStage onward', () => {
      ui.stageComplete(1, 3);
      ui.stageComplete(2, 5);
      ui.stageStart(3);
      ui.stageProgress(4, 40, 'working');

      ui.resetFromStage(3);

      // Stages 1-2 should retain their status
      expect(ui.getStageStatus(1)).toBe('complete');
      expect(ui.getStageStatus(2)).toBe('complete');

      // Stages 3+ should be reset
      for (let i = 3; i <= 8; i++) {
        expect(ui.getStageStatus(i)).toBe('pending');
      }
    });
  });

  // ── getOverallProgress ───────────────────────────────────────

  describe('getOverallProgress', () => {
    it('returns average progress of all stages', () => {
      // Stage 1 = 100, stages 2-8 = 0 => average = 100/8 = 12.5
      ui.stageComplete(1, 3);
      expect(ui.getOverallProgress()).toBe(12.5);
    });
  });

  // ── getStageStatus ───────────────────────────────────────────

  describe('getStageStatus', () => {
    it('returns correct status for each lifecycle state', () => {
      expect(ui.getStageStatus(1)).toBe('pending');

      ui.stageStart(1);
      expect(ui.getStageStatus(1)).toBe('running');

      ui.stageComplete(1, 5);
      expect(ui.getStageStatus(1)).toBe('complete');
    });
  });

  // ── updateAll ────────────────────────────────────────────────

  describe('updateAll', () => {
    it('applies status dict to all stages', () => {
      ui.updateAll({
        stages: {
          '1': { status: 'complete', progress: 100, elapsed: 3 },
          '2': { status: 'running', progress: 45, detail: 'encoding' },
          '3': { status: 'failed', progress: 20 },
        },
      });

      expect(ui.getStageStatus(1)).toBe('complete');
      expect(ui.getStageStatus(2)).toBe('running');
      expect(ui.getStageStatus(3)).toBe('failed');
      // Unmentioned stages remain pending
      expect(ui.getStageStatus(4)).toBe('pending');

      const pill2 = document.querySelector('.stage-pill[data-stage="2"]');
      expect(pill2.title).toBe('encoding');
    });
  });
});
