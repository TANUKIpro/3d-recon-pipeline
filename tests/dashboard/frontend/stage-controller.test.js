import { describe, it, expect, vi, beforeEach } from 'vitest';
import { buildStageControllerDOM } from './helpers/dom-factory.js';
import { StageController } from '../../../scripts/dashboard/static/js/stage-controller.js';

describe('StageController', () => {
  let ctrl;

  beforeEach(() => {
    buildStageControllerDOM();
    ctrl = new StageController();
  });

  // ── constructor ──────────────────────────────────────────────

  describe('constructor', () => {
    it('defaults to stage 1 active', () => {
      expect(ctrl.activeStage).toBe(1);
      const panel1 = document.getElementById('stage-panel-1');
      expect(panel1.classList.contains('active')).toBe(true);
    });
  });

  // ── activateStage ────────────────────────────────────────────

  describe('activateStage', () => {
    it('shows panel for activated stage', () => {
      ctrl.activateStage(3);
      const panel3 = document.getElementById('stage-panel-3');
      expect(panel3.classList.contains('active')).toBe(true);
    });

    it('hides all other panels', () => {
      ctrl.activateStage(3);
      for (let i = 1; i <= 8; i++) {
        if (i === 3) continue;
        const panel = document.getElementById(`stage-panel-${i}`);
        expect(panel.classList.contains('active')).toBe(false);
      }
    });

    it('sets selected class on the activated pill', () => {
      ctrl.activateStage(3);
      const pill3 = document.querySelector('.stage-pill[data-stage="3"]');
      expect(pill3.classList.contains('selected')).toBe(true);
    });

    it('dispatches stage-activated CustomEvent with stage in detail', () => {
      const handler = vi.fn();
      document.addEventListener('stage-activated', handler);

      ctrl.activateStage(4);

      expect(handler).toHaveBeenCalledTimes(1);
      const event = handler.mock.calls[0][0];
      expect(event.detail).toEqual({ stage: 4 });

      document.removeEventListener('stage-activated', handler);
    });

    it('selects active mesh-method-pill for stage 5 instead of regular pill', () => {
      ctrl.activateStage(5);

      // The active mesh-method-pill should get 'selected'
      const activeMethodPill = document.querySelector('.mesh-method-pill.active');
      expect(activeMethodPill.classList.contains('selected')).toBe(true);

      // The regular stage-pill for stage 5 should NOT get 'selected'
      const regularPill5 = document.querySelector('.stage-pill[data-stage="5"]');
      expect(regularPill5.classList.contains('selected')).toBe(false);
    });
  });

  // ── setStageState / getStageState ────────────────────────────

  describe('setStageState / getStageState', () => {
    it('stores and retrieves state', () => {
      ctrl.setStageState(2, 'complete');
      expect(ctrl.getStageState(2)).toBe('complete');
    });

    it('returns pending for unset stage', () => {
      expect(ctrl.getStageState(99)).toBe('pending');
    });
  });

  // ── pill click ───────────────────────────────────────────────

  describe('pill click', () => {
    it('activates corresponding stage when pill is clicked', () => {
      const pill6 = document.querySelector('.stage-pill[data-stage="6"]');
      pill6.click();
      expect(ctrl.activeStage).toBe(6);
      const panel6 = document.getElementById('stage-panel-6');
      expect(panel6.classList.contains('active')).toBe(true);
    });
  });

  // ── activeStage getter ───────────────────────────────────────

  describe('activeStage getter', () => {
    it('returns current stage number after activation', () => {
      ctrl.activateStage(7);
      expect(ctrl.activeStage).toBe(7);
    });
  });
});
