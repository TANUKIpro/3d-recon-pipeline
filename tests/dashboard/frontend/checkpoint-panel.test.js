import { describe, it, expect, vi, beforeEach } from 'vitest';
import { buildCheckpointPanelDOM } from './helpers/dom-factory.js';
import { CheckpointPanel } from '../../../scripts/dashboard/static/js/checkpoint-panel.js';

/**
 * Helper: returns an array of { className, indicator, text } for each
 * checkpoint <li> in the rendered list.
 */
function readCheckpointItems() {
  const list = document.getElementById('checkpoint-list');
  return Array.from(list.children).map((li) => ({
    className: li.className,
    indicator: li.querySelector('.checkpoint-indicator')?.textContent || '',
    text: li.querySelector('.checkpoint-text')?.textContent || '',
  }));
}

/** Helper: returns just the states (e.g. 'running', 'complete') from class names. */
function readStates() {
  return readCheckpointItems().map((item) => {
    const match = item.className.match(/checkpoint-(running|complete|failed|pending)/);
    return match ? match[1] : 'unknown';
  });
}

/** Helper: returns just the indicator tokens. */
function readIndicators() {
  return readCheckpointItems().map((item) => item.indicator);
}

describe('CheckpointPanel', () => {
  let panel;

  beforeEach(() => {
    buildCheckpointPanelDOM();
    panel = new CheckpointPanel();
  });

  // --- Module helper functions (tested indirectly through class behavior) ---

  describe('module helpers (indirect)', () => {
    it('clampStage: NaN defaults to 1', () => {
      // onStageStart with NaN should affect stage 1
      panel.reset(1);
      panel.onStageStart(NaN);
      panel.setActiveStage(1);
      const states = readStates();
      expect(states.some((s) => s === 'running')).toBe(true);
    });

    it('clampStage: below min clamps to 1, above max clamps to 8', () => {
      panel.reset(1);
      panel.onStageStart(-5);
      panel.setActiveStage(1);
      expect(readStates().some((s) => s === 'running')).toBe(true);

      panel.reset(1);
      panel.onStageStart(99);
      panel.setActiveStage(8);
      expect(readStates().some((s) => s === 'running')).toBe(true);
    });

    it('clampStage: float is rounded', () => {
      panel.reset(1);
      panel.onStageStart(2.7); // rounds to 3
      panel.setActiveStage(3);
      expect(readStates().some((s) => s === 'running')).toBe(true);
    });

    it('clampStage: valid value passes through', () => {
      panel.reset(1);
      panel.onStageStart(4);
      panel.setActiveStage(4);
      expect(readStates().some((s) => s === 'running')).toBe(true);
    });

    it('normalizeStatus: unknown maps to pending', () => {
      panel.applyStatusSnapshot({
        stages: { '1': { status: 'banana' } },
      });
      panel.setActiveStage(1);
      expect(readStates().every((s) => s === 'pending')).toBe(true);
    });

    it('normalizeMeshMethod: unknown defaults to poisson', () => {
      panel.setMeshMethod('unknown_method');
      panel.setActiveStage(5);
      const items = readCheckpointItems();
      // poisson template: "Reconstruct mesh (Poisson)" is checkpoint index 1
      expect(items.some((i) => i.text.includes('Poisson'))).toBe(true);
    });

    it('isTransitionConfirmationDetail: "continue to" triggers all complete', () => {
      panel.applyStatusSnapshot({
        stages: { '2': { status: 'interactive', detail: 'continue to stage 3' } },
      });
      panel.setActiveStage(2);
      expect(readStates().every((s) => s === 'complete')).toBe(true);
    });

    it('isTransitionConfirmationDetail: "stage N complete" triggers all complete', () => {
      panel.applyStatusSnapshot({
        stages: { '1': { status: 'interactive', detail: 'stage 1 complete' } },
      });
      panel.setActiveStage(1);
      expect(readStates().every((s) => s === 'complete')).toBe(true);
    });

    it('isTransitionConfirmationDetail: null does not trigger', () => {
      panel.applyStatusSnapshot({
        stages: { '1': { status: 'interactive', detail: null } },
      });
      panel.setActiveStage(1);
      // interactive without transition detail -> should have a running checkpoint
      expect(readStates().some((s) => s === 'running')).toBe(true);
    });

    it('stateToken: maps to correct indicators', () => {
      panel.reset(1);

      // Stage 1 pending -> all [ ]
      panel.setActiveStage(1);
      expect(readIndicators().every((ind) => ind === '[ ]')).toBe(true);

      // Stage 1 running -> first is [o]
      panel.onStageStart(1);
      const runningIndicators = readIndicators();
      expect(runningIndicators[0]).toBe('[o]');

      // Stage 1 complete -> all [x]
      panel.onStageComplete(1);
      expect(readIndicators().every((ind) => ind === '[x]')).toBe(true);

      // Stage 2 failed -> first is [!]
      panel.onStageFailed(2);
      panel.setActiveStage(2);
      expect(readIndicators()[0]).toBe('[!]');
    });
  });

  // --- _resolveCurrentIndex ---

  describe('_resolveCurrentIndex (via onStageProgress)', () => {
    it('checkpoint_id match returns correct index', () => {
      panel.onStageStart(1);
      panel.onStageProgress(1, 'some detail', 'running', 's1.finalize');
      panel.setActiveStage(1);
      const states = readStates();
      // s1.finalize is index 2 (last); front checkpoints should be complete
      expect(states[0]).toBe('complete');
      expect(states[1]).toBe('complete');
      expect(states[2]).toBe('running');
    });

    it('detail regex fallback works', () => {
      // Use applyStatusSnapshot to set running status with detail but no checkpoint_id,
      // so _resolveCurrentIndex falls through to the detail regex matchers.
      panel.applyStatusSnapshot({
        stages: {
          '1': { status: 'running', detail: 'extracting frames now' },
        },
      });
      panel.setActiveStage(1);
      const states = readStates();
      // "extracting frames" matches idx 1 in DETAIL_MATCHERS[1]
      expect(states[0]).toBe('complete');
      expect(states[1]).toBe('running');
    });

    it('stage 7 ground-plane detail advances to repair checkpoint', () => {
      panel.applyStatusSnapshot({
        stages: {
          '7': { status: 'running', detail: 'Contact Hole Repair: scanning clip heights' },
        },
      });
      panel.setActiveStage(7);
      const states = readStates();
      expect(states[0]).toBe('complete');
      expect(states[1]).toBe('running');
    });

    it('"complete" in detail resolves to maxIndex', () => {
      // Use applyStatusSnapshot with detail containing "complete" but no checkpoint_id,
      // so _resolveCurrentIndex falls through to the /complete/i fallback -> maxIndex.
      panel.applyStatusSnapshot({
        stages: {
          '4': { status: 'running', detail: 'stage complete' },
        },
      });
      panel.setActiveStage(4);
      const states = readStates();
      // Stage 4 has 5 checkpoints; "complete" fallback -> maxIndex = 4
      const lastIndex = states.length - 1;
      expect(states[lastIndex]).toBe('running');
    });

    it('stage 4 filter detail maps to sparse filter checkpoint', () => {
      panel.applyStatusSnapshot({
        stages: {
          '4': { status: 'running', detail: 'Filtering COLMAP sparse points with SAM2 masks' },
        },
      });
      panel.setActiveStage(4);
      const states = readStates();
      expect(states[0]).toBe('running');
    });

    it('no match uses previous index', () => {
      panel.onStageStart(2);
      // First, advance to index 1 via checkpoint_id
      panel.onStageProgress(2, 'running pi3x fitting', 'running', 's2.infer');
      // Then send a detail that doesn't match anything and no checkpoint_id
      panel.onStageProgress(2, 'some unknown detail text xyz');
      panel.setActiveStage(2);
      const states = readStates();
      // Should remain at previous index 1
      expect(states[0]).toBe('complete');
      expect(states[1]).toBe('running');
    });
  });

  // --- _resolveCheckpointStates ---

  describe('_resolveCheckpointStates', () => {
    it('complete status sets all checkpoints complete', () => {
      panel.onStageComplete(1);
      panel.setActiveStage(1);
      expect(readStates().every((s) => s === 'complete')).toBe(true);
    });

    it('pending status sets all checkpoints pending', () => {
      panel.reset(1);
      panel.setActiveStage(1);
      expect(readStates().every((s) => s === 'pending')).toBe(true);
    });

    it('running: front complete, current running', () => {
      panel.onStageStart(2);
      panel.onStageProgress(2, 'running pi3x fitting', 'running', 's2.infer');
      panel.setActiveStage(2);
      const states = readStates();
      // s2.infer is index 1
      expect(states[0]).toBe('complete');
      expect(states[1]).toBe('running');
      expect(states[2]).toBe('pending');
    });

    it('failed: current checkpoint is failed', () => {
      panel.onStageStart(1);
      panel.onStageFailed(1, 'something broke', 's1.extract');
      panel.setActiveStage(1);
      const states = readStates();
      // s1.extract is index 1
      expect(states[0]).toBe('complete');
      expect(states[1]).toBe('failed');
      expect(states[2]).toBe('pending');
    });

    it('stage 5 uses poisson templates by default', () => {
      panel.setActiveStage(5);
      const items = readCheckpointItems();
      expect(items.length).toBe(4);
      expect(items.some((i) => i.text.includes('Poisson'))).toBe(true);
    });

    it('setMeshMethod("diffcd") switches stage 5 templates', () => {
      panel.setMeshMethod('diffcd');
      panel.setActiveStage(5);
      const items = readCheckpointItems();
      expect(items.length).toBe(4);
      expect(items.some((i) => i.text.includes('DiffCD'))).toBe(true);
      expect(items.some((i) => i.text.includes('Poisson'))).toBe(false);
    });

    it('"waiting for next-stage confirmation" sets all complete', () => {
      panel.onStageStart(1);
      panel.onStageProgress(1, 'waiting for next-stage confirmation', 'running');
      panel.setActiveStage(1);
      expect(readStates().every((s) => s === 'complete')).toBe(true);
    });
  });

  // --- Lifecycle ---

  describe('lifecycle', () => {
    it('reset(1): all stages pending', () => {
      panel.reset(1);
      // Check stage 1
      panel.setActiveStage(1);
      expect(readStates().every((s) => s === 'pending')).toBe(true);
    });

    it('reset(3): stages 1-2 complete, 3-8 pending', () => {
      panel.reset(3);
      // Stage 1 should be complete
      panel.setActiveStage(1);
      expect(readStates().every((s) => s === 'complete')).toBe(true);

      // Stage 2 should be complete
      panel.setActiveStage(2);
      expect(readStates().every((s) => s === 'complete')).toBe(true);

      // Stage 3 should be pending
      panel.setActiveStage(3);
      expect(readStates().every((s) => s === 'pending')).toBe(true);

      // Stage 8 should be pending
      panel.setActiveStage(8);
      expect(readStates().every((s) => s === 'pending')).toBe(true);
    });

    it('onStageStart sets running', () => {
      panel.onStageStart(1);
      panel.setActiveStage(1);
      expect(readStates().some((s) => s === 'running')).toBe(true);
    });

    it('onStageComplete sets complete', () => {
      panel.onStageStart(1);
      panel.onStageComplete(1);
      panel.setActiveStage(1);
      expect(readStates().every((s) => s === 'complete')).toBe(true);
    });

    it('onStageFailed sets failed', () => {
      panel.onStageStart(1);
      panel.onStageFailed(1, 'error occurred');
      panel.setActiveStage(1);
      expect(readStates().some((s) => s === 'failed')).toBe(true);
    });
  });

  // --- applyStatusSnapshot ---

  describe('applyStatusSnapshot', () => {
    it('applies status for all stages', () => {
      panel.applyStatusSnapshot({
        stages: {
          '1': { status: 'complete' },
          '2': { status: 'complete' },
          '3': { status: 'running', detail: 'propagating masks', checkpoint_id: 's3.propagate' },
          '4': { status: 'pending' },
        },
      });

      panel.setActiveStage(1);
      expect(readStates().every((s) => s === 'complete')).toBe(true);

      panel.setActiveStage(3);
      expect(readStates().some((s) => s === 'running')).toBe(true);

      panel.setActiveStage(4);
      expect(readStates().every((s) => s === 'pending')).toBe(true);
    });

    it('missing stages default to pending', () => {
      panel.applyStatusSnapshot({
        stages: {
          '1': { status: 'complete' },
          // stages 2-8 are missing
        },
      });

      panel.setActiveStage(2);
      expect(readStates().every((s) => s === 'pending')).toBe(true);

      panel.setActiveStage(5);
      expect(readStates().every((s) => s === 'pending')).toBe(true);
    });

    it('mesh_method from snapshot applied', () => {
      panel.applyStatusSnapshot({
        mesh_method: 'diffcd',
        stages: {
          '5': { status: 'running', detail: 'running diffcd fitting' },
        },
      });

      panel.setActiveStage(5);
      const items = readCheckpointItems();
      expect(items.some((i) => i.text.includes('DiffCD'))).toBe(true);
      expect(items.some((i) => i.text.includes('Poisson'))).toBe(false);
    });

    it('interactive status with transition detail sets all complete', () => {
      panel.applyStatusSnapshot({
        stages: {
          '2': { status: 'interactive', detail: 'confirm to continue to stage 3' },
        },
      });

      panel.setActiveStage(2);
      expect(readStates().every((s) => s === 'complete')).toBe(true);
    });

    it('checkpointId from snapshot used for index resolution', () => {
      panel.applyStatusSnapshot({
        stages: {
          '2': { status: 'running', detail: 'working', checkpoint_id: 's2.filter' },
        },
      });

      panel.setActiveStage(2);
      const states = readStates();
      // s2.filter is index 2; front should be complete, index 2 running
      expect(states[0]).toBe('complete');
      expect(states[1]).toBe('complete');
      expect(states[2]).toBe('running');
    });

    it('render updates DOM list items', () => {
      const list = document.getElementById('checkpoint-list');
      panel.applyStatusSnapshot({
        stages: {
          '1': { status: 'complete' },
        },
      });
      panel.setActiveStage(1);

      // Verify DOM <li> elements exist with correct classes
      const items = list.querySelectorAll('.checkpoint-item');
      expect(items.length).toBeGreaterThan(0);

      for (const item of items) {
        expect(item.classList.contains('checkpoint-complete')).toBe(true);
        const indicator = item.querySelector('.checkpoint-indicator');
        expect(indicator.textContent).toBe('[x]');
      }
    });
  });
});
