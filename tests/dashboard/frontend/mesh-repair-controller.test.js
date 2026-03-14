import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { useFetchMock } from './helpers/fetch-mock.js';
import { buildMeshRepairDOM } from './helpers/dom-factory.js';
import { MeshRepairController } from '../../../scripts/dashboard/static/js/mesh-repair-controller.js';
import { MESH_REPAIR_THRESHOLD_DEFAULT } from '../../../scripts/dashboard/static/js/constants.js';
import { formatMeshRepairThreshold } from '../../../scripts/dashboard/static/js/utils.js';

const { installFetch, restoreFetch, addRoute, clearRoutes, jsonResponse } = useFetchMock();

beforeEach(() => {
  installFetch();
  clearRoutes();
});

afterEach(() => {
  restoreFetch();
});

function createMockPreview() {
  return {
    clearMeshRepairSelection: vi.fn(),
    setMeshRepairThreshold: vi.fn(),
    getMeshRepairSelectedLoopIds: vi.fn(() => []),
    getMeshRepairTotalLoopCount: vi.fn(() => 0),
    getMeshRepairVisibleLoopCount: vi.fn(() => 0),
    beginMeshRepairSelection: vi.fn(async () => {}),
    setMeshRepairConfirmed: vi.fn(),
  };
}

describe('MeshRepairController', () => {
  let controller;
  let preview;
  let appendLog;

  beforeEach(() => {
    buildMeshRepairDOM();
    preview = createMockPreview();
    appendLog = vi.fn();
    controller = new MeshRepairController({ preview, appendLog });
  });

  // --- init ---

  describe('init', () => {
    it('sets toolbar hidden', () => {
      controller.init();
      const toolbar = document.getElementById('mesh-repair-toolbar');
      expect(toolbar.style.display).toBe('none');
    });

    it('sets controls disabled', () => {
      controller.init();
      expect(document.getElementById('mesh-repair-threshold').disabled).toBe(true);
      expect(document.getElementById('mesh-repair-clear').disabled).toBe(true);
      expect(document.getElementById('mesh-repair-apply').disabled).toBe(true);
    });
  });

  // --- setToolbarVisible ---

  describe('setToolbarVisible', () => {
    it('shows toolbar when passed true', () => {
      controller.setToolbarVisible(true);
      const toolbar = document.getElementById('mesh-repair-toolbar');
      expect(toolbar.style.display).toBe('flex');
    });

    it('hides toolbar when passed false', () => {
      controller.setToolbarVisible(true);
      controller.setToolbarVisible(false);
      const toolbar = document.getElementById('mesh-repair-toolbar');
      expect(toolbar.style.display).toBe('none');
    });
  });

  // --- setEnabled ---

  describe('setEnabled', () => {
    it('enables threshold input and buttons when passed true', () => {
      controller.setEnabled(false);
      controller.setEnabled(true);
      expect(document.getElementById('mesh-repair-threshold').disabled).toBe(false);
      expect(document.getElementById('mesh-repair-clear').disabled).toBe(false);
      expect(document.getElementById('mesh-repair-apply').disabled).toBe(false);
    });

    it('disables threshold input and buttons when passed false', () => {
      controller.setEnabled(true);
      controller.setEnabled(false);
      expect(document.getElementById('mesh-repair-threshold').disabled).toBe(true);
      expect(document.getElementById('mesh-repair-clear').disabled).toBe(true);
      expect(document.getElementById('mesh-repair-apply').disabled).toBe(true);
    });
  });

  // --- setStatus ---

  describe('setStatus', () => {
    it('sets text content on the status element', () => {
      controller.setStatus('Some status message');
      const statusEl = document.getElementById('mesh-repair-status');
      expect(statusEl.textContent).toBe('Some status message');
    });

    it('adds error class when tone is error', () => {
      controller.setStatus('Error occurred', 'error');
      const statusEl = document.getElementById('mesh-repair-status');
      expect(statusEl.classList.contains('error')).toBe(true);
      expect(statusEl.classList.contains('success')).toBe(false);
    });

    it('adds success class when tone is success', () => {
      controller.setStatus('All good', 'success');
      const statusEl = document.getElementById('mesh-repair-status');
      expect(statusEl.classList.contains('success')).toBe(true);
      expect(statusEl.classList.contains('error')).toBe(false);
    });

    it('removes previous tone classes when tone is empty', () => {
      controller.setStatus('Error', 'error');
      controller.setStatus('No tone');
      const statusEl = document.getElementById('mesh-repair-status');
      expect(statusEl.classList.contains('error')).toBe(false);
      expect(statusEl.classList.contains('success')).toBe(false);
    });
  });

  // --- setThresholdValue ---

  describe('setThresholdValue', () => {
    it('clamps value within range and updates input and display', () => {
      const result = controller.setThresholdValue(-0.5);
      expect(result).toBe(-0.5);
      const input = document.getElementById('mesh-repair-threshold');
      const display = document.getElementById('mesh-repair-threshold-value');
      expect(input.value).toBe('-0.50');
      expect(display.textContent).toBe('-0.50');
    });

    it('clamps value above max to 0.00', () => {
      const result = controller.setThresholdValue(5);
      expect(result).toBe(0);
      const display = document.getElementById('mesh-repair-threshold-value');
      expect(display.textContent).toBe('0.00');
    });

    it('clamps value below min to -1.00', () => {
      const result = controller.setThresholdValue(-10);
      expect(result).toBe(-1);
      const display = document.getElementById('mesh-repair-threshold-value');
      expect(display.textContent).toBe('-1.00');
    });

    it('calls preview.setMeshRepairThreshold when applyPreview is true (default)', () => {
      controller.setThresholdValue(-0.3);
      expect(preview.setMeshRepairThreshold).toHaveBeenCalledWith(-0.3);
    });

    it('does not call preview.setMeshRepairThreshold when applyPreview is false', () => {
      controller.setThresholdValue(-0.3, { applyPreview: false });
      expect(preview.setMeshRepairThreshold).not.toHaveBeenCalled();
    });
  });

  // --- resetThreshold ---

  describe('resetThreshold', () => {
    it('resets to default threshold value', () => {
      controller.setThresholdValue(-0.8, { applyPreview: false });
      const result = controller.resetThreshold({ applyPreview: false });
      expect(result).toBe(MESH_REPAIR_THRESHOLD_DEFAULT);
      const display = document.getElementById('mesh-repair-threshold-value');
      expect(display.textContent).toBe(formatMeshRepairThreshold(MESH_REPAIR_THRESHOLD_DEFAULT));
    });
  });

  // --- resetState ---

  describe('resetState', () => {
    it('resets candidateCount to 0', () => {
      controller._candidateCount = 10;
      controller.resetState();
      expect(controller.candidateCount).toBe(0);
    });

    it('resets inFlight to false', () => {
      controller._inFlight = true;
      controller.resetState();
      expect(controller.inFlight).toBe(false);
    });

    it('resets threshold to default when resetThreshold option is true', () => {
      controller.setThresholdValue(-0.9, { applyPreview: false });
      controller.resetState({ resetThreshold: true });
      expect(controller._threshold).toBe(MESH_REPAIR_THRESHOLD_DEFAULT);
    });

    it('preserves threshold when resetThreshold option is false', () => {
      controller.setThresholdValue(-0.9, { applyPreview: false });
      controller.resetState({ resetThreshold: false });
      expect(controller._threshold).toBe(-0.9);
    });
  });

  // --- getVisibleCount ---

  describe('getVisibleCount', () => {
    it('uses preview visible count when available and total is positive', () => {
      preview.getMeshRepairTotalLoopCount.mockReturnValue(10);
      preview.getMeshRepairVisibleLoopCount.mockReturnValue(7);
      controller._candidateCount = 10;
      expect(controller.getVisibleCount(10)).toBe(7);
    });

    it('falls back to candidateCount when preview visible is NaN', () => {
      preview.getMeshRepairTotalLoopCount.mockReturnValue(10);
      preview.getMeshRepairVisibleLoopCount.mockReturnValue(NaN);
      controller._candidateCount = 5;
      expect(controller.getVisibleCount(5)).toBe(5);
    });

    it('falls back to candidateCount when preview total is 0', () => {
      preview.getMeshRepairTotalLoopCount.mockReturnValue(0);
      preview.getMeshRepairVisibleLoopCount.mockReturnValue(0);
      controller._candidateCount = 8;
      expect(controller.getVisibleCount(8)).toBe(8);
    });

    it('clamps visible to be at most candidateCount', () => {
      preview.getMeshRepairTotalLoopCount.mockReturnValue(20);
      preview.getMeshRepairVisibleLoopCount.mockReturnValue(15);
      // candidateCount is 10, so visible should be clamped to 10
      expect(controller.getVisibleCount(10)).toBe(10);
    });

    it('returns 0 for negative candidateCount', () => {
      preview.getMeshRepairTotalLoopCount.mockReturnValue(0);
      preview.getMeshRepairVisibleLoopCount.mockReturnValue(0);
      expect(controller.getVisibleCount(-5)).toBe(0);
    });
  });

  // --- countLabel ---

  describe('countLabel', () => {
    it('formats correct label string', () => {
      const label = controller.countLabel(3, 10, 8, -0.25);
      expect(label).toBe('Closed surfaces: 3/10 selected (visible 8/10, threshold <= -0.25).');
    });

    it('handles zero values', () => {
      const label = controller.countLabel(0, 0, 0, -0.25);
      expect(label).toBe('Closed surfaces: 0/0 selected (visible 0/0, threshold <= -0.25).');
    });
  });

  // --- updateStatus ---

  describe('updateStatus', () => {
    it('updates status text from selected/visible/candidate counts', () => {
      controller.setToolbarVisible(true);
      controller._candidateCount = 10;
      preview.getMeshRepairSelectedLoopIds.mockReturnValue([1, 2, 3]);
      preview.getMeshRepairTotalLoopCount.mockReturnValue(10);
      preview.getMeshRepairVisibleLoopCount.mockReturnValue(8);

      controller.updateStatus();

      const statusEl = document.getElementById('mesh-repair-status');
      expect(statusEl.textContent).toContain('3/10 selected');
      expect(statusEl.textContent).toContain('visible 8/10');
    });

    it('uses provided selectedIds array when given', () => {
      controller.setToolbarVisible(true);
      controller._candidateCount = 5;
      preview.getMeshRepairTotalLoopCount.mockReturnValue(0);
      preview.getMeshRepairVisibleLoopCount.mockReturnValue(0);

      controller.updateStatus([10, 20]);

      const statusEl = document.getElementById('mesh-repair-status');
      expect(statusEl.textContent).toContain('2/5 selected');
    });

    it('does nothing when toolbar is hidden', () => {
      controller.setToolbarVisible(false);
      controller.setStatus('initial');
      controller.updateStatus();
      const statusEl = document.getElementById('mesh-repair-status');
      expect(statusEl.textContent).toBe('initial');
    });
  });

  // --- activateFromApi ---

  describe('activateFromApi', () => {
    it('fetches candidates and activates the toolbar', async () => {
      const mockData = { loop_count: 12, loops: [] };
      addRoute('/api/mesh-repair/candidates', jsonResponse(mockData));

      await controller.activateFromApi();

      expect(globalThis.fetch).toHaveBeenCalledWith(
        '/api/mesh-repair/candidates',
        expect.objectContaining({ cache: 'no-store' }),
      );
      expect(preview.beginMeshRepairSelection).toHaveBeenCalledWith(mockData);
      expect(controller.candidateCount).toBe(12);
      expect(controller.inFlight).toBe(false);
    });

    it('sets toolbar visible and enables controls', async () => {
      addRoute('/api/mesh-repair/candidates', jsonResponse({ loop_count: 5, loops: [] }));

      await controller.activateFromApi();

      const toolbar = document.getElementById('mesh-repair-toolbar');
      expect(toolbar.style.display).toBe('flex');
      expect(document.getElementById('mesh-repair-threshold').disabled).toBe(false);
      expect(document.getElementById('mesh-repair-apply').disabled).toBe(false);
    });

    it('applies threshold to preview after activation', async () => {
      addRoute('/api/mesh-repair/candidates', jsonResponse({ loop_count: 5, loops: [] }));

      await controller.activateFromApi();

      expect(preview.setMeshRepairThreshold).toHaveBeenCalledWith(MESH_REPAIR_THRESHOLD_DEFAULT);
    });

    it('throws on API error', async () => {
      addRoute('/api/mesh-repair/candidates', jsonResponse({ error: 'No mesh' }, 400));

      await expect(controller.activateFromApi()).rejects.toThrow('No mesh');
    });
  });

  // --- applySelection ---

  describe('applySelection', () => {
    it('sends selected IDs to /api/mesh-repair/confirm', async () => {
      preview.getMeshRepairSelectedLoopIds.mockReturnValue([1, 3, 5]);
      addRoute('/api/mesh-repair/confirm', jsonResponse({ selected_count: 3 }));

      await controller.applySelection();

      expect(globalThis.fetch).toHaveBeenCalledWith(
        '/api/mesh-repair/confirm',
        expect.objectContaining({
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
        }),
      );
      const callArgs = globalThis.fetch.mock.calls.find(c => c[0] === '/api/mesh-repair/confirm');
      const body = JSON.parse(callArgs[1].body);
      expect(body.selected_loop_ids).toEqual([1, 3, 5]);
    });

    it('handles success with selected loops', async () => {
      preview.getMeshRepairSelectedLoopIds.mockReturnValue([1, 2]);
      controller._candidateCount = 5;
      addRoute('/api/mesh-repair/confirm', jsonResponse({ selected_count: 2 }));

      await controller.applySelection();

      expect(preview.setMeshRepairConfirmed).toHaveBeenCalled();
      const statusEl = document.getElementById('mesh-repair-status');
      expect(statusEl.textContent).toContain('Repairing mesh');
      expect(statusEl.classList.contains('success')).toBe(true);
      expect(appendLog).toHaveBeenCalledWith(
        'stdout',
        expect.stringContaining('2 selected loops'),
        { stage: 7 },
      );
    });

    it('handles success with 0 loops (skip)', async () => {
      preview.getMeshRepairSelectedLoopIds.mockReturnValue([]);
      controller._candidateCount = 5;
      addRoute('/api/mesh-repair/confirm', jsonResponse({ selected_count: 0 }));

      await controller.applySelection();

      expect(preview.setMeshRepairConfirmed).toHaveBeenCalled();
      const statusEl = document.getElementById('mesh-repair-status');
      expect(statusEl.textContent).toContain('Skipping repair');
      expect(statusEl.classList.contains('success')).toBe(true);
      expect(appendLog).toHaveBeenCalledWith(
        'stdout',
        expect.stringContaining('0 selected loops'),
        { stage: 7 },
      );
    });

    it('handles error by re-enabling controls', async () => {
      preview.getMeshRepairSelectedLoopIds.mockReturnValue([1]);
      addRoute('/api/mesh-repair/confirm', jsonResponse({ error: 'Server error' }, 500));

      await controller.applySelection();

      expect(controller.inFlight).toBe(false);
      expect(document.getElementById('mesh-repair-apply').disabled).toBe(false);
      const statusEl = document.getElementById('mesh-repair-status');
      expect(statusEl.textContent).toContain('Failed to confirm');
      expect(statusEl.classList.contains('error')).toBe(true);
      expect(appendLog).toHaveBeenCalledWith(
        'stderr',
        expect.stringContaining('confirm failed'),
        { stage: 7 },
      );
    });

    it('does nothing when already in flight', async () => {
      controller._inFlight = true;
      await controller.applySelection();
      expect(globalThis.fetch).not.toHaveBeenCalledWith(
        '/api/mesh-repair/confirm',
        expect.anything(),
      );
    });
  });

  // --- syncFromStatus ---

  describe('syncFromStatus', () => {
    function makeStatusMsg({ running = true, stageStatus = 'interactive', ready = true, selectedLoopCount = 0, candidateCount = 10 } = {}) {
      return {
        running,
        stages: { '7': { status: stageStatus, progress: 0 } },
        mesh_repair: { ready, selected_loop_count: selectedLoopCount, candidate_count: candidateCount },
      };
    }

    it('shows toolbar when stage 7 is interactive and mesh_repair.ready is true', () => {
      controller.syncFromStatus(makeStatusMsg());
      const toolbar = document.getElementById('mesh-repair-toolbar');
      expect(toolbar.style.display).toBe('flex');
    });

    it('hides toolbar when not running', () => {
      controller.setToolbarVisible(true);
      controller.syncFromStatus(makeStatusMsg({ running: false }));
      const toolbar = document.getElementById('mesh-repair-toolbar');
      expect(toolbar.style.display).toBe('none');
    });

    it('hides toolbar when stage 7 is not interactive', () => {
      controller.setToolbarVisible(true);
      controller.syncFromStatus(makeStatusMsg({ stageStatus: 'running' }));
      const toolbar = document.getElementById('mesh-repair-toolbar');
      expect(toolbar.style.display).toBe('none');
    });

    it('hides toolbar when mesh_repair.ready is false', () => {
      controller.setToolbarVisible(true);
      controller.syncFromStatus(makeStatusMsg({ ready: false }));
      const toolbar = document.getElementById('mesh-repair-toolbar');
      expect(toolbar.style.display).toBe('none');
    });

    it('respects inFlight and does not hide toolbar when ready becomes false', () => {
      controller._inFlight = true;
      controller.setToolbarVisible(true);
      controller.setStatus('Confirming...');

      controller.syncFromStatus(makeStatusMsg({ ready: false }));

      const toolbar = document.getElementById('mesh-repair-toolbar');
      // Toolbar remains visible because inFlight prevents hiding
      expect(toolbar.style.display).toBe('flex');
      expect(document.getElementById('mesh-repair-status').textContent).toBe('Confirming...');
    });

    it('respects inFlight and does not modify state when ready is true', () => {
      controller._inFlight = true;
      controller.setToolbarVisible(true);
      controller.setStatus('Confirming...');

      controller.syncFromStatus(makeStatusMsg({ ready: true }));

      // inFlight means early return, so status should not be overwritten
      expect(document.getElementById('mesh-repair-status').textContent).toBe('Confirming...');
    });

    it('updates candidateCount from status message', () => {
      controller.syncFromStatus(makeStatusMsg({ candidateCount: 42 }));
      expect(controller.candidateCount).toBe(42);
    });

    it('updates status text with count info', () => {
      preview.getMeshRepairTotalLoopCount.mockReturnValue(0);
      preview.getMeshRepairVisibleLoopCount.mockReturnValue(0);

      controller.syncFromStatus(makeStatusMsg({ selectedLoopCount: 3, candidateCount: 10 }));

      const statusEl = document.getElementById('mesh-repair-status');
      expect(statusEl.textContent).toContain('3/10 selected');
    });
  });
});
