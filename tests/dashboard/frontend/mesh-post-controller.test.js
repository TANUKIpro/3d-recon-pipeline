import { describe, it, expect, vi, beforeEach } from 'vitest';
import { useFetchMock } from './helpers/fetch-mock.js';
import { MeshPostController } from '../../../scripts/dashboard/static/js/mesh-post-controller.js';
import { DEFAULT_TAUBIN_NU, CLASSICAL_PREVIEW_TITLE } from '../../../scripts/dashboard/static/js/constants.js';

// ── DOM builder (local to this test file) ──────────────────────
function buildMeshPostDOM() {
  document.body.innerHTML = `
    <div id="mesh-post-toolbar" class="hidden">
      <select id="mesh-post-method">
        <option value="laplacian">Laplacian</option>
        <option value="taubin">Taubin</option>
      </select>
      <input type="number" id="mesh-post-iterations" value="8">
      <input type="number" id="mesh-post-lambda" value="0.5">
      <input type="checkbox" id="mesh-post-downsample-enabled" checked>
      <input type="number" id="mesh-post-target-faces" value="120000">
      <input type="number" id="mesh-post-trigger-faces" value="170000">
      <button id="mesh-post-apply">Apply</button>
      <button id="mesh-post-reset">Reset</button>
      <span id="mesh-post-status"></span>
    </div>
  `;
}

// ── Mock dependency factory ────────────────────────────────────
function makeDeps(overrides = {}) {
  return {
    preview: { loadStageResult: vi.fn().mockResolvedValue(undefined) },
    stageCtrl: { activateStage: vi.fn() },
    appendLog: vi.fn(),
    getMeshMethod: vi.fn().mockReturnValue('poisson'),
    isPipelineRunning: vi.fn().mockReturnValue(false),
    setMeshPhaseStatus: vi.fn(),
    applyStatusSnapshot: vi.fn().mockResolvedValue(undefined),
    ...overrides,
  };
}

describe('MeshPostController', () => {
  const fetchMock = useFetchMock();
  let ctrl;
  let deps;

  beforeEach(() => {
    buildMeshPostDOM();
    fetchMock.installFetch();
    fetchMock.clearRoutes();
    deps = makeDeps();
    ctrl = new MeshPostController(deps);
  });

  // ── init ──────────────────────────────────────────────────────

  describe('init', () => {
    it('sets toolbar hidden', () => {
      ctrl.init();
      expect(ctrl._toolbar.classList.contains('hidden')).toBe(true);
    });

    it('sets controls disabled', () => {
      ctrl.init();
      expect(ctrl._methodInput.disabled).toBe(true);
      expect(ctrl._iterationsInput.disabled).toBe(true);
      expect(ctrl._lambdaInput.disabled).toBe(true);
      expect(ctrl._downsampleEnabledInput.disabled).toBe(true);
      expect(ctrl._targetFacesInput.disabled).toBe(true);
      expect(ctrl._triggerFacesInput.disabled).toBe(true);
      expect(ctrl._applyBtn.disabled).toBe(true);
      expect(ctrl._resetBtn.disabled).toBe(true);
    });

    it('wires apply button click handler', async () => {
      ctrl.init();
      const spy = vi.spyOn(ctrl, 'applyPostprocess').mockResolvedValue(undefined);
      // init() disables buttons; re-enable so jsdom fires the click
      ctrl.setEnabled(true);
      ctrl._applyBtn.click();
      expect(spy).toHaveBeenCalledWith({ resetToRaw: false });
    });

    it('wires reset button click handler', async () => {
      ctrl.init();
      const spy = vi.spyOn(ctrl, 'applyPostprocess').mockResolvedValue(undefined);
      ctrl.setEnabled(true);
      ctrl._resetBtn.click();
      expect(spy).toHaveBeenCalledWith({ resetToRaw: true });
    });
  });

  // ── setToolbarVisible ─────────────────────────────────────────

  describe('setToolbarVisible', () => {
    it('shows toolbar when true', () => {
      ctrl.setToolbarVisible(true);
      expect(ctrl._toolbar.classList.contains('hidden')).toBe(false);
    });

    it('hides toolbar when false', () => {
      ctrl.setToolbarVisible(true);
      ctrl.setToolbarVisible(false);
      expect(ctrl._toolbar.classList.contains('hidden')).toBe(true);
    });
  });

  // ── setEnabled ────────────────────────────────────────────────

  describe('setEnabled', () => {
    it('enables all inputs and buttons when true', () => {
      ctrl.setEnabled(false);
      ctrl.setEnabled(true);
      expect(ctrl._methodInput.disabled).toBe(false);
      expect(ctrl._iterationsInput.disabled).toBe(false);
      expect(ctrl._lambdaInput.disabled).toBe(false);
      expect(ctrl._downsampleEnabledInput.disabled).toBe(false);
      expect(ctrl._targetFacesInput.disabled).toBe(false);
      expect(ctrl._triggerFacesInput.disabled).toBe(false);
      expect(ctrl._applyBtn.disabled).toBe(false);
      expect(ctrl._resetBtn.disabled).toBe(false);
    });

    it('disables all inputs and buttons when false', () => {
      ctrl.setEnabled(true);
      ctrl.setEnabled(false);
      expect(ctrl._methodInput.disabled).toBe(true);
      expect(ctrl._iterationsInput.disabled).toBe(true);
      expect(ctrl._lambdaInput.disabled).toBe(true);
      expect(ctrl._downsampleEnabledInput.disabled).toBe(true);
      expect(ctrl._targetFacesInput.disabled).toBe(true);
      expect(ctrl._triggerFacesInput.disabled).toBe(true);
      expect(ctrl._applyBtn.disabled).toBe(true);
      expect(ctrl._resetBtn.disabled).toBe(true);
    });
  });

  // ── setStatus ─────────────────────────────────────────────────

  describe('setStatus', () => {
    it('sets text content', () => {
      ctrl.setStatus('hello');
      expect(ctrl._statusEl.textContent).toBe('hello');
    });

    it('clears previous tone classes', () => {
      ctrl._statusEl.classList.add('error', 'success');
      ctrl.setStatus('plain');
      expect(ctrl._statusEl.classList.contains('error')).toBe(false);
      expect(ctrl._statusEl.classList.contains('success')).toBe(false);
    });

    it('adds error class when tone is error', () => {
      ctrl.setStatus('oops', 'error');
      expect(ctrl._statusEl.classList.contains('error')).toBe(true);
      expect(ctrl._statusEl.classList.contains('success')).toBe(false);
    });

    it('adds success class when tone is success', () => {
      ctrl.setStatus('done', 'success');
      expect(ctrl._statusEl.classList.contains('success')).toBe(true);
      expect(ctrl._statusEl.classList.contains('error')).toBe(false);
    });

    it('coerces null/undefined to empty string', () => {
      ctrl.setStatus(null);
      expect(ctrl._statusEl.textContent).toBe('');
      ctrl.setStatus(undefined);
      expect(ctrl._statusEl.textContent).toBe('');
    });
  });

  // ── canRun ────────────────────────────────────────────────────

  describe('canRun', () => {
    it('returns true when statusMsg is null', () => {
      expect(ctrl.canRun(null)).toBe(true);
    });

    it('returns true when not running', () => {
      expect(ctrl.canRun({ running: false })).toBe(true);
    });

    it('returns true when running and waiting at stage 5', () => {
      const status = {
        running: true,
        next_stage_confirmation: { required: true, from_stage: 5 },
      };
      expect(ctrl.canRun(status)).toBe(true);
    });

    it('returns false when running at other stage', () => {
      const status = {
        running: true,
        next_stage_confirmation: { required: true, from_stage: 3 },
      };
      expect(ctrl.canRun(status)).toBe(false);
    });

    it('returns false when running with no confirmation required', () => {
      const status = {
        running: true,
        next_stage_confirmation: { required: false },
      };
      expect(ctrl.canRun(status)).toBe(false);
    });
  });

  // ── syncFromStatus ────────────────────────────────────────────

  describe('syncFromStatus', () => {
    it('does nothing when statusMsg is null', () => {
      ctrl.syncFromStatus(null);
      // toolbar should remain in its prior state (no error thrown)
    });

    it('shows toolbar when stage 5 is done and object_name present', () => {
      const status = {
        object_name: 'test-obj',
        running: false,
        stages: { '5': { status: 'complete', progress: 100 } },
      };
      ctrl.syncFromStatus(status);
      expect(ctrl._toolbar.classList.contains('hidden')).toBe(false);
    });

    it('hides toolbar when stage 5 not done', () => {
      ctrl.setToolbarVisible(true);
      const status = {
        object_name: 'test-obj',
        running: false,
        stages: { '5': { status: 'running', progress: 50 } },
      };
      ctrl.syncFromStatus(status);
      expect(ctrl._toolbar.classList.contains('hidden')).toBe(true);
    });

    it('hides toolbar when object_name missing', () => {
      ctrl.setToolbarVisible(true);
      const status = {
        running: false,
        stages: { '5': { status: 'complete', progress: 100 } },
      };
      ctrl.syncFromStatus(status);
      expect(ctrl._toolbar.classList.contains('hidden')).toBe(true);
    });

    it('disables controls when pipeline running at non-stage-5', () => {
      ctrl.setEnabled(true);
      const status = {
        object_name: 'test-obj',
        running: true,
        next_stage_confirmation: { required: true, from_stage: 3 },
        stages: { '5': { status: 'complete', progress: 100 } },
      };
      ctrl.syncFromStatus(status);
      expect(ctrl._applyBtn.disabled).toBe(true);
    });

    it('enables controls when stage 5 done and not running', () => {
      const status = {
        object_name: 'test-obj',
        running: false,
        stages: { '5': { status: 'complete', progress: 100 } },
      };
      ctrl.syncFromStatus(status);
      expect(ctrl._applyBtn.disabled).toBe(false);
    });

    it('keeps controls disabled when inFlight is true', () => {
      ctrl._inFlight = true;
      const status = {
        object_name: 'test-obj',
        running: false,
        stages: { '5': { status: 'complete', progress: 100 } },
      };
      ctrl.syncFromStatus(status);
      expect(ctrl._applyBtn.disabled).toBe(true);
    });

    it('shows disabled message when not allowed', () => {
      const status = {
        object_name: 'test-obj',
        running: true,
        next_stage_confirmation: { required: false },
        stages: { '5': { status: 'complete', progress: 100 } },
      };
      ctrl.syncFromStatus(status);
      expect(ctrl._statusEl.textContent).toBe(
        'Mesh post-process is disabled while pipeline is running.',
      );
    });

    it('clears disabled message when allowed again', () => {
      // First set the disabled message
      ctrl._statusEl.textContent = 'Mesh post-process is disabled while pipeline is running.';
      const status = {
        object_name: 'test-obj',
        running: false,
        stages: { '5': { status: 'complete', progress: 100 } },
      };
      ctrl.syncFromStatus(status);
      expect(ctrl._statusEl.textContent).toBe('');
    });
  });

  // ── applyPostprocess ──────────────────────────────────────────

  describe('applyPostprocess', () => {
    const successData = {
      method: 'laplacian',
      iterations: 8,
      lamb: 0.5,
      source: 'raw',
      vertices: 5000,
      faces: 10000,
      downsample_enabled: true,
      downsample_target_faces: 120000,
      downsample_trigger_faces: 170000,
      downsample_applied: false,
      texture_invalidated: false,
    };

    function setupPostRoute(data = successData, status = 200) {
      fetchMock.addRoute('/api/mesh/postprocess', fetchMock.jsonResponse(data, status));
    }

    function setupStatusRoute(statusData = { running: false }) {
      fetchMock.addRoute('/api/pipeline/status', fetchMock.jsonResponse(statusData));
    }

    it('sends correct POST body', async () => {
      setupPostRoute();
      setupStatusRoute();
      ctrl._methodInput.value = 'taubin';
      ctrl._iterationsInput.value = '12';
      ctrl._lambdaInput.value = '0.8';
      ctrl._downsampleEnabledInput.checked = true;
      ctrl._targetFacesInput.value = '100000';
      ctrl._triggerFacesInput.value = '200000';

      await ctrl.applyPostprocess({ resetToRaw: false });

      const call = globalThis.fetch.mock.calls.find(c => c[0] === '/api/mesh/postprocess');
      expect(call).toBeDefined();
      const body = JSON.parse(call[1].body);
      expect(body).toEqual({
        method: 'taubin',
        iterations: 12,
        lamb: 0.8,
        taubin_nu: DEFAULT_TAUBIN_NU,
        downsample_enabled: true,
        downsample_target_faces: 100000,
        downsample_trigger_faces: 200000,
        source: 'raw',
        invalidate_texture: true,
      });
    });

    it('clamps iterations to 0-100 range', async () => {
      setupPostRoute();
      setupStatusRoute();
      ctrl._iterationsInput.value = '999';

      await ctrl.applyPostprocess({ resetToRaw: false });

      const call = globalThis.fetch.mock.calls.find(c => c[0] === '/api/mesh/postprocess');
      const body = JSON.parse(call[1].body);
      expect(body.iterations).toBe(100);
    });

    it('clamps lambda to 0.01-1.5 range', async () => {
      setupPostRoute();
      setupStatusRoute();
      ctrl._lambdaInput.value = '0.001';

      await ctrl.applyPostprocess({ resetToRaw: false });

      const call = globalThis.fetch.mock.calls.find(c => c[0] === '/api/mesh/postprocess');
      const body = JSON.parse(call[1].body);
      expect(body.lamb).toBe(0.01);
    });

    it('clamps lambda upper bound', async () => {
      setupPostRoute();
      setupStatusRoute();
      ctrl._lambdaInput.value = '99';

      await ctrl.applyPostprocess({ resetToRaw: false });

      const call = globalThis.fetch.mock.calls.find(c => c[0] === '/api/mesh/postprocess');
      const body = JSON.parse(call[1].body);
      expect(body.lamb).toBe(1.5);
    });

    it('handles success: updates status, loads result, activates stage 5', async () => {
      setupPostRoute();
      setupStatusRoute();

      await ctrl.applyPostprocess({ resetToRaw: false });

      expect(deps.preview.loadStageResult).toHaveBeenCalledWith(5, { preferPreview: true });
      expect(deps.stageCtrl.activateStage).toHaveBeenCalledWith(5);
      expect(ctrl._statusEl.classList.contains('success')).toBe(true);
      expect(ctrl._statusEl.textContent).toContain('Smoothing complete');
    });

    it('calls setMeshPhaseStatus with preview title for poisson method', async () => {
      setupPostRoute();
      setupStatusRoute();
      deps.getMeshMethod.mockReturnValue('poisson');

      await ctrl.applyPostprocess({ resetToRaw: false });

      expect(deps.setMeshPhaseStatus).toHaveBeenCalledWith(
        `Showing: ${CLASSICAL_PREVIEW_TITLE}`,
        'ready',
      );
    });

    it('does not call setMeshPhaseStatus for non-poisson method', async () => {
      setupPostRoute();
      setupStatusRoute();
      deps.getMeshMethod.mockReturnValue('diffcd');

      await ctrl.applyPostprocess({ resetToRaw: false });

      expect(deps.setMeshPhaseStatus).not.toHaveBeenCalled();
    });

    it('handles error: shows error status', async () => {
      fetchMock.addRoute('/api/mesh/postprocess', fetchMock.jsonResponse(
        { error: 'Something broke' }, 500,
      ));

      await ctrl.applyPostprocess({ resetToRaw: false });

      expect(ctrl._statusEl.textContent).toContain('Failed');
      expect(ctrl._statusEl.textContent).toContain('Something broke');
      expect(ctrl._statusEl.classList.contains('error')).toBe(true);
    });

    it('calls onPostprocessComplete callback in finally', async () => {
      setupPostRoute();
      setupStatusRoute();
      const callback = vi.fn();
      ctrl.onPostprocessComplete = callback;

      await ctrl.applyPostprocess({ resetToRaw: false });

      expect(callback).toHaveBeenCalledTimes(1);
    });

    it('calls onPostprocessComplete even on error', async () => {
      fetchMock.addRoute('/api/mesh/postprocess', fetchMock.jsonResponse(
        { error: 'fail' }, 500,
      ));
      const callback = vi.fn();
      ctrl.onPostprocessComplete = callback;

      await ctrl.applyPostprocess({ resetToRaw: false });

      expect(callback).toHaveBeenCalledTimes(1);
    });

    it('resets inFlight to false after completion', async () => {
      setupPostRoute();
      setupStatusRoute();

      await ctrl.applyPostprocess({ resetToRaw: false });

      expect(ctrl.inFlight).toBe(false);
    });

    it('appends stdout log on success', async () => {
      setupPostRoute();
      setupStatusRoute();

      await ctrl.applyPostprocess({ resetToRaw: false });

      expect(deps.appendLog).toHaveBeenCalledWith(
        'stdout',
        expect.stringContaining('Mesh post-process:'),
        { stage: 5 },
      );
    });

    it('appends stderr log on error', async () => {
      fetchMock.addRoute('/api/mesh/postprocess', fetchMock.jsonResponse(
        { error: 'badness' }, 500,
      ));

      await ctrl.applyPostprocess({ resetToRaw: false });

      expect(deps.appendLog).toHaveBeenCalledWith(
        'stderr',
        expect.stringContaining('Mesh post-process failed'),
        { stage: 5 },
      );
    });
  });

  // ── applyPostprocess resetToRaw ───────────────────────────────

  describe('applyPostprocess resetToRaw', () => {
    it('sends iterations=0 for reset', async () => {
      fetchMock.addRoute('/api/mesh/postprocess', fetchMock.jsonResponse({
        method: 'laplacian', iterations: 0, lamb: 0.5, source: 'raw',
        vertices: 3000, faces: 6000,
        downsample_enabled: true, downsample_target_faces: 120000,
        downsample_trigger_faces: 170000, downsample_applied: false,
        texture_invalidated: false,
      }));
      fetchMock.addRoute('/api/pipeline/status', fetchMock.jsonResponse({ running: false }));

      ctrl._methodInput.value = 'taubin';
      ctrl._iterationsInput.value = '20';

      await ctrl.applyPostprocess({ resetToRaw: true });

      const call = globalThis.fetch.mock.calls.find(c => c[0] === '/api/mesh/postprocess');
      const body = JSON.parse(call[1].body);
      expect(body.iterations).toBe(0);
      expect(body.method).toBe('laplacian');
    });

    it('shows Reset complete message on success', async () => {
      fetchMock.addRoute('/api/mesh/postprocess', fetchMock.jsonResponse({
        method: 'laplacian', iterations: 0, lamb: 0.5, source: 'raw',
        vertices: 3000, faces: 6000,
        downsample_enabled: true, downsample_target_faces: 120000,
        downsample_trigger_faces: 170000, downsample_applied: false,
        texture_invalidated: false,
      }));
      fetchMock.addRoute('/api/pipeline/status', fetchMock.jsonResponse({ running: false }));

      await ctrl.applyPostprocess({ resetToRaw: true });

      expect(ctrl._statusEl.textContent).toContain('Reset complete');
    });

    it('shows Resetting to raw mesh status while in flight', async () => {
      let resolvePost;
      fetchMock.addRoute('/api/mesh/postprocess', () => new Promise(r => { resolvePost = r; }));
      fetchMock.addRoute('/api/pipeline/status', fetchMock.jsonResponse({ running: false }));

      const promise = ctrl.applyPostprocess({ resetToRaw: true });

      expect(ctrl._statusEl.textContent).toBe('Resetting to raw mesh...');

      // Resolve to avoid hanging
      resolvePost(fetchMock.jsonResponse({
        method: 'laplacian', iterations: 0, lamb: 0.5, source: 'raw',
        vertices: 1, faces: 1,
        downsample_enabled: true, downsample_target_faces: 120000,
        downsample_trigger_faces: 170000, downsample_applied: false,
        texture_invalidated: false,
      }));
      await promise;
    });
  });

  // ── applyPostprocess downsample fields ────────────────────────

  describe('applyPostprocess downsample fields', () => {
    it('reads and sends downsample config', async () => {
      fetchMock.addRoute('/api/mesh/postprocess', fetchMock.jsonResponse({
        method: 'laplacian', iterations: 8, lamb: 0.5, source: 'raw',
        vertices: 5000, faces: 10000,
        downsample_enabled: false, downsample_target_faces: 50000,
        downsample_trigger_faces: 80000, downsample_applied: false,
        texture_invalidated: false,
      }));
      fetchMock.addRoute('/api/pipeline/status', fetchMock.jsonResponse({ running: false }));

      ctrl._downsampleEnabledInput.checked = false;
      ctrl._targetFacesInput.value = '50000';
      ctrl._triggerFacesInput.value = '80000';

      await ctrl.applyPostprocess({ resetToRaw: false });

      const call = globalThis.fetch.mock.calls.find(c => c[0] === '/api/mesh/postprocess');
      const body = JSON.parse(call[1].body);
      expect(body.downsample_enabled).toBe(false);
      expect(body.downsample_target_faces).toBe(50000);
      expect(body.downsample_trigger_faces).toBe(80000);
    });

    it('clamps target faces to minimum 1000', async () => {
      fetchMock.addRoute('/api/mesh/postprocess', fetchMock.jsonResponse({
        method: 'laplacian', iterations: 8, lamb: 0.5, source: 'raw',
        vertices: 5000, faces: 10000,
        downsample_enabled: true, downsample_target_faces: 1000,
        downsample_trigger_faces: 1000, downsample_applied: false,
        texture_invalidated: false,
      }));
      fetchMock.addRoute('/api/pipeline/status', fetchMock.jsonResponse({ running: false }));

      ctrl._targetFacesInput.value = '10';
      ctrl._triggerFacesInput.value = '5';

      await ctrl.applyPostprocess({ resetToRaw: false });

      const call = globalThis.fetch.mock.calls.find(c => c[0] === '/api/mesh/postprocess');
      const body = JSON.parse(call[1].body);
      expect(body.downsample_target_faces).toBe(1000);
      // trigger is clamped to at least target
      expect(body.downsample_trigger_faces).toBe(1000);
    });

    it('clamps trigger faces to at least target faces', async () => {
      fetchMock.addRoute('/api/mesh/postprocess', fetchMock.jsonResponse({
        method: 'laplacian', iterations: 8, lamb: 0.5, source: 'raw',
        vertices: 5000, faces: 10000,
        downsample_enabled: true, downsample_target_faces: 200000,
        downsample_trigger_faces: 200000, downsample_applied: false,
        texture_invalidated: false,
      }));
      fetchMock.addRoute('/api/pipeline/status', fetchMock.jsonResponse({ running: false }));

      ctrl._targetFacesInput.value = '200000';
      ctrl._triggerFacesInput.value = '100000';

      await ctrl.applyPostprocess({ resetToRaw: false });

      const call = globalThis.fetch.mock.calls.find(c => c[0] === '/api/mesh/postprocess');
      const body = JSON.parse(call[1].body);
      expect(body.downsample_trigger_faces).toBe(200000);
    });

    it('updates UI fields from API response', async () => {
      fetchMock.addRoute('/api/mesh/postprocess', fetchMock.jsonResponse({
        method: 'laplacian', iterations: 8, lamb: 0.5, source: 'raw',
        vertices: 5000, faces: 10000,
        downsample_enabled: false, downsample_target_faces: 90000,
        downsample_trigger_faces: 150000, downsample_applied: false,
        texture_invalidated: false,
      }));
      fetchMock.addRoute('/api/pipeline/status', fetchMock.jsonResponse({ running: false }));

      await ctrl.applyPostprocess({ resetToRaw: false });

      expect(ctrl._downsampleEnabledInput.checked).toBe(false);
      expect(ctrl._targetFacesInput.value).toBe('90000');
      expect(ctrl._triggerFacesInput.value).toBe('150000');
    });

    it('shows downsample applied message', async () => {
      fetchMock.addRoute('/api/mesh/postprocess', fetchMock.jsonResponse({
        method: 'laplacian', iterations: 8, lamb: 0.5, source: 'raw',
        vertices: 5000, faces: 10000,
        downsample_enabled: true, downsample_target_faces: 120000,
        downsample_trigger_faces: 170000, downsample_applied: true,
        texture_invalidated: false,
      }));
      fetchMock.addRoute('/api/pipeline/status', fetchMock.jsonResponse({ running: false }));

      await ctrl.applyPostprocess({ resetToRaw: false });

      expect(ctrl._statusEl.textContent).toContain('Downsample applied.');
    });

    it('shows downsample disabled message', async () => {
      fetchMock.addRoute('/api/mesh/postprocess', fetchMock.jsonResponse({
        method: 'laplacian', iterations: 8, lamb: 0.5, source: 'raw',
        vertices: 5000, faces: 10000,
        downsample_enabled: false, downsample_target_faces: 120000,
        downsample_trigger_faces: 170000, downsample_applied: false,
        texture_invalidated: false,
      }));
      fetchMock.addRoute('/api/pipeline/status', fetchMock.jsonResponse({ running: false }));

      await ctrl.applyPostprocess({ resetToRaw: false });

      expect(ctrl._statusEl.textContent).toContain('Downsample disabled.');
    });

    it('shows texture invalidated message', async () => {
      fetchMock.addRoute('/api/mesh/postprocess', fetchMock.jsonResponse({
        method: 'laplacian', iterations: 8, lamb: 0.5, source: 'raw',
        vertices: 5000, faces: 10000,
        downsample_enabled: true, downsample_target_faces: 120000,
        downsample_trigger_faces: 170000, downsample_applied: false,
        texture_invalidated: true,
      }));
      fetchMock.addRoute('/api/pipeline/status', fetchMock.jsonResponse({ running: false }));

      await ctrl.applyPostprocess({ resetToRaw: false });

      expect(ctrl._statusEl.textContent).toContain('Downstream Stage 6/7/8 artifacts were cleared.');
      expect(deps.appendLog).toHaveBeenCalledWith(
        'stdout',
        expect.stringContaining('Wrap/repair/texture outputs removed'),
        { stage: 6 },
      );
    });
  });

  // ── applyPostprocess status refresh ───────────────────────────

  describe('applyPostprocess status refresh', () => {
    it('fetches /api/pipeline/status and calls applyStatusSnapshot when not running', async () => {
      fetchMock.addRoute('/api/mesh/postprocess', fetchMock.jsonResponse({
        method: 'laplacian', iterations: 8, lamb: 0.5, source: 'raw',
        vertices: 5000, faces: 10000,
        downsample_enabled: true, downsample_target_faces: 120000,
        downsample_trigger_faces: 170000, downsample_applied: false,
        texture_invalidated: false,
      }));
      const statusData = { running: false, stages: {} };
      fetchMock.addRoute('/api/pipeline/status', fetchMock.jsonResponse(statusData));
      deps.isPipelineRunning.mockReturnValue(false);

      await ctrl.applyPostprocess({ resetToRaw: false });

      const statusCall = globalThis.fetch.mock.calls.find(c =>
        typeof c[0] === 'string' && c[0].startsWith('/api/pipeline/status'),
      );
      expect(statusCall).toBeDefined();
      expect(deps.applyStatusSnapshot).toHaveBeenCalledWith(statusData, { forceHydrate: true });
      // activateStage called twice: once after loadStageResult, once after status refresh
      expect(deps.stageCtrl.activateStage).toHaveBeenCalledTimes(2);
    });

    it('does not fetch status when pipeline is running', async () => {
      fetchMock.addRoute('/api/mesh/postprocess', fetchMock.jsonResponse({
        method: 'laplacian', iterations: 8, lamb: 0.5, source: 'raw',
        vertices: 5000, faces: 10000,
        downsample_enabled: true, downsample_target_faces: 120000,
        downsample_trigger_faces: 170000, downsample_applied: false,
        texture_invalidated: false,
      }));
      deps.isPipelineRunning.mockReturnValue(true);

      await ctrl.applyPostprocess({ resetToRaw: false });

      const statusCall = globalThis.fetch.mock.calls.find(c =>
        typeof c[0] === 'string' && c[0].startsWith('/api/pipeline/status'),
      );
      expect(statusCall).toBeUndefined();
      expect(deps.applyStatusSnapshot).not.toHaveBeenCalled();
    });
  });

  // ── inFlight guard ────────────────────────────────────────────

  describe('inFlight guard', () => {
    it('prevents concurrent calls', async () => {
      let resolvePost;
      fetchMock.addRoute('/api/mesh/postprocess', () => new Promise(r => { resolvePost = r; }));
      fetchMock.addRoute('/api/pipeline/status', fetchMock.jsonResponse({ running: false }));

      // Start first call — it will hang on the fetch
      const first = ctrl.applyPostprocess({ resetToRaw: false });
      expect(ctrl.inFlight).toBe(true);

      // Second call should return immediately without a new fetch
      const callCountBefore = globalThis.fetch.mock.calls.length;
      await ctrl.applyPostprocess({ resetToRaw: false });
      expect(globalThis.fetch.mock.calls.length).toBe(callCountBefore);

      // Resolve the first call to clean up
      resolvePost(fetchMock.jsonResponse({
        method: 'laplacian', iterations: 8, lamb: 0.5, source: 'raw',
        vertices: 1, faces: 1,
        downsample_enabled: true, downsample_target_faces: 120000,
        downsample_trigger_faces: 170000, downsample_applied: false,
        texture_invalidated: false,
      }));
      await first;
      expect(ctrl.inFlight).toBe(false);
    });

    it('sets inFlight true during execution', async () => {
      let capturedInFlight;
      fetchMock.addRoute('/api/mesh/postprocess', () => {
        capturedInFlight = ctrl.inFlight;
        return fetchMock.jsonResponse({
          method: 'laplacian', iterations: 8, lamb: 0.5, source: 'raw',
          vertices: 1, faces: 1,
          downsample_enabled: true, downsample_target_faces: 120000,
          downsample_trigger_faces: 170000, downsample_applied: false,
          texture_invalidated: false,
        });
      });
      fetchMock.addRoute('/api/pipeline/status', fetchMock.jsonResponse({ running: false }));

      await ctrl.applyPostprocess({ resetToRaw: false });

      expect(capturedInFlight).toBe(true);
      expect(ctrl.inFlight).toBe(false);
    });
  });
});
