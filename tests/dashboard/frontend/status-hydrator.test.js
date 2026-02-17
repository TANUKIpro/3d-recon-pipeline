import { describe, it, expect, vi, beforeEach } from 'vitest';
import { StatusHydrator } from '../../../scripts/dashboard/static/js/status-hydrator.js';

// ── Helpers ──────────────────────────────────────────────────

function mkStage(status = 'pending', progress = 0) {
  return { status, progress };
}

function mkStatus(overrides = {}) {
  return {
    object_name: 'test-obj',
    running: false,
    current_stage: null,
    frame_count: 30,
    next_stage_confirmation: {},
    overall_progress: 50,
    stages: {},
    ...overrides,
  };
}

/** Build a status with all 8 stages complete. */
function mkAllDone(overrides = {}) {
  const stages = {};
  for (let i = 1; i <= 8; i++) stages[String(i)] = mkStage('complete', 100);
  return mkStatus({ stages, ...overrides });
}

/** Create the mock deps bag used by StatusHydrator. */
function createDeps(overrides = {}) {
  return {
    preview: {
      reset: vi.fn(),
      loadGallery: vi.fn().mockResolvedValue(undefined),
      loadPi3xResults: vi.fn().mockResolvedValue(undefined),
      loadStageResult: vi.fn().mockResolvedValue(undefined),
      showCropBbox: vi.fn().mockResolvedValue(undefined),
    },
    cameraOverlay: { remove: vi.fn() },
    sam2: { deactivate: vi.fn() },
    sam2Verify: { hide: vi.fn() },
    config: {
      setObjectName: vi.fn(),
      setRunning: vi.fn(),
      setActiveStage: vi.fn(),
      getCropScale: vi.fn().mockReturnValue(1.0),
    },
    stageCtrl: {
      activateStage: vi.fn(),
      setStageState: vi.fn(),
    },
    checkpoints: { applyStatusSnapshot: vi.fn() },
    pipelineUI: {
      setMeshMethodEnabled: vi.fn(),
      updateAll: vi.fn(),
      getOverallProgress: vi.fn().mockReturnValue(50),
    },
    taskConfirm: { syncFromStatus: vi.fn() },
    meshPost: { syncFromStatus: vi.fn() },
    meshRepair: {
      syncFromStatus: vi.fn(),
      resetState: vi.fn(),
      setToolbarVisible: vi.fn(),
      setStatus: vi.fn(),
      activateFromApi: vi.fn().mockResolvedValue(undefined),
    },
    getMeshMethod: vi.fn().mockReturnValue('poisson'),
    applyMeshMethod: vi.fn(),
    setMeshPhaseStatus: vi.fn(),
    setStatus: vi.fn(),
    setOverallProgress: vi.fn(),
    ...overrides,
  };
}

// ── constructor ──────────────────────────────────────────────

describe('StatusHydrator', () => {
  let deps;
  let hydrator;

  beforeEach(() => {
    deps = createDeps();
    hydrator = new StatusHydrator(deps);
  });

  describe('constructor', () => {
    it('initializes with empty hydrated key and null snapshot', () => {
      expect(hydrator.latestSnapshot).toBe(null);
      // The internal key is empty — calling hydrateOutputs with any valid
      // status should proceed (not skip as "same key").
    });
  });

  // ── latestSnapshot getter ────────────────────────────────────

  describe('latestSnapshot', () => {
    it('returns null initially', () => {
      expect(hydrator.latestSnapshot).toBe(null);
    });

    it('returns the snapshot after applySnapshot', async () => {
      const status = mkStatus();
      await hydrator.applySnapshot(status);
      expect(hydrator.latestSnapshot).toBe(status);
    });
  });

  // ── reset ────────────────────────────────────────────────────

  describe('reset', () => {
    it('clears both key and snapshot', async () => {
      const status = mkStatus({ stages: { '1': mkStage('complete', 100) } });
      await hydrator.applySnapshot(status);
      expect(hydrator.latestSnapshot).not.toBe(null);

      hydrator.reset();
      expect(hydrator.latestSnapshot).toBe(null);
    });

    it('allows re-hydration after reset', async () => {
      const status = mkStatus({ stages: { '1': mkStage('complete', 100) } });

      // Set up DOM for stage 1 empty check
      document.body.innerHTML = '<div id="stage-panel-1"><div class="stage-panel-empty"></div></div>';

      await hydrator.hydrateOutputs(status);
      expect(deps.preview.loadGallery).toHaveBeenCalledTimes(1);

      // Same status again — should skip (same key)
      await hydrator.hydrateOutputs(status);
      expect(deps.preview.loadGallery).toHaveBeenCalledTimes(1);

      // After reset, the same status should trigger hydration again
      hydrator.reset();
      await hydrator.hydrateOutputs(status);
      expect(deps.preview.loadGallery).toHaveBeenCalledTimes(2);
    });
  });

  // ── resetHydratedKey ─────────────────────────────────────────

  describe('resetHydratedKey', () => {
    it('clears only key, keeps snapshot', async () => {
      const status = mkStatus();
      await hydrator.applySnapshot(status);
      expect(hydrator.latestSnapshot).toBe(status);

      hydrator.resetHydratedKey();
      // Snapshot preserved
      expect(hydrator.latestSnapshot).toBe(status);
    });

    it('allows re-hydration without clearing snapshot', async () => {
      const status = mkStatus({ stages: { '1': mkStage('complete', 100) } });
      document.body.innerHTML = '<div id="stage-panel-1"><div class="stage-panel-empty"></div></div>';

      await hydrator.hydrateOutputs(status);
      expect(deps.preview.loadGallery).toHaveBeenCalledTimes(1);

      // Same key — skipped
      await hydrator.hydrateOutputs(status);
      expect(deps.preview.loadGallery).toHaveBeenCalledTimes(1);

      hydrator.resetHydratedKey();
      await hydrator.hydrateOutputs(status);
      expect(deps.preview.loadGallery).toHaveBeenCalledTimes(2);
    });
  });

  // ── hydrateOutputs ───────────────────────────────────────────

  describe('hydrateOutputs', () => {
    it('skips if no object_name', async () => {
      const status = mkStatus({ object_name: '' });
      await hydrator.hydrateOutputs(status);
      expect(deps.preview.reset).not.toHaveBeenCalled();
    });

    it('skips if object_name is undefined', async () => {
      await hydrator.hydrateOutputs({});
      expect(deps.preview.reset).not.toHaveBeenCalled();
    });

    it('skips if same key and not forced', async () => {
      const status = mkStatus({ stages: { '1': mkStage('complete', 100) } });
      document.body.innerHTML = '<div id="stage-panel-1"><div class="stage-panel-empty"></div></div>';

      await hydrator.hydrateOutputs(status);
      expect(deps.preview.reset).toHaveBeenCalledTimes(1);

      await hydrator.hydrateOutputs(status);
      expect(deps.preview.reset).toHaveBeenCalledTimes(1);
    });

    it('resets UI elements (camera, sam2, preview)', async () => {
      const status = mkStatus();
      await hydrator.hydrateOutputs(status);

      expect(deps.cameraOverlay.remove).toHaveBeenCalledTimes(1);
      expect(deps.sam2.deactivate).toHaveBeenCalledTimes(1);
      expect(deps.sam2Verify.hide).toHaveBeenCalledTimes(1);
      expect(deps.preview.reset).toHaveBeenCalledTimes(1);
    });

    it('loads gallery for stage 1 done and hides empty panel', async () => {
      const status = mkStatus({
        stages: { '1': mkStage('complete', 100) },
        frame_count: 42,
      });
      document.body.innerHTML = '<div id="stage-panel-1"><div class="stage-panel-empty"></div></div>';

      await hydrator.hydrateOutputs(status);

      expect(deps.preview.loadGallery).toHaveBeenCalledWith(42);
      const empty = document.querySelector('#stage-panel-1 .stage-panel-empty');
      expect(empty.classList.contains('hidden')).toBe(true);
    });

    it('does not load gallery when stage 1 is not done', async () => {
      const status = mkStatus({
        stages: { '1': mkStage('running', 50) },
      });
      await hydrator.hydrateOutputs(status);
      expect(deps.preview.loadGallery).not.toHaveBeenCalled();
    });

    it('loads Pi3X results with object.ply when stage 3 done', async () => {
      const status = mkStatus({
        stages: {
          '2': mkStage('complete', 100),
          '3': mkStage('complete', 100),
        },
      });
      await hydrator.hydrateOutputs(status);
      expect(deps.preview.loadPi3xResults).toHaveBeenCalledWith(deps.cameraOverlay, 'object.ply');
    });

    it('loads Pi3X results with object_full.ply when stage 2 available but 3 not done', async () => {
      const status = mkStatus({
        stages: {
          '2': mkStage('complete', 100),
          '3': mkStage('running', 50),
        },
      });
      await hydrator.hydrateOutputs(status);
      expect(deps.preview.loadPi3xResults).toHaveBeenCalledWith(deps.cameraOverlay, 'object_full.ply');
    });

    it('does not load Pi3X when stage 2 is not available', async () => {
      const status = mkStatus({
        stages: { '2': mkStage('pending', 0) },
      });
      await hydrator.hydrateOutputs(status);
      expect(deps.preview.loadPi3xResults).not.toHaveBeenCalled();
    });

    it('loads stage results for stages 4-8 when done', async () => {
      const status = mkAllDone();
      deps.getMeshMethod.mockReturnValue('poisson');
      await hydrator.hydrateOutputs(status);

      expect(deps.preview.loadStageResult).toHaveBeenCalledWith(4);
      expect(deps.preview.loadStageResult).toHaveBeenCalledWith(5, { preferPreview: true });
      expect(deps.preview.loadStageResult).toHaveBeenCalledWith(6);
      expect(deps.preview.loadStageResult).toHaveBeenCalledWith(7);
      expect(deps.preview.loadStageResult).toHaveBeenCalledWith(8);
    });

    it('shows crop bbox when stage 5 done but not stage 6', async () => {
      const status = mkStatus({
        stages: {
          '5': mkStage('complete', 100),
        },
      });
      deps.getMeshMethod.mockReturnValue('poisson');
      await hydrator.hydrateOutputs(status);

      expect(deps.preview.showCropBbox).toHaveBeenCalledWith(1.0, { preferPreview: true });
    });

    it('does not show crop bbox when both stage 5 and 6 are done', async () => {
      const status = mkStatus({
        stages: {
          '5': mkStage('complete', 100),
          '6': mkStage('complete', 100),
        },
      });
      await hydrator.hydrateOutputs(status);
      expect(deps.preview.showCropBbox).not.toHaveBeenCalled();
    });

    it('activates preferred stage by default', async () => {
      const status = mkStatus({ current_stage: 3 });
      await hydrator.hydrateOutputs(status);
      expect(deps.stageCtrl.activateStage).toHaveBeenCalledWith(3);
    });

    it('sets mesh phase status when stage 5 done with poisson', async () => {
      const status = mkStatus({
        stages: { '5': mkStage('complete', 100) },
        mesh_method: 'poisson',
      });
      deps.getMeshMethod.mockReturnValue('poisson');
      await hydrator.hydrateOutputs(status);

      expect(deps.setMeshPhaseStatus).toHaveBeenCalledWith(
        'Showing: Classical Mesh result',
        'ready',
      );
    });

    it('clears mesh phase status when stage 5 done with diffcd', async () => {
      const status = mkStatus({
        stages: { '5': mkStage('complete', 100) },
        mesh_method: 'diffcd',
      });
      deps.getMeshMethod.mockReturnValue('diffcd');
      await hydrator.hydrateOutputs(status);

      expect(deps.setMeshPhaseStatus).toHaveBeenCalledWith('', '');
    });

    it('sets mesh phase live status when stage 5 is running with poisson', async () => {
      const status = mkStatus({
        running: true,
        current_stage: 5,
        stages: { '5': mkStage('running', 50) },
        mesh_method: 'poisson',
      });
      deps.getMeshMethod.mockReturnValue('poisson');
      await hydrator.hydrateOutputs(status);

      expect(deps.setMeshPhaseStatus).toHaveBeenCalledWith('Running: Classical Mesh', 'live');
    });

    it('loads stage 5 with preferPreview false for diffcd', async () => {
      const status = mkStatus({
        stages: { '5': mkStage('complete', 100) },
        mesh_method: 'diffcd',
      });
      deps.getMeshMethod.mockReturnValue('diffcd');
      await hydrator.hydrateOutputs(status);

      expect(deps.preview.loadStageResult).toHaveBeenCalledWith(5, { preferPreview: false });
    });
  });

  // ── hydrateOutputs force ─────────────────────────────────────

  describe('hydrateOutputs force', () => {
    it('re-hydrates even with same key', async () => {
      const status = mkStatus();
      await hydrator.hydrateOutputs(status);
      expect(deps.preview.reset).toHaveBeenCalledTimes(1);

      await hydrator.hydrateOutputs(status, { force: true });
      expect(deps.preview.reset).toHaveBeenCalledTimes(2);
    });
  });

  // ── hydrateOutputs activate=false ────────────────────────────

  describe('hydrateOutputs activate=false', () => {
    it('does not call activateStage', async () => {
      const status = mkStatus({ current_stage: 3 });
      await hydrator.hydrateOutputs(status, { activate: false });
      expect(deps.stageCtrl.activateStage).not.toHaveBeenCalled();
    });
  });

  // ── applySnapshot running ────────────────────────────────────

  describe('applySnapshot running', () => {
    it('sets running state', async () => {
      const status = mkStatus({ running: true, current_stage: 2 });
      await hydrator.applySnapshot(status);

      expect(deps.config.setRunning).toHaveBeenCalledWith(true);
      expect(deps.setStatus).toHaveBeenCalledWith('running', 'Running');
    });

    it('does not set idle config when running', async () => {
      const status = mkStatus({ running: true, current_stage: 2 });
      await hydrator.applySnapshot(status);

      expect(deps.config.setRunning).not.toHaveBeenCalledWith(false);
      expect(deps.config.setActiveStage).not.toHaveBeenCalled();
    });

    it('activates preferred stage for running status without confirmation', async () => {
      const status = mkStatus({ running: true, current_stage: 4 });
      await hydrator.applySnapshot(status);

      expect(deps.stageCtrl.activateStage).toHaveBeenCalledWith(4);
    });

    it('hydrates outputs for waiting confirmation', async () => {
      const status = mkStatus({
        running: true,
        current_stage: 3,
        next_stage_confirmation: { required: true, from_stage: 2, to_stage: 3 },
        stages: { '2': mkStage('complete', 100) },
      });

      await hydrator.applySnapshot(status);

      // Should hydrate with activate: false and then explicitly activate from_stage
      expect(deps.stageCtrl.activateStage).toHaveBeenCalledWith(2);
    });

    it('activates preferred stage when waiting from_stage is invalid', async () => {
      const status = mkStatus({
        running: true,
        current_stage: 3,
        next_stage_confirmation: { required: true, from_stage: 'invalid' },
      });

      await hydrator.applySnapshot(status);

      // from_stage is NaN, so falls back to resolvePreferredStage → current_stage = 3
      expect(deps.stageCtrl.activateStage).toHaveBeenCalledWith(3);
    });

    it('activates mesh repair for interactive stage 7', async () => {
      const status = mkStatus({
        running: true,
        current_stage: 7,
        mesh_repair: { ready: true },
        stages: {
          '7': mkStage('interactive', 0),
        },
      });

      await hydrator.applySnapshot(status);

      expect(deps.meshRepair.activateFromApi).toHaveBeenCalledTimes(1);
      expect(deps.stageCtrl.activateStage).toHaveBeenCalledWith(7);
    });

    it('handles mesh repair activation failure gracefully', async () => {
      deps.meshRepair.activateFromApi.mockRejectedValue(new Error('API error'));
      const status = mkStatus({
        running: true,
        current_stage: 7,
        mesh_repair: { ready: true },
        stages: {
          '7': mkStage('interactive', 0),
        },
      });

      await hydrator.applySnapshot(status);

      expect(deps.meshRepair.setToolbarVisible).toHaveBeenCalledWith(false);
      expect(deps.meshRepair.setStatus).toHaveBeenCalledWith(
        'Mesh Repair UI failed: API error',
        'error',
      );
      expect(deps.stageCtrl.activateStage).toHaveBeenCalledWith(7);
    });
  });

  // ── applySnapshot idle ───────────────────────────────────────

  describe('applySnapshot idle', () => {
    it('sets idle state', async () => {
      const status = mkStatus({ running: false });
      await hydrator.applySnapshot(status);

      expect(deps.config.setRunning).toHaveBeenCalledWith(false);
      expect(deps.config.setActiveStage).toHaveBeenCalledWith(null);
      expect(deps.setStatus).toHaveBeenCalledWith('idle', 'Idle');
    });

    it('resets mesh repair', async () => {
      const status = mkStatus({ running: false });
      await hydrator.applySnapshot(status);

      expect(deps.meshRepair.resetState).toHaveBeenCalledWith({ resetThreshold: true });
      expect(deps.meshRepair.setToolbarVisible).toHaveBeenCalledWith(false);
      expect(deps.meshRepair.setStatus).toHaveBeenCalledWith('');
    });

    it('hydrates outputs for idle status with object_name', async () => {
      const status = mkStatus({ running: false });
      await hydrator.applySnapshot(status);

      // hydrateOutputs should have been called — preview.reset is evidence
      expect(deps.preview.reset).toHaveBeenCalled();
    });

    it('does not hydrate outputs without object_name', async () => {
      const status = mkStatus({ running: false, object_name: '' });
      await hydrator.applySnapshot(status);

      expect(deps.preview.reset).not.toHaveBeenCalled();
    });
  });

  // ── applySnapshot all done ───────────────────────────────────

  describe('applySnapshot all done', () => {
    it('sets complete status when all stages are done', async () => {
      const status = mkAllDone({ running: false });
      await hydrator.applySnapshot(status);

      expect(deps.setStatus).toHaveBeenCalledWith('complete', 'Done');
    });

    it('sets idle when not all stages done', async () => {
      const status = mkStatus({
        running: false,
        stages: {
          '1': mkStage('complete', 100),
          '2': mkStage('running', 50),
        },
      });
      await hydrator.applySnapshot(status);

      expect(deps.setStatus).toHaveBeenCalledWith('idle', 'Idle');
    });
  });

  // ── applySnapshot mesh method ────────────────────────────────

  describe('applySnapshot mesh method', () => {
    it('applies mesh method from status', async () => {
      const status = mkStatus({ mesh_method: 'diffcd' });
      deps.getMeshMethod.mockReturnValue('diffcd');
      await hydrator.applySnapshot(status);

      expect(deps.applyMeshMethod).toHaveBeenCalledWith('diffcd', { announce: false });
    });

    it('falls back to getMeshMethod when mesh_method not in status', async () => {
      const status = mkStatus();
      delete status.mesh_method;
      deps.getMeshMethod.mockReturnValue('poisson');
      await hydrator.applySnapshot(status);

      expect(deps.applyMeshMethod).toHaveBeenCalledWith('poisson', { announce: false });
    });

    it('clears mesh phase status for non-poisson method', async () => {
      const status = mkStatus({ mesh_method: 'diffcd' });
      deps.getMeshMethod.mockReturnValue('diffcd');
      await hydrator.applySnapshot(status);

      expect(deps.setMeshPhaseStatus).toHaveBeenCalledWith('', '');
    });

    it('does not clear mesh phase status for poisson method', async () => {
      const status = mkStatus({ mesh_method: 'poisson' });
      deps.getMeshMethod.mockReturnValue('poisson');
      await hydrator.applySnapshot(status);

      // setMeshPhaseStatus should NOT have been called with ('', '') in the method check
      // (it may be called later during hydrateOutputs for other reasons)
      const calls = deps.setMeshPhaseStatus.mock.calls;
      const clearCalls = calls.filter(
        ([msg, state]) => msg === '' && state === '',
      );
      expect(clearCalls.length).toBe(0);
    });

    it('disables mesh method selector when running', async () => {
      const status = mkStatus({ running: true, current_stage: 1 });
      await hydrator.applySnapshot(status);

      expect(deps.pipelineUI.setMeshMethodEnabled).toHaveBeenCalledWith(false);
    });

    it('enables mesh method selector when idle', async () => {
      const status = mkStatus({ running: false });
      await hydrator.applySnapshot(status);

      expect(deps.pipelineUI.setMeshMethodEnabled).toHaveBeenCalledWith(true);
    });
  });

  // ── applySnapshot common behavior ────────────────────────────

  describe('applySnapshot common behavior', () => {
    it('stores snapshot', async () => {
      const status = mkStatus();
      await hydrator.applySnapshot(status);
      expect(hydrator.latestSnapshot).toBe(status);
    });

    it('applies checkpoint snapshot', async () => {
      const status = mkStatus();
      await hydrator.applySnapshot(status);
      expect(deps.checkpoints.applyStatusSnapshot).toHaveBeenCalledWith(status);
    });

    it('updates all pipeline UI', async () => {
      const status = mkStatus();
      await hydrator.applySnapshot(status);
      expect(deps.pipelineUI.updateAll).toHaveBeenCalledWith(status);
    });

    it('sets stage states for all 8 stages', async () => {
      const status = mkStatus({
        stages: {
          '1': mkStage('complete', 100),
          '3': mkStage('running', 50),
        },
      });
      await hydrator.applySnapshot(status);

      expect(deps.stageCtrl.setStageState).toHaveBeenCalledTimes(8);
      expect(deps.stageCtrl.setStageState).toHaveBeenCalledWith(1, 'complete');
      expect(deps.stageCtrl.setStageState).toHaveBeenCalledWith(2, 'pending');
      expect(deps.stageCtrl.setStageState).toHaveBeenCalledWith(3, 'running');
    });

    it('syncs task confirm, meshPost, meshRepair from status', async () => {
      const status = mkStatus();
      await hydrator.applySnapshot(status);

      expect(deps.taskConfirm.syncFromStatus).toHaveBeenCalledWith(status);
      expect(deps.meshPost.syncFromStatus).toHaveBeenCalledWith(status);
      expect(deps.meshRepair.syncFromStatus).toHaveBeenCalledWith(status);
    });

    it('sets overall progress from status', async () => {
      const status = mkStatus({ overall_progress: 75 });
      await hydrator.applySnapshot(status);
      expect(deps.setOverallProgress).toHaveBeenCalledWith(75);
    });

    it('falls back to pipelineUI.getOverallProgress when not in status', async () => {
      const status = mkStatus();
      delete status.overall_progress;
      deps.pipelineUI.getOverallProgress.mockReturnValue(42);
      await hydrator.applySnapshot(status);
      expect(deps.setOverallProgress).toHaveBeenCalledWith(42);
    });

    it('sets object name from status', async () => {
      const status = mkStatus({ object_name: 'my-object' });
      await hydrator.applySnapshot(status);
      expect(deps.config.setObjectName).toHaveBeenCalledWith('my-object');
    });

    it('does not set object name when empty', async () => {
      const status = mkStatus({ object_name: '' });
      await hydrator.applySnapshot(status);
      expect(deps.config.setObjectName).not.toHaveBeenCalled();
    });
  });

  // ── applySnapshot forceHydrate ───────────────────────────────

  describe('applySnapshot forceHydrate', () => {
    it('passes force flag through to hydrateOutputs', async () => {
      const status = mkStatus({ running: false });
      document.body.innerHTML = '<div id="stage-panel-1"><div class="stage-panel-empty"></div></div>';

      // First apply — hydrates
      await hydrator.applySnapshot(status);
      expect(deps.preview.reset).toHaveBeenCalledTimes(1);

      // Second apply without force — same key, hydrateOutputs skips
      await hydrator.applySnapshot(status);
      expect(deps.preview.reset).toHaveBeenCalledTimes(1);

      // Third apply with forceHydrate — re-hydrates
      await hydrator.applySnapshot(status, { forceHydrate: true });
      expect(deps.preview.reset).toHaveBeenCalledTimes(2);
    });
  });
});
