/**
 * Status snapshot hydration — applies pipeline status updates to all UI modules.
 */

import { STAGE_COUNT, CLASSICAL_PREVIEW_TITLE } from './constants.js';
import { normalizeMeshMethod } from './utils.js';
import {
  getStageInfo,
  isStageDone,
  isStageAvailable,
  resolvePreferredStage,
  buildStatusKey,
} from './pipeline-status.js';

export class StatusHydrator {
  /**
   * @param {object} deps - All module references needed for hydration.
   */
  constructor(deps) {
    this._deps = deps;
    this._hydratedStatusKey = '';
    this._latestStatusSnapshot = null;
  }

  get latestSnapshot() { return this._latestStatusSnapshot; }

  reset() {
    this._hydratedStatusKey = '';
    this._latestStatusSnapshot = null;
  }

  resetHydratedKey() { this._hydratedStatusKey = ''; }

  async hydrateOutputs(statusMsg, opts = {}) {
    const { preview, cameraOverlay, sam2, sam2Verify, config, stageCtrl, setMeshPhaseStatus, getMeshMethod } = this._deps;
    if (!statusMsg?.object_name) return;
    const statusMeshMethod = normalizeMeshMethod(statusMsg?.mesh_method || getMeshMethod());
    const isPoissonMesh = statusMeshMethod === 'poisson';

    const key = buildStatusKey(statusMsg);
    if (!opts.force && key === this._hydratedStatusKey) {
      return;
    }
    this._hydratedStatusKey = key;

    cameraOverlay.remove();
    sam2.deactivate();
    sam2Verify.hide();
    preview.reset();

    if (isStageDone(statusMsg, 1)) {
      const empty = document.querySelector('#stage-panel-1 .stage-panel-empty');
      if (empty) empty.classList.add('hidden');
      await preview.loadGallery(Number(statusMsg.frame_count) || 0);
    }

    if (isStageAvailable(statusMsg, 2)) {
      const pi3xFile = isStageDone(statusMsg, 3) ? 'object.ply' : 'object_full.ply';
      await preview.loadPi3xResults(cameraOverlay, pi3xFile);
    }

    if (isStageDone(statusMsg, 4)) await preview.loadStageResult(4);
    if (isStageDone(statusMsg, 5)) {
      await preview.loadStageResult(5, { preferPreview: isPoissonMesh });
      if (isPoissonMesh) {
        setMeshPhaseStatus(`Showing: ${CLASSICAL_PREVIEW_TITLE}`, 'ready');
      } else {
        setMeshPhaseStatus('', '');
      }
    } else if (statusMsg.running && Number(statusMsg.current_stage) === 5 && isPoissonMesh) {
      setMeshPhaseStatus('Running: Classical Mesh', 'live');
    }
    if (isStageDone(statusMsg, 5) && !isStageDone(statusMsg, 6)) {
      await preview.showCropBbox(config.getCropScale(), { preferPreview: isPoissonMesh });
    }
    if (isStageDone(statusMsg, 6)) await preview.loadStageResult(6);
    if (isStageDone(statusMsg, 7)) await preview.loadStageResult(7);
    if (isStageDone(statusMsg, 8)) await preview.loadStageResult(8);

    if (opts.activate !== false) {
      stageCtrl.activateStage(resolvePreferredStage(statusMsg));
    }
  }

  async applySnapshot(statusMsg, opts = {}) {
    const {
      checkpoints, pipelineUI, stageCtrl, config,
      taskConfirm, meshPost, meshRepair,
      applyMeshMethod, getMeshMethod,
      setMeshPhaseStatus, setStatus, setOverallProgress,
    } = this._deps;

    this._latestStatusSnapshot = statusMsg;
    checkpoints.applyStatusSnapshot(statusMsg);
    applyMeshMethod(statusMsg?.mesh_method || getMeshMethod(), { announce: false });
    if (normalizeMeshMethod(statusMsg?.mesh_method || getMeshMethod()) !== 'poisson') {
      setMeshPhaseStatus('', '');
    }
    pipelineUI.setMeshMethodEnabled(statusMsg?.running !== true);
    pipelineUI.updateAll(statusMsg);

    for (let i = 1; i <= STAGE_COUNT; i++) {
      const info = getStageInfo(statusMsg, i);
      stageCtrl.setStageState(i, info?.status || 'pending');
    }

    taskConfirm.syncFromStatus(statusMsg);
    meshPost.syncFromStatus(statusMsg);
    meshRepair.syncFromStatus(statusMsg);
    setOverallProgress(statusMsg.overall_progress ?? pipelineUI.getOverallProgress());

    if (statusMsg.object_name) {
      config.setObjectName(statusMsg.object_name);
    }

    if (statusMsg.running) {
      config.setRunning(true);
      setStatus('running', 'Running');
      const waiting = statusMsg?.next_stage_confirmation?.required === true;
      const waitingFromStage = Number(statusMsg?.next_stage_confirmation?.from_stage);
      const meshRepairInteractive = statusMsg?.mesh_repair?.ready === true
        && getStageInfo(statusMsg, 7)?.status === 'interactive';
      if (waiting) {
        await this.hydrateOutputs(statusMsg, {
          force: opts.forceHydrate === true,
          activate: false,
        });
        if (Number.isFinite(waitingFromStage) && waitingFromStage >= 1 && waitingFromStage <= STAGE_COUNT) {
          stageCtrl.activateStage(waitingFromStage);
        } else {
          stageCtrl.activateStage(resolvePreferredStage(statusMsg));
        }
      } else if (meshRepairInteractive) {
        await this.hydrateOutputs(statusMsg, {
          force: opts.forceHydrate === true,
          activate: false,
        });
        try {
          await meshRepair.activateFromApi();
        } catch (e) {
          meshRepair.setToolbarVisible(false);
          meshRepair.setStatus(`Mesh Repair UI failed: ${e.message}`, 'error');
        }
        stageCtrl.activateStage(7);
      } else {
        stageCtrl.activateStage(resolvePreferredStage(statusMsg));
      }
      return;
    }

    config.setRunning(false);
    config.setActiveStage(null);
    meshRepair.resetState({ resetThreshold: true });
    meshRepair.setToolbarVisible(false);
    meshRepair.setStatus('');

    const allDone = [1, 2, 3, 4, 5, 6, 7, 8].every((stage) => isStageDone(statusMsg, stage));
    if (allDone) setStatus('complete', 'Done');
    else setStatus('idle', 'Idle');

    if (statusMsg.object_name) {
      await this.hydrateOutputs(statusMsg, {
        force: opts.forceHydrate === true,
        activate: true,
      });
    }
  }
}
