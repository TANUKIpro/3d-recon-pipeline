/**
 * Status snapshot hydration — applies pipeline status updates to all UI modules.
 */

import { STAGE_COUNT } from './constants.js';
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
    this._deps.postTextureCleanupReview?.reset?.();
  }

  resetHydratedKey() { this._hydratedStatusKey = ''; }

  async hydrateOutputs(statusMsg, opts = {}) {
    const {
      preview,
      cameraOverlay,
      sam2,
      sam2Verify,
      stageCtrl,
      postTextureCleanupReview,
    } = this._deps;
    if (!statusMsg?.object_name) return;

    const key = buildStatusKey(statusMsg);
    if (!opts.force && key === this._hydratedStatusKey) {
      return;
    }
    this._hydratedStatusKey = key;

    cameraOverlay.remove();
    sam2.deactivate();
    postTextureCleanupReview?.reset?.();
    preview.reset();

    // If stage 3 is done, show frozen verification; otherwise hide
    if (isStageDone(statusMsg, 3)) {
      const frameCount = Number(statusMsg.frame_count) || 0;
      const hasGround = statusMsg.has_ground_plane === true;
      if (frameCount > 0) {
        sam2Verify.showFrozen(frameCount, hasGround);
      } else {
        sam2Verify.hide();
      }
    } else {
      sam2Verify.hide();
    }

    if (isStageDone(statusMsg, 1)) {
      const empty = document.querySelector('#stage-panel-1 .stage-panel-empty');
      if (empty) empty.classList.add('hidden');
      await preview.loadGallery(Number(statusMsg.frame_count) || 0);
    }

    if (isStageAvailable(statusMsg, 2)) {
      await preview.loadColmapResults(cameraOverlay);
    }

    if (isStageDone(statusMsg, 4)) await preview.loadStageResult(4);
    if (isStageDone(statusMsg, 5)) await preview.loadStageResult(5);
    if (isStageDone(statusMsg, 6)) await preview.loadStageResult(6);

    if (opts.activate !== false) {
      stageCtrl.activateStage(resolvePreferredStage(statusMsg));
    }
  }

  async applySnapshot(statusMsg, opts = {}) {
    const {
      checkpoints, pipelineUI, stageCtrl, config,
      taskConfirm,
      postTextureCleanupReview,
      setStatus, setOverallProgress,
      preview,
    } = this._deps;
    preview?.setActiveObjectName?.(statusMsg?.object_name || '');

    this._latestStatusSnapshot = statusMsg;
    checkpoints.applyStatusSnapshot(statusMsg);
    pipelineUI.updateAll(statusMsg);

    for (let i = 1; i <= STAGE_COUNT; i++) {
      const info = getStageInfo(statusMsg, i);
      stageCtrl.setStageState(i, info?.status || 'pending');
    }

    taskConfirm.syncFromStatus(statusMsg);
    setOverallProgress(statusMsg.overall_progress ?? pipelineUI.getOverallProgress());
    await postTextureCleanupReview?.syncFromStatus?.(statusMsg);

    if (statusMsg.object_name) {
      const shouldApplyConfig = opts.applyConfig !== false;
      config.setObjectName(statusMsg.object_name, { applyConfig: shouldApplyConfig });
    }

    if (statusMsg.running) {
      config.setRunning(true);
      setStatus('running', 'Running');
      const waiting = statusMsg?.next_stage_confirmation?.required === true;
      const waitingFromStage = Number(statusMsg?.next_stage_confirmation?.from_stage);
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
      } else {
        stageCtrl.activateStage(resolvePreferredStage(statusMsg));
      }
      return;
    }

    config.setRunning(false);
    config.setActiveStage(null);

    const allDone = Array.from({ length: STAGE_COUNT }, (_, idx) => idx + 1)
      .every((stage) => isStageDone(statusMsg, stage));
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
