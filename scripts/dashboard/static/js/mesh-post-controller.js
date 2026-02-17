/**
 * Mesh post-process toolbar UI controller.
 *
 * Manages smoothing/reset operations on the classical mesh stage output.
 */

import { DEFAULT_TAUBIN_NU, CLASSICAL_PREVIEW_TITLE } from './constants.js';
import { isStageDone } from './pipeline-status.js';

export class MeshPostController {
  /**
   * @param {object} deps
   * @param {object} deps.preview - PreviewPanel instance
   * @param {object} deps.stageCtrl - StageController instance
   * @param {function} deps.appendLog - appendLog(stream, text, opts)
   * @param {function} deps.getMeshMethod - () => current mesh method string
   * @param {function} deps.isPipelineRunning - () => boolean
   * @param {function} deps.setMeshPhaseStatus - (message, tone) => void
   * @param {function} deps.applyStatusSnapshot - async (status, opts) => void
   */
  constructor({
    preview, stageCtrl, appendLog,
    getMeshMethod, isPipelineRunning,
    setMeshPhaseStatus, applyStatusSnapshot,
  }) {
    this._preview = preview;
    this._stageCtrl = stageCtrl;
    this._appendLog = appendLog;
    this._getMeshMethod = getMeshMethod;
    this._isPipelineRunning = isPipelineRunning;
    this._setMeshPhaseStatus = setMeshPhaseStatus;
    this._applyStatusSnapshot = applyStatusSnapshot;
    this._inFlight = false;

    this._toolbar = document.getElementById('mesh-post-toolbar');
    this._methodInput = document.getElementById('mesh-post-method');
    this._iterationsInput = document.getElementById('mesh-post-iterations');
    this._lambdaInput = document.getElementById('mesh-post-lambda');
    this._downsampleEnabledInput = document.getElementById('mesh-post-downsample-enabled');
    this._targetFacesInput = document.getElementById('mesh-post-target-faces');
    this._triggerFacesInput = document.getElementById('mesh-post-trigger-faces');
    this._applyBtn = document.getElementById('mesh-post-apply');
    this._resetBtn = document.getElementById('mesh-post-reset');
    this._statusEl = document.getElementById('mesh-post-status');
  }

  get inFlight() { return this._inFlight; }

  init() {
    if (!this._toolbar) return;
    this.setToolbarVisible(false);
    this.setEnabled(false);
    this.setStatus('');

    this._applyBtn?.addEventListener('click', () => {
      this.applyPostprocess({ resetToRaw: false });
    });
    this._resetBtn?.addEventListener('click', () => {
      this.applyPostprocess({ resetToRaw: true });
    });
  }

  setToolbarVisible(visible) {
    if (!this._toolbar) return;
    this._toolbar.style.display = visible ? 'flex' : 'none';
  }

  setEnabled(enabled) {
    const disabled = !enabled;
    if (this._methodInput) this._methodInput.disabled = disabled;
    if (this._iterationsInput) this._iterationsInput.disabled = disabled;
    if (this._lambdaInput) this._lambdaInput.disabled = disabled;
    if (this._downsampleEnabledInput) this._downsampleEnabledInput.disabled = disabled;
    if (this._targetFacesInput) this._targetFacesInput.disabled = disabled;
    if (this._triggerFacesInput) this._triggerFacesInput.disabled = disabled;
    if (this._applyBtn) this._applyBtn.disabled = disabled;
    if (this._resetBtn) this._resetBtn.disabled = disabled;
  }

  setStatus(message, tone = '') {
    if (!this._statusEl) return;
    this._statusEl.textContent = String(message || '');
    this._statusEl.classList.remove('error', 'success');
    if (tone === 'error' || tone === 'success') {
      this._statusEl.classList.add(tone);
    }
  }

  canRun(statusMsg) {
    if (!statusMsg) return true;
    if (!statusMsg.running) return true;
    const next = statusMsg.next_stage_confirmation || {};
    return next.required === true && Number(next.from_stage) === 5;
  }

  syncFromStatus(statusMsg) {
    if (!statusMsg) return;
    const ready = Boolean(statusMsg?.object_name) && isStageDone(statusMsg, 5);
    this.setToolbarVisible(ready);
    if (!ready) {
      this.setEnabled(false);
      if (!this._inFlight) {
        this.setStatus('');
      }
      return;
    }

    const allowed = this.canRun(statusMsg);
    this.setEnabled(allowed && !this._inFlight);
    if (this._inFlight) return;

    if (!allowed) {
      this.setStatus('Mesh post-process is disabled while pipeline is running.');
      return;
    }

    if (this._statusEl?.textContent === 'Mesh post-process is disabled while pipeline is running.') {
      this.setStatus('');
    }
  }

  async applyPostprocess({ resetToRaw = false } = {}) {
    if (this._inFlight) return;

    const method = resetToRaw ? 'laplacian' : (this._methodInput?.value || 'laplacian');
    const iterationsRaw = Number.parseInt(this._iterationsInput?.value || '8', 10);
    const iterations = resetToRaw ? 0 : Math.max(0, Math.min(100, Number.isFinite(iterationsRaw) ? iterationsRaw : 8));
    const lambRaw = Number.parseFloat(this._lambdaInput?.value || '0.5');
    const lamb = Math.max(0.01, Math.min(1.5, Number.isFinite(lambRaw) ? lambRaw : 0.5));
    const downsampleEnabled = this._downsampleEnabledInput?.checked ?? true;
    const targetRaw = Number.parseInt(this._targetFacesInput?.value || '120000', 10);
    const downsampleTargetFaces = Math.max(
      1000,
      Math.min(5_000_000, Number.isFinite(targetRaw) ? targetRaw : 120000),
    );
    const triggerRaw = Number.parseInt(this._triggerFacesInput?.value || '170000', 10);
    const downsampleTriggerFaces = Math.max(
      downsampleTargetFaces,
      Math.min(5_000_000, Number.isFinite(triggerRaw) ? triggerRaw : 170000),
    );

    if (this._targetFacesInput) this._targetFacesInput.value = String(downsampleTargetFaces);
    if (this._triggerFacesInput) this._triggerFacesInput.value = String(downsampleTriggerFaces);

    this._inFlight = true;
    this.setEnabled(false);
    this.setStatus(resetToRaw ? 'Resetting to raw mesh...' : 'Applying smoothing...');

    try {
      const res = await fetch('/api/mesh/postprocess', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          method,
          iterations,
          lamb,
          taubin_nu: DEFAULT_TAUBIN_NU,
          downsample_enabled: downsampleEnabled,
          downsample_target_faces: downsampleTargetFaces,
          downsample_trigger_faces: downsampleTriggerFaces,
          source: 'raw',
          invalidate_texture: true,
        }),
      });
      const data = await res.json();
      if (!res.ok || data.error) {
        throw new Error(data.error || `HTTP ${res.status}`);
      }

      const meshMethod = this._getMeshMethod();
      await this._preview.loadStageResult(5, { preferPreview: meshMethod === 'poisson' });
      if (meshMethod === 'poisson') {
        this._setMeshPhaseStatus(`Showing: ${CLASSICAL_PREVIEW_TITLE}`, 'ready');
      }
      this._stageCtrl.activateStage(5);

      const vertices = Number(data.vertices) || 0;
      const faces = Number(data.faces) || 0;
      const apiDownsampleEnabled = data.downsample_enabled !== false;
      const apiTargetFaces = Number(data.downsample_target_faces) || downsampleTargetFaces;
      const apiTriggerFaces = Number(data.downsample_trigger_faces) || downsampleTriggerFaces;
      if (this._downsampleEnabledInput) {
        this._downsampleEnabledInput.checked = apiDownsampleEnabled;
      }
      if (this._targetFacesInput) {
        this._targetFacesInput.value = String(apiTargetFaces);
      }
      if (this._triggerFacesInput) {
        this._triggerFacesInput.value = String(apiTriggerFaces);
      }
      let msg = resetToRaw
        ? `Reset complete (${vertices.toLocaleString()}v / ${faces.toLocaleString()}f).`
        : `Smoothing complete (${vertices.toLocaleString()}v / ${faces.toLocaleString()}f).`;
      if (!apiDownsampleEnabled) {
        msg += ' Downsample disabled.';
      } else if (data.downsample_applied) {
        msg += ' Downsample applied.';
      } else {
        msg += ` Downsample skipped (target=${apiTargetFaces.toLocaleString()}, trigger=${apiTriggerFaces.toLocaleString()}).`;
      }
      if (data.texture_invalidated) {
        msg += ' Downstream Stage 6/7/8 artifacts were cleared.';
      }
      this.setStatus(msg, 'success');

      this._appendLog(
        'stdout',
        `Mesh post-process: method=${data.method}, iterations=${data.iterations}, lambda=${Number(data.lamb).toFixed(2)}, source=${data.source}, downsample_enabled=${apiDownsampleEnabled ? 'yes' : 'no'}, target_faces=${apiTargetFaces}, trigger_faces=${apiTriggerFaces}, applied=${data.downsample_applied ? 'yes' : 'no'}\n`,
        { stage: 5 },
      );
      if (data.texture_invalidated) {
        this._appendLog('stdout', 'Wrap/repair/texture outputs removed. Re-run from Stage 6 for updated texturing.\n', { stage: 6 });
      }

      if (!this._isPipelineRunning()) {
        try {
          const statusRes = await fetch('/api/pipeline/status', { cache: 'no-store' });
          if (statusRes.ok) {
            const status = await statusRes.json();
            await this._applyStatusSnapshot(status, { forceHydrate: true });
            this._stageCtrl.activateStage(5);
          }
        } catch (statusErr) {
          console.warn('Failed to refresh status after mesh post-process:', statusErr);
        }
      }
    } catch (e) {
      this.setStatus(`Failed: ${e.message}`, 'error');
      this._appendLog('stderr', `Mesh post-process failed: ${e.message}\n`, { stage: 5 });
    } finally {
      this._inFlight = false;
      this.onPostprocessComplete?.();
    }
  }
}
