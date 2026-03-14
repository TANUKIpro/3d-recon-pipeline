/**
 * Mesh repair toolbar UI controller.
 *
 * Manages the threshold slider, selection display, and confirm/clear actions
 * for the interactive mesh-repair stage.
 */

import { MESH_REPAIR_THRESHOLD_DEFAULT } from './constants.js';
import { clampMeshRepairThreshold, formatMeshRepairThreshold } from './utils.js';
import { getStageInfo } from './pipeline-status.js';

export class MeshRepairController {
  /**
   * @param {object} deps
   * @param {object} deps.preview - PreviewPanel instance
   * @param {function} deps.appendLog - appendLog(stream, text, opts)
   */
  constructor({ preview, appendLog }) {
    this._preview = preview;
    this._appendLog = appendLog;
    this._candidateCount = 0;
    this._inFlight = false;
    this._threshold = MESH_REPAIR_THRESHOLD_DEFAULT;

    this._toolbar = document.getElementById('mesh-repair-toolbar');
    this._statusEl = document.getElementById('mesh-repair-status');
    this._thresholdInput = document.getElementById('mesh-repair-threshold');
    this._thresholdValueEl = document.getElementById('mesh-repair-threshold-value');
    this._clearBtn = document.getElementById('mesh-repair-clear');
    this._applyBtn = document.getElementById('mesh-repair-apply');
  }

  get inFlight() { return this._inFlight; }
  get candidateCount() { return this._candidateCount; }
  set candidateCount(v) { this._candidateCount = v; }

  init() {
    if (!this._toolbar) return;
    this.setToolbarVisible(false);
    this.setEnabled(false);
    this.resetThreshold({ applyPreview: false });
    this.setStatus('');
    this._thresholdInput?.addEventListener('input', (event) => {
      this.setThresholdValue(event?.target?.value, { applyPreview: true });
      this.updateStatus();
    });
    this._clearBtn?.addEventListener('click', () => {
      this._preview.clearMeshRepairSelection();
      this.updateStatus();
    });
    this._applyBtn?.addEventListener('click', () => {
      this.applySelection();
    });
  }

  setToolbarVisible(visible) {
    if (!this._toolbar) return;
    this._toolbar.classList.toggle('hidden', !visible);
  }

  setEnabled(enabled) {
    const disabled = !enabled;
    if (this._thresholdInput) this._thresholdInput.disabled = disabled;
    if (this._clearBtn) this._clearBtn.disabled = disabled;
    if (this._applyBtn) this._applyBtn.disabled = disabled;
  }

  setStatus(message, tone = '') {
    if (!this._statusEl) return;
    this._statusEl.textContent = String(message || '');
    this._statusEl.classList.remove('error', 'success');
    if (tone === 'error' || tone === 'success') {
      this._statusEl.classList.add(tone);
    }
  }

  setThresholdValue(value, { applyPreview = true } = {}) {
    const next = clampMeshRepairThreshold(value);
    this._threshold = next;
    const text = formatMeshRepairThreshold(next);
    if (this._thresholdInput) {
      this._thresholdInput.value = text;
    }
    if (this._thresholdValueEl) {
      this._thresholdValueEl.textContent = text;
    }
    if (applyPreview) {
      this._preview.setMeshRepairThreshold(next);
    }
    return next;
  }

  resetThreshold(opts = {}) {
    return this.setThresholdValue(MESH_REPAIR_THRESHOLD_DEFAULT, opts);
  }

  resetState({ resetThreshold = true } = {}) {
    this._candidateCount = 0;
    this._inFlight = false;
    if (resetThreshold) {
      this.resetThreshold({ applyPreview: true });
    }
  }

  getVisibleCount(candidateCount = this._candidateCount) {
    const candidates = Math.max(0, Number(candidateCount) || 0);
    const previewTotalRaw = Number(this._preview.getMeshRepairTotalLoopCount?.());
    const previewVisibleRaw = Number(this._preview.getMeshRepairVisibleLoopCount?.());
    const previewTotal = Number.isFinite(previewTotalRaw) ? previewTotalRaw : 0;
    const previewVisible = Number.isFinite(previewVisibleRaw) ? previewVisibleRaw : NaN;

    if (!Number.isFinite(previewVisible) || previewVisible < 0) {
      return candidates;
    }
    if (previewTotal <= 0) {
      return candidates;
    }
    if (candidates > 0) {
      return Math.max(0, Math.min(candidates, Math.round(previewVisible)));
    }
    return Math.max(0, Math.round(previewVisible));
  }

  countLabel(
    selectedCount,
    candidateCount = this._candidateCount,
    visibleCount = this.getVisibleCount(candidateCount),
    threshold = this._threshold,
  ) {
    const selected = Math.max(0, Number(selectedCount) || 0);
    const candidates = Math.max(0, Number(candidateCount) || 0);
    const visible = Math.max(0, Number(visibleCount) || 0);
    const th = formatMeshRepairThreshold(threshold);
    return `Closed surfaces: ${selected}/${candidates} selected (visible ${visible}/${candidates}, threshold <= ${th}).`;
  }

  updateStatus(selectedIds = null) {
    if (!this._toolbar || this._toolbar.classList.contains('hidden')) return;
    const selected = Array.isArray(selectedIds)
      ? selectedIds.length
      : this._preview.getMeshRepairSelectedLoopIds().length;
    const visible = this.getVisibleCount(this._candidateCount);
    this.setStatus(this.countLabel(selected, this._candidateCount, visible, this._threshold));
  }

  async activateFromApi() {
    const res = await fetch('/api/mesh-repair/candidates', { cache: 'no-store' });
    const data = await res.json();
    if (!res.ok || data.error) {
      throw new Error(data.error || `HTTP ${res.status}`);
    }
    this.resetThreshold({ applyPreview: false });
    await this._preview.beginMeshRepairSelection(data);
    this._candidateCount = Number(data.loop_count) || 0;
    this._inFlight = false;
    this.setThresholdValue(this._threshold, { applyPreview: true });
    this.setToolbarVisible(true);
    this.setEnabled(true);
    this.updateStatus([]);
  }

  async applySelection() {
    if (this._inFlight) return;
    const selectedLoopIds = this._preview.getMeshRepairSelectedLoopIds();
    const selectedCount = Array.isArray(selectedLoopIds) ? selectedLoopIds.length : 0;
    const candidateCount = this._candidateCount;

    this._inFlight = true;
    this.setEnabled(false);
    this.setStatus(
      selectedCount > 0
        ? `${this.countLabel(selectedCount, candidateCount)} Confirming repair selection...`
        : `${this.countLabel(0, candidateCount)} No surface selected. Confirming skip...`,
    );
    try {
      const res = await fetch('/api/mesh-repair/confirm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ selected_loop_ids: selectedLoopIds || [] }),
      });
      const data = await res.json();
      if (!res.ok || data.error) {
        throw new Error(data.error || `HTTP ${res.status}`);
      }
      const confirmedCountRaw = Number(data.selected_count);
      const confirmedCount = Number.isFinite(confirmedCountRaw) ? confirmedCountRaw : selectedCount;
      this._preview.setMeshRepairConfirmed();
      if (confirmedCount > 0) {
        this.setStatus(
          `${this.countLabel(confirmedCount, candidateCount)} Repairing mesh...`,
          'success',
        );
        this._appendLog('stdout', `Mesh Repair confirmed with ${confirmedCount} selected loops.\n`, { stage: 7 });
      } else {
        this.setStatus(
          `${this.countLabel(0, candidateCount)} Skipping repair and continuing...`,
          'success',
        );
        this._appendLog('stdout', 'Mesh Repair confirmed with 0 selected loops. Skipping repair.\n', { stage: 7 });
      }
    } catch (e) {
      this._inFlight = false;
      this.setEnabled(true);
      this.setStatus(`Failed to confirm repair selection: ${e.message}`, 'error');
      this._appendLog('stderr', `Mesh Repair confirm failed: ${e.message}\n`, { stage: 7 });
    }
  }

  syncFromStatus(statusMsg) {
    const stage7 = getStageInfo(statusMsg, 7);
    const ready = statusMsg?.running === true
      && stage7?.status === 'interactive'
      && statusMsg?.mesh_repair?.ready === true;
    if (!ready) {
      if (!this._inFlight) {
        this.setToolbarVisible(false);
        this.setStatus('');
      }
      return;
    }
    if (this._inFlight) return;
    this.setToolbarVisible(true);
    this.setEnabled(true);
    const selected = Number(statusMsg?.mesh_repair?.selected_loop_count) || 0;
    const candidateCount = Number(statusMsg?.mesh_repair?.candidate_count) || this._candidateCount;
    this._candidateCount = candidateCount;
    const visible = this.getVisibleCount(candidateCount);
    this.setStatus(this.countLabel(selected, candidateCount, visible, this._threshold));
  }
}
