/**
 * Post-texture cleanup review UI controller.
 */

export class PostTextureCleanupReviewController {
  constructor({ preview, stageCtrl, appendLog }) {
    this._preview = preview;
    this._stageCtrl = stageCtrl;
    this._appendLog = appendLog;
    this._proposal = null;
    this._pending = false;

    this._panel = document.getElementById('post-texture-cleanup-review');
    this._summary = document.getElementById('post-cleanup-review-summary');
    this._stats = document.getElementById('post-cleanup-review-stats');
    this._applyBtn = document.getElementById('post-cleanup-apply');
    this._skipBtn = document.getElementById('post-cleanup-skip');
    this._overlayToggle = document.getElementById('post-cleanup-overlay-toggle');

    this._applyBtn?.addEventListener('click', () => this.confirm('apply'));
    this._skipBtn?.addEventListener('click', () => this.confirm('skip'));
    this._overlayToggle?.addEventListener('change', () => this._syncOverlay());
  }

  reset() {
    this._proposal = null;
    this._pending = false;
    if (this._overlayToggle) this._overlayToggle.checked = true;
    if (this._summary) this._summary.textContent = '';
    if (this._stats) this._stats.textContent = '';
    this.hide({ keepOverlay: false });
  }

  hide({ keepOverlay = false } = {}) {
    this._proposal = keepOverlay ? this._proposal : null;
    this._pending = false;
    if (this._panel) this._panel.classList.add('hidden');
    if (!keepOverlay) {
      if (this._summary) this._summary.textContent = '';
      if (this._stats) this._stats.textContent = '';
    }
    if (!keepOverlay) {
      this._preview.clearStageOverlay(6);
    }
    if (this._applyBtn) this._applyBtn.textContent = 'Apply Cleanup';
    if (this._skipBtn) this._skipBtn.textContent = 'Skip Cleanup';
    this._setButtonsEnabled(false);
  }

  async loadProposal() {
    const res = await fetch('/api/post-texture-cleanup/proposal');
    const data = await res.json();
    if (!res.ok || !data?.proposal) {
      throw new Error(data?.error || `HTTP ${res.status}`);
    }
    return data.proposal;
  }

  async showReview(proposal) {
    this._proposal = proposal || null;
    this._pending = false;
    if (!this._proposal) {
      this.hide({ keepOverlay: false });
      return;
    }
    if (this._applyBtn) this._applyBtn.textContent = 'Apply Cleanup';
    if (this._skipBtn) this._skipBtn.textContent = 'Skip Cleanup';
    if (this._panel) this._panel.classList.remove('hidden');
    this._setButtonsEnabled(true);
    this._renderProposal();
    await this._preview.loadStageResult(6, { file: 'textured_mesh.obj' });
    await this._syncOverlay();
    this._stageCtrl.activateStage(6);
  }

  async showProposal(proposal) {
    await this.showReview(proposal);
  }

  async syncFromStatus(statusMsg) {
    const isInteractive = Number(statusMsg?.current_stage) === 6
      && statusMsg?.running === true
      && statusMsg?.stages?.['6']?.status === 'interactive'
      && Boolean(statusMsg?.cleanup_proposal_path);
    if (!isInteractive) {
      if (statusMsg?.stages?.['6']?.status === 'complete') {
        this.hide({ keepOverlay: false });
      }
      return;
    }
    const proposal = await this.loadProposal();
    await this.showReview(proposal);
  }

  markCompleted() {
    this._pending = false;
    this._preview.clearStageOverlay(6);
    if (this._panel) this._panel.classList.remove('hidden');
    if (this._summary) {
      this._summary.textContent = 'Cleanup stage completed.';
    }
    this._setButtonsEnabled(false);
  }

  async confirm(decision) {
    if (!this._proposal || this._pending) return;
    this._pending = true;
    this._setButtonsEnabled(false);
    try {
      const res = await fetch('/api/post-texture-cleanup/confirm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ decision }),
      });
      const data = await res.json();
      if (!res.ok || data?.error) {
        throw new Error(data?.error || `HTTP ${res.status}`);
      }
      this._appendLog('stdout', `Post-texture cleanup decision: ${decision}\n`, { stage: 6 });
      if (this._summary) {
        this._summary.textContent = `Decision submitted: ${decision}. Waiting for stage completion.`;
      }
    } catch (e) {
      this._pending = false;
      this._setButtonsEnabled(true);
      this._appendLog('stderr', `Failed to submit cleanup decision: ${e.message}\n`, { stage: 6 });
    }
  }

  async _syncOverlay() {
    const overlayPath = this._proposal?.artifacts?.removed_region_ply;
    const showOverlay = this._overlayToggle?.checked !== false;
    if (!overlayPath || !showOverlay) {
      this._preview.clearStageOverlay(6);
      return;
    }
    await this._preview.loadStageOverlay(6, overlayPath, {
      color: 0xff5533,
      opacity: 0.68,
    });
  }

  _renderProposal() {
    const summaryInfo = this._proposal?.summary || {};
    const maskInfo = this._proposal?.mask_consistency || {};
    const planeInfo = this._proposal?.ground_plane || {};
    const summary = [];
    if (Number(summaryInfo.removed_face_count) > 0) {
      summary.push(`removed faces: ${Number(summaryInfo.removed_face_count)}`);
    }
    if (Number(summaryInfo.cap_face_count) > 0) {
      summary.push(`cap faces: ${Number(summaryInfo.cap_face_count)}`);
    }
    if (Number.isFinite(Number(summaryInfo.confidence))) {
      summary.push(`confidence: ${(Number(summaryInfo.confidence) * 100).toFixed(1)}%`);
    }
    if (this._summary) {
      this._summary.textContent = summary.length
        ? `Candidate overlay is shown in red. ${summary.join(' | ')}`
        : 'Candidate overlay is shown in red on top of the textured mesh.';
    }

    const statLines = [
      `Recommended decision: ${this._proposal?.recommended_decision || 'skip'}`,
      `Clip shift: ${Number(planeInfo.selected_shift || 0).toFixed(6)}`,
      `Section area: ${Number(planeInfo.selected_loop_area || 0).toFixed(6)}`,
    ];
    if (Number(maskInfo.visible_samples) > 0) {
      statLines.push(`visible samples: ${Number(maskInfo.visible_samples)}`);
    }
    if (Number.isFinite(Number(maskInfo.object_outside_ratio))) {
      statLines.push(`outside object: ${(Number(maskInfo.object_outside_ratio) * 100).toFixed(1)}%`);
    }
    if (Number.isFinite(Number(maskInfo.ground_overlap_ratio))) {
      statLines.push(`ground overlap: ${(Number(maskInfo.ground_overlap_ratio) * 100).toFixed(1)}%`);
    }
    if (this._stats) {
      this._stats.textContent = statLines.join(' | ');
    }
  }

  _setButtonsEnabled(enabled) {
    if (this._applyBtn) this._applyBtn.disabled = !enabled;
    if (this._skipBtn) this._skipBtn.disabled = !enabled;
    if (this._overlayToggle) this._overlayToggle.disabled = this._proposal == null;
  }
}
