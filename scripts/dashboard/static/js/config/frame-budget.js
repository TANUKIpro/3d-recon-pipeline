/**
 * Pi3X frame budget calculation and UI sync.
 *
 * All exported functions are mixin methods — they use `this` to refer to
 * the ConfigPanel instance.
 */

export function _onFrameIntervalInput() {
  if (this._videoMeta) {
    const interval = this._parsePositiveInt(
      this._inputs.frame_interval.value,
      this._extractDefaults.frame_interval,
    );
    this._extractDefaults.frame_interval = interval;
    if (this._maxFramesAuto) {
      const maxFrames = this._estimateMaxFrames(this._videoMeta.total_frames, interval);
      this._extractDefaults.max_frames = maxFrames;
      this._inputs.max_frames.value = String(maxFrames);
    }
  }
  this._updateFrameBudgetPreview();
}

export function _onMaxFramesInput() {
  const raw = (this._inputs.max_frames.value || '').trim();
  if (!raw) {
    this._maxFramesAuto = true;
    this._onFrameIntervalInput();
    return;
  }
  const parsed = Number.parseInt(raw, 10);
  if (Number.isFinite(parsed) && parsed > 0) {
    this._maxFramesAuto = false;
  }
  this._updateFrameBudgetPreview();
}

export function _onPi3xFrameTargetInput() {
  this._pi3xFrameTargetAuto = false;
  this._syncPi3xFrameTargetInput();
}

export function _estimateMaxFrames(totalFrames, frameInterval) {
  const total = this._parsePositiveInt(totalFrames, 0);
  const interval = this._parsePositiveInt(frameInterval, 1);
  if (total <= 0) return this._extractDefaults.max_frames;
  return Math.max(2, Math.ceil(total / interval));
}

export function _resolveMaxFrames() {
  return Math.max(
    2,
    this._parsePositiveInt(
      this._inputs.max_frames.value,
      this._extractDefaults.max_frames,
    ),
  );
}

export function _clampPi3xFrameTarget(targetFrames, maxFrames) {
  const maxAllowed = Math.max(2, this._parsePositiveInt(maxFrames, this._extractDefaults.max_frames));
  const parsed = this._parsePositiveInt(targetFrames, maxAllowed);
  return Math.max(2, Math.min(parsed, maxAllowed));
}

export function _resolvePi3xFrameTarget(maxFrames = this._resolveMaxFrames()) {
  const fallback = this._clampPi3xFrameTarget(
    this._pi3xFrameTargetRecommended ?? maxFrames,
    maxFrames,
  );
  const parsed = this._parsePositiveInt(this._inputs.pi3x_frame_target.value, fallback);
  return this._clampPi3xFrameTarget(parsed, maxFrames);
}

export function _syncPi3xFrameTargetInput(opts = {}) {
  const maxFrames = this._resolveMaxFrames();
  const recommendedRaw = opts.recommendedFrames;
  if (recommendedRaw != null) {
    this._pi3xFrameTargetRecommended = this._clampPi3xFrameTarget(recommendedRaw, maxFrames);
  } else if (this._pi3xFrameTargetRecommended == null) {
    this._pi3xFrameTargetRecommended = maxFrames;
  } else {
    this._pi3xFrameTargetRecommended = this._clampPi3xFrameTarget(this._pi3xFrameTargetRecommended, maxFrames);
  }

  this._inputs.pi3x_frame_target.min = '2';
  this._inputs.pi3x_frame_target.max = String(maxFrames);

  const fallback = this._pi3xFrameTargetRecommended;
  let selectedFrames = this._resolvePi3xFrameTarget(maxFrames);
  if (this._pi3xFrameTargetAuto || !(this._inputs.pi3x_frame_target.value || '').trim()) {
    selectedFrames = maxFrames;
    this._inputs.pi3x_frame_target.value = String(selectedFrames);
  } else {
    this._inputs.pi3x_frame_target.value = String(selectedFrames);
  }

  if (this._pi3xFrameTargetValue) {
    this._pi3xFrameTargetValue.textContent = String(selectedFrames);
  }
  if (this._pi3xFrameTargetNote) {
    this._pi3xFrameTargetNote.textContent =
      `AutoTarget recommendation: ${fallback} frames (max ${maxFrames})`;
  }

  this._updatePi3xFrameTargetMarker();

  return {
    requestedFrames: maxFrames,
    selectedFrames,
    recommendedFrames: fallback,
  };
}

export function _updatePi3xFrameTargetMarker() {
  if (!this._pi3xFrameTargetMarker) return;
  const recommended = this._pi3xFrameTargetRecommended;
  const input = this._inputs.pi3x_frame_target;
  const min = parseFloat(input.min) || 2;
  const max = parseFloat(input.max) || 50;
  if (recommended == null || max <= min || recommended >= max) {
    this._pi3xFrameTargetMarker.style.display = 'none';
    return;
  }
  const pct = ((recommended - min) / (max - min)) * 100;
  this._pi3xFrameTargetMarker.style.display = '';
  this._pi3xFrameTargetMarker.style.left = `${pct}%`;
}

export function _resolveRequestedPi3xFrames() {
  return this._resolveMaxFrames();
}

export function _updateFrameBudgetPreview() {
  const requestedFrames = this._resolveRequestedPi3xFrames();
  this._syncPi3xFrameTargetInput();
  if (!this._pi3xPlanNote) return;
  const pixelLimit = this._parsePositiveInt(this._inputs.pixel_limit.value, 255000);
  this._pi3xPlanNote.textContent = 'Pi3X VRAM plan: estimating...';
  if (this._pi3xPlanDebounce) {
    clearTimeout(this._pi3xPlanDebounce);
  }
  this._pi3xPlanDebounce = setTimeout(() => {
    this._updatePi3xPlanPreview(requestedFrames, pixelLimit);
  }, 120);
}

export async function _updatePi3xPlanPreview(requestedFrames, pixelLimit) {
  const reqId = ++this._pi3xPlanRequestId;
  try {
    const res = await fetch(
      `/api/pipeline/pi3x-plan?requested_frames=${requestedFrames}&pixel_limit=${pixelLimit}`,
    );
    if (reqId !== this._pi3xPlanRequestId) return;
    if (!res.ok) {
      this._syncPi3xFrameTargetInput({ recommendedFrames: requestedFrames });
      this._pi3xPlanNote.textContent = 'Pi3X VRAM plan: unavailable';
      return;
    }
    const data = await res.json();
    const autoFrames = this._parsePositiveInt(data.auto_target_frames, requestedFrames);
    const synced = this._syncPi3xFrameTargetInput({ recommendedFrames: autoFrames });
    const targetPct = Number(data.target_vram_utilization) * 100;
    const usedPct = Number(data.predicted_used_pct);
    const parts = [`Auto target: ${synced.recommendedFrames}/${synced.requestedFrames} frames`];
    if (synced.selectedFrames !== synced.recommendedFrames) {
      parts.push(`selected ${synced.selectedFrames}`);
    }
    if (Number.isFinite(targetPct) && targetPct > 0) {
      parts.push(`target VRAM ${targetPct.toFixed(0)}%`);
    }
    if (Number.isFinite(usedPct) && usedPct > 0) {
      parts.push(`estimated ${usedPct.toFixed(1)}%`);
    }
    if (data.auto_reduced) {
      parts.push('auto-reduced');
    }
    this._pi3xPlanNote.textContent = parts.join(' | ');
  } catch {
    if (reqId !== this._pi3xPlanRequestId) return;
    this._syncPi3xFrameTargetInput({ recommendedFrames: requestedFrames });
    this._pi3xPlanNote.textContent = 'Pi3X VRAM plan: unavailable';
  }
}
