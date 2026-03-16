/**
 * Form utility methods: name normalization, select value, float comparison,
 * resume stage logic, texture size helpers.
 *
 * All exported functions are mixin methods — they use `this` to refer to
 * the ConfigPanel instance.
 */

import {
  parsePositiveInt,
  parseNonNegativeInt,
  parsePositiveFloat,
  parseNonNegativeFloat,
} from '../utils.js';
import { STAGE_COUNT } from '../constants.js';
import { NEW_OBJECT_VALUE, STAGE_LABELS } from './presets.js';

export function _normalizeObjectName(value) {
  const raw = String(value || '').trim();
  if (!raw) return '';
  let normalized = raw
    .replace(/[\\/]/g, '-')
    .replace(/\s+/g, '-')
    .replace(/[^\p{Letter}\p{Number}_.-]/gu, '-')
    .replace(/-+/g, '-')
    .replace(/^[.-]+/, '')
    .replace(/[.-]+$/, '');
  if (!normalized) normalized = 'object';
  return normalized.slice(0, 80);
}

export function _formatSize(sizeMb) {
  if (!Number.isFinite(sizeMb)) return '';
  if (sizeMb < 0.01) return '<0.01 MB';
  return `${Number(sizeMb).toFixed(2)} MB`;
}

export function _setSelectValue(select, value) {
  if (!select) return;
  if (!Array.from(select.options || []).some(opt => opt.value === value)) return;
  select.value = value;
}

export function _valuesAlmostEqual(a, b) {
  const aNum = Number(a);
  const bNum = Number(b);
  if (!Number.isFinite(aNum) || !Number.isFinite(bNum)) return false;
  return Math.abs(aNum - bNum) <= 1e-9;
}

export function _parseTextureSize(value, fallback = 0) {
  const n = Number.parseInt(value, 10);
  if (!Number.isFinite(n)) return fallback;
  if (n <= 0) return 0;
  return n;
}

export function _computeAutoTextureSize(width, height) {
  const w = Number.parseInt(width, 10);
  const h = Number.parseInt(height, 10);
  if (!Number.isFinite(w) || !Number.isFinite(h) || w <= 0 || h <= 0) {
    return null;
  }
  return Math.max(1, Math.round(Math.sqrt(w * h)));
}

export function _updateTextureAutoOption(videoMeta) {
  const textureInput = this._inputs.texture_size;
  if (!textureInput) return;
  const autoOption = Array.from(textureInput.options || []).find((opt) => opt.value === '0');
  if (!autoOption) return;
  const autoSize = this._computeAutoTextureSize(videoMeta?.width, videoMeta?.height);
  autoOption.textContent = autoSize == null ? 'Auto' : `Auto (~${autoSize})`;
}

export function _applySuggestedObjectName(force = false) {
  const selectedExisting = this._objectSelect.value && this._objectSelect.value !== NEW_OBJECT_VALUE;
  if (selectedExisting && !force) return;
  if (this._objectNameDirty && !force) return;
  const suggested = this._suggestObjectNameFromVideo();
  if (!suggested) return;
  this._objectNameInput.value = suggested;
  this._objectNameDirty = false;
  this._selectedObjectSummary = null;
  this._renderObjectSummary(null, suggested);
  this._renderArtifacts(null);
  this._updateResumeHint();
}

export function _suggestObjectNameFromVideo() {
  const option = this._videoSelect.selectedOptions?.[0];
  const raw = option?.dataset?.suggestedObjectName || '';
  return this._normalizeObjectName(raw);
}

export function _resolveResumeStage() {
  return this._clampStage(this._startStage);
}

export function _clampStage(stage) {
  const n = Number(stage) || 1;
  return Math.max(1, Math.min(STAGE_COUNT, Math.round(n)));
}

export function _updateResumeHint() {
  if (!this._resumeStageInfo) return;
  const manualStage = this._resolveResumeStage();
  const manualLabel = STAGE_LABELS[manualStage] || `Stage ${manualStage}`;
  this._resumeStageInfo.textContent =
    `Start Pipeline from selected task: Stage ${manualStage} (${manualLabel}).`;
}

export async function _applyObjectVideoPath(path) {
  if (!path) return;
  const target = String(path);
  const matched = Array.from(this._videoSelect.options || []).some((opt) => opt.value === target);
  if (!matched) return;
  if (this._videoSelect.value === target) return;
  this._videoSelect.value = target;
  await this._onVideoChange();
}

// ── Parse wrappers ────────────────────────────────────────────────

export function _parsePositiveInt(value, fallback) {
  return parsePositiveInt(value, fallback);
}

export function _parseNonNegativeInt(value, fallback) {
  return parseNonNegativeInt(value, fallback);
}

export function _parsePositiveFloat(value, fallback) {
  return parsePositiveFloat(value, fallback);
}

export function _parseNonNegativeFloat(value, fallback) {
  return parseNonNegativeFloat(value, fallback);
}
