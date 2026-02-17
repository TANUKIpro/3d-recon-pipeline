/**
 * Shared utility functions used across dashboard modules.
 */

import {
  MESH_METHOD_DEFAULT,
  MESH_METHOD_SET,
  MESH_REPAIR_THRESHOLD_MIN,
  MESH_REPAIR_THRESHOLD_MAX,
  MESH_REPAIR_THRESHOLD_DEFAULT,
} from './constants.js';

export function formatTime(secs) {
  if (secs == null) return '';
  if (secs < 60) return `${Math.round(secs)}s`;
  const m = Math.floor(secs / 60);
  const s = Math.round(secs % 60);
  return `${m}m${s}s`;
}

export function normalizeMeshMethod(value) {
  const method = String(value || '').trim().toLowerCase();
  return MESH_METHOD_SET.has(method) ? method : MESH_METHOD_DEFAULT;
}

export function clampMeshRepairThreshold(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return MESH_REPAIR_THRESHOLD_DEFAULT;
  return Math.max(MESH_REPAIR_THRESHOLD_MIN, Math.min(MESH_REPAIR_THRESHOLD_MAX, parsed));
}

export function formatMeshRepairThreshold(value) {
  return clampMeshRepairThreshold(value).toFixed(2);
}

export function parsePositiveInt(value, fallback) {
  const n = Number.parseInt(value, 10);
  if (!Number.isFinite(n) || n <= 0) return fallback;
  return n;
}

export function parseNonNegativeInt(value, fallback) {
  const n = Number.parseInt(value, 10);
  if (!Number.isFinite(n) || n < 0) return fallback;
  return n;
}

export function parsePositiveFloat(value, fallback) {
  const n = Number.parseFloat(value);
  if (!Number.isFinite(n) || n <= 0) return fallback;
  return n;
}

export function parseNonNegativeFloat(value, fallback) {
  const n = Number.parseFloat(value);
  if (!Number.isFinite(n) || n < 0) return fallback;
  return n;
}
