/**
 * Shared utility functions used across dashboard modules.
 */

export function formatTime(secs) {
  if (secs == null) return '';
  if (secs < 60) return `${Math.round(secs)}s`;
  const m = Math.floor(secs / 60);
  const s = Math.round(secs % 60);
  return `${m}m${s}s`;
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
