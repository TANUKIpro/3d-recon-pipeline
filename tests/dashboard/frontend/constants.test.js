import { describe, it, expect } from 'vitest';
import {
  STAGE_COUNT,
  TRANSITION_STAGE_MAX,
  MESH_METHOD_DEFAULT,
  MESH_METHOD_SET,
  MESH_REPAIR_THRESHOLD_MIN,
  MESH_REPAIR_THRESHOLD_MAX,
  MESH_REPAIR_THRESHOLD_DEFAULT,
  DEFAULT_TAUBIN_NU,
  CLASSICAL_PREVIEW_TITLE,
  STORAGE_KEYS,
} from '../../../scripts/dashboard/static/js/constants.js';

describe('constants', () => {
  it('STAGE_COUNT is 8', () => {
    expect(STAGE_COUNT).toBe(8);
  });

  it('TRANSITION_STAGE_MAX is 7', () => {
    expect(TRANSITION_STAGE_MAX).toBe(7);
  });

  it('MESH_METHOD_DEFAULT is poisson', () => {
    expect(MESH_METHOD_DEFAULT).toBe('poisson');
  });

  it('MESH_METHOD_SET contains diffcd and poisson', () => {
    expect(MESH_METHOD_SET.has('diffcd')).toBe(true);
    expect(MESH_METHOD_SET.has('poisson')).toBe(true);
    expect(MESH_METHOD_SET.size).toBe(2);
  });

  it('MESH_REPAIR_THRESHOLD bounds are valid', () => {
    expect(MESH_REPAIR_THRESHOLD_MIN).toBe(-1.0);
    expect(MESH_REPAIR_THRESHOLD_MAX).toBe(0.0);
    expect(MESH_REPAIR_THRESHOLD_DEFAULT).toBe(-0.25);
    expect(MESH_REPAIR_THRESHOLD_DEFAULT).toBeGreaterThanOrEqual(MESH_REPAIR_THRESHOLD_MIN);
    expect(MESH_REPAIR_THRESHOLD_DEFAULT).toBeLessThanOrEqual(MESH_REPAIR_THRESHOLD_MAX);
  });

  it('DEFAULT_TAUBIN_NU is negative', () => {
    expect(DEFAULT_TAUBIN_NU).toBe(-0.53);
  });

  it('CLASSICAL_PREVIEW_TITLE is a non-empty string', () => {
    expect(typeof CLASSICAL_PREVIEW_TITLE).toBe('string');
    expect(CLASSICAL_PREVIEW_TITLE.length).toBeGreaterThan(0);
  });

  it('STORAGE_KEYS all have clip2mesh prefix', () => {
    for (const [key, value] of Object.entries(STORAGE_KEYS)) {
      expect(value).toMatch(/^clip2mesh:/);
    }
  });
});
