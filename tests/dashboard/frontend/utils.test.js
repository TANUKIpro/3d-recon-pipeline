import { describe, it, expect } from 'vitest';
import {
  formatTime,
  normalizeMeshMethod,
  clampMeshRepairThreshold,
  formatMeshRepairThreshold,
  parsePositiveInt,
  parseNonNegativeInt,
  parsePositiveFloat,
  parseNonNegativeFloat,
} from '../../../scripts/dashboard/static/js/utils.js';

describe('formatTime', () => {
  it('returns empty string for null', () => {
    expect(formatTime(null)).toBe('');
  });

  it('returns empty string for undefined', () => {
    expect(formatTime(undefined)).toBe('');
  });

  it('returns "0s" for 0', () => {
    expect(formatTime(0)).toBe('0s');
  });

  it('returns "45s" for 45', () => {
    expect(formatTime(45)).toBe('45s');
  });

  it('returns "1m0s" for 60', () => {
    expect(formatTime(60)).toBe('1m0s');
  });

  it('returns "2m5s" for 125', () => {
    expect(formatTime(125)).toBe('2m5s');
  });
});

describe('normalizeMeshMethod', () => {
  it('returns poisson for empty', () => {
    expect(normalizeMeshMethod('')).toBe('poisson');
  });

  it('returns poisson for null', () => {
    expect(normalizeMeshMethod(null)).toBe('poisson');
  });

  it('returns poisson for unknown value', () => {
    expect(normalizeMeshMethod('unknown')).toBe('poisson');
  });

  it('returns diffcd for "diffcd"', () => {
    expect(normalizeMeshMethod('diffcd')).toBe('diffcd');
  });

  it('returns poisson for "poisson"', () => {
    expect(normalizeMeshMethod('poisson')).toBe('poisson');
  });

  it('is case insensitive', () => {
    expect(normalizeMeshMethod('DIFFCD')).toBe('diffcd');
    expect(normalizeMeshMethod('Poisson')).toBe('poisson');
  });

  it('trims whitespace', () => {
    expect(normalizeMeshMethod('  diffcd  ')).toBe('diffcd');
  });
});

describe('clampMeshRepairThreshold', () => {
  it('returns default for NaN', () => {
    expect(clampMeshRepairThreshold('abc')).toBe(-0.25);
  });

  it('returns default for Infinity', () => {
    expect(clampMeshRepairThreshold(Infinity)).toBe(-0.25);
  });

  it('clamps below min to min', () => {
    expect(clampMeshRepairThreshold(-2.0)).toBe(-1.0);
  });

  it('clamps above max to max', () => {
    expect(clampMeshRepairThreshold(1.0)).toBe(0.0);
  });

  it('returns value within range as-is', () => {
    expect(clampMeshRepairThreshold(-0.5)).toBe(-0.5);
  });

  it('handles boundary values', () => {
    expect(clampMeshRepairThreshold(-1.0)).toBe(-1.0);
    expect(clampMeshRepairThreshold(0.0)).toBe(0.0);
  });
});

describe('formatMeshRepairThreshold', () => {
  it('formats to 2 decimal places', () => {
    expect(formatMeshRepairThreshold(-0.5)).toBe('-0.50');
  });

  it('formats clamped value', () => {
    expect(formatMeshRepairThreshold(-2.0)).toBe('-1.00');
  });

  it('formats default for NaN', () => {
    expect(formatMeshRepairThreshold('abc')).toBe('-0.25');
  });
});

describe('parsePositiveInt', () => {
  it('parses valid positive int', () => {
    expect(parsePositiveInt('42', 10)).toBe(42);
  });

  it('returns fallback for zero', () => {
    expect(parsePositiveInt('0', 10)).toBe(10);
  });

  it('returns fallback for negative', () => {
    expect(parsePositiveInt('-5', 10)).toBe(10);
  });

  it('returns fallback for non-numeric', () => {
    expect(parsePositiveInt('abc', 10)).toBe(10);
  });

  it('truncates floats', () => {
    expect(parsePositiveInt('3.9', 10)).toBe(3);
  });
});

describe('parseNonNegativeInt', () => {
  it('parses valid non-negative int', () => {
    expect(parseNonNegativeInt('42', 10)).toBe(42);
  });

  it('accepts zero', () => {
    expect(parseNonNegativeInt('0', 10)).toBe(0);
  });

  it('returns fallback for negative', () => {
    expect(parseNonNegativeInt('-5', 10)).toBe(10);
  });

  it('returns fallback for non-numeric', () => {
    expect(parseNonNegativeInt('abc', 10)).toBe(10);
  });
});

describe('parsePositiveFloat', () => {
  it('parses valid positive float', () => {
    expect(parsePositiveFloat('3.14', 1.0)).toBeCloseTo(3.14);
  });

  it('returns fallback for zero', () => {
    expect(parsePositiveFloat('0', 1.0)).toBe(1.0);
  });

  it('returns fallback for negative', () => {
    expect(parsePositiveFloat('-2.5', 1.0)).toBe(1.0);
  });

  it('returns fallback for non-numeric', () => {
    expect(parsePositiveFloat('abc', 1.0)).toBe(1.0);
  });
});

describe('parseNonNegativeFloat', () => {
  it('parses valid non-negative float', () => {
    expect(parseNonNegativeFloat('3.14', 1.0)).toBeCloseTo(3.14);
  });

  it('accepts zero', () => {
    expect(parseNonNegativeFloat('0', 1.0)).toBe(0);
  });

  it('returns fallback for negative', () => {
    expect(parseNonNegativeFloat('-2.5', 1.0)).toBe(1.0);
  });

  it('returns fallback for non-numeric', () => {
    expect(parseNonNegativeFloat('abc', 1.0)).toBe(1.0);
  });
});
