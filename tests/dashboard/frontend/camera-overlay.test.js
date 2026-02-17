import { describe, it, expect, beforeEach } from 'vitest';
import { CameraOverlay } from '../../../scripts/dashboard/static/js/camera-overlay.js';

describe('CameraOverlay', () => {
  let overlay;

  beforeEach(() => {
    overlay = new CameraOverlay();
  });

  // ── constructor ─────────────────────────────────────────────

  describe('constructor', () => {
    it('_group is null, _parent is null', () => {
      expect(overlay._group).toBe(null);
      expect(overlay._parent).toBe(null);
    });
  });

  // ── remove ──────────────────────────────────────────────────

  describe('remove', () => {
    it('null group causes no exception', () => {
      expect(() => overlay.remove()).not.toThrow();
    });
  });

  // ── setVisible ──────────────────────────────────────────────

  describe('setVisible', () => {
    it('null group causes no exception', () => {
      expect(() => overlay.setVisible(true)).not.toThrow();
      expect(() => overlay.setVisible(false)).not.toThrow();
    });
  });

  // ── applyOffset ─────────────────────────────────────────────

  describe('applyOffset', () => {
    it('null group causes no exception', () => {
      expect(() => overlay.applyOffset({ x: 1, y: 2, z: 3 })).not.toThrow();
    });
  });
});
