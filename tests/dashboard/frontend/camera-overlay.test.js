import { describe, it, expect, vi, beforeEach } from 'vitest';
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

  // ── create ─────────────────────────────────────────────────────────

  describe('create', () => {
    it('23.5 — creates frustum group', () => {
      // Provide THREE stub
      const THREE = {
        Group: class { constructor() { this.name = ''; this.add = () => {}; this.position = { set() {} }; this.visible = true; } },
        Matrix4: class { fromArray() {} decompose(p, q, s) { p.x=0; p.y=0; p.z=0; } },
        Vector3: class { constructor() { this.x=0; this.y=0; this.z=0; } copy() { return this; } },
        Quaternion: class { copy() { return this; } },
        BufferGeometry: class { setAttribute() {} setIndex() {} dispose() {} },
        BufferAttribute: class { constructor() {} },
        Color: class { setHSL() { return this; } },
        LineBasicMaterial: class { dispose() {} },
        LineSegments: class {
          constructor() { this.position = { copy() { return this; } }; this.quaternion = { copy() { return this; } }; this.geometry = { dispose() {} }; this.material = { dispose() {} }; }
        },
      };

      const parent = { add: vi.fn(), remove: vi.fn() };
      const poses = [
        { matrix: [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1], frame_index: 0 },
      ];

      overlay.create(THREE, parent, poses);

      expect(overlay._group).not.toBe(null);
      expect(parent.add).toHaveBeenCalled();
    });
  });

  // ── setVisible after create ────────────────────────────────────────

  describe('setVisible after create', () => {
    it('23.6 — setVisible(false) hides group', () => {
      // Create a minimal group
      overlay._group = { visible: true, position: { set() {} } };
      overlay._parent = {};

      overlay.setVisible(false);

      expect(overlay._group.visible).toBe(false);
    });
  });
});
