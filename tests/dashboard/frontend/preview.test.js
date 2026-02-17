import { describe, it, expect, beforeEach } from 'vitest';
import { buildPreviewDOM } from './helpers/dom-factory.js';
import { PreviewPanel } from '../../../scripts/dashboard/static/js/preview.js';

describe('PreviewPanel', () => {
  let preview;

  beforeEach(() => {
    buildPreviewDOM();
    preview = new PreviewPanel();
  });

  // ── constructor state ───────────────────────────────────────

  describe('constructor', () => {
    it('_stages is empty object', () => {
      expect(preview._stages).toEqual({});
    });

    it('_renderer is null', () => {
      expect(preview._renderer).toBe(null);
    });

    it('_currentTheme is dark', () => {
      expect(preview._currentTheme).toBe('dark');
    });

    it('_threeLoaded is false', () => {
      expect(preview._threeLoaded).toBe(false);
    });

    it('_animating is false', () => {
      expect(preview._animating).toBe(false);
    });

    it('_previewAssetRevision is 0', () => {
      expect(preview._previewAssetRevision).toBe(0);
    });
  });

  // ── _previewAssetRevision ───────────────────────────────────

  describe('_previewAssetRevision', () => {
    it('can be incremented', () => {
      expect(preview._previewAssetRevision).toBe(0);
      preview._previewAssetRevision += 1;
      expect(preview._previewAssetRevision).toBe(1);
    });
  });

  // ── _meshRepair ─────────────────────────────────────────────

  describe('_meshRepair', () => {
    it('initial state has active=false', () => {
      expect(preview._meshRepair.active).toBe(false);
    });
  });
});
