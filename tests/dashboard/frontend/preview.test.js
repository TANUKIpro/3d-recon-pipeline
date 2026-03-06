import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
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

  // ── cache busting for object switch ─────────────────────────

  describe('cache busting', () => {
    afterEach(() => {
      vi.restoreAllMocks();
    });

    it('loadPi3xResults uses same rev token for PLY and camera_poses fetch', async () => {
      document.body.innerHTML = `
        <div id="gallery-grid"></div>
        <div id="stage-2-empty" class="hidden"></div>
        <div id="pi3x-toolbar" style="display:none"></div>
        <div id="pi3x-point-count"></div>
        <div id="pi3x-camera-count"></div>
      `;
      preview = new PreviewPanel();

      preview.initSceneForStage = vi.fn().mockResolvedValue(undefined);
      preview.activateStage = vi.fn();

      const container = document.createElement('div');
      preview._stages[2] = {
        centerOffset: null,
        currentObject: null,
        sceneRoot: { rotation: { x: 0 } },
        container,
      };

      const loadPlySpy = vi.spyOn(preview, '_loadPLYIntoStage').mockResolvedValue(false);
      const fetchMock = vi.fn().mockResolvedValue({
        ok: false,
        status: 404,
        json: vi.fn().mockResolvedValue({}),
      });
      global.fetch = fetchMock;

      const cameraOverlay = {
        remove: vi.fn(),
        create: vi.fn(),
        applyOffset: vi.fn(),
        setVisible: vi.fn(),
      };

      await preview.loadPi3xResults(cameraOverlay, 'object_full.ply');

      expect(loadPlySpy).toHaveBeenCalledWith(
        2,
        'object_full.ply',
        expect.objectContaining({ cacheToken: 1 }),
      );
      expect(fetchMock).toHaveBeenCalledWith('/api/preview/file/camera_poses.json?rev=1');
    });

    it('_resolveSceneFlipFromCameraPoses uses cache token in URL', async () => {
      const fetchMock = vi.fn().mockResolvedValue({
        ok: false,
        status: 404,
        json: vi.fn().mockResolvedValue({}),
      });
      global.fetch = fetchMock;

      const flip = await preview._resolveSceneFlipFromCameraPoses(99);

      expect(flip).toBe(false);
      expect(fetchMock).toHaveBeenCalledWith('/api/preview/file/camera_poses.json?rev=99');
    });
  });
});
