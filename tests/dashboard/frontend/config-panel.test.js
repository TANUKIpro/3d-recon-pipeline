import { describe, it, expect, vi, beforeEach } from 'vitest';
import { buildConfigPanelDOM } from './helpers/dom-factory.js';
import { useFetchMock } from './helpers/fetch-mock.js';
import { ConfigPanel } from '../../../scripts/dashboard/static/js/config-panel.js';

describe('ConfigPanel', () => {
  let panel;
  const fetchMock = useFetchMock();

  beforeEach(async () => {
    buildConfigPanelDOM();

    // Install fetch mock BEFORE constructing ConfigPanel because the
    // constructor calls _loadVideos(), refreshObjects(), and
    // _updateFrameBudgetPreview() which all make fetch requests.
    fetchMock.installFetch();
    fetchMock.addRoute('/api/pipeline/videos', fetchMock.jsonResponse({ videos: [] }));
    fetchMock.addRoute('/api/pipeline/objects', fetchMock.jsonResponse({ objects: [] }));
    fetchMock.addRoute('/api/pipeline/pi3x-plan', fetchMock.jsonResponse({ auto_target_frames: 50 }));

    panel = new ConfigPanel();

    // Wait for async constructor calls (_loadVideos, refreshObjects) to settle.
    await new Promise(r => setTimeout(r, 0));
  });

  // ── _normalizeObjectName ────────────────────────────────────

  describe('_normalizeObjectName', () => {
    const normalize = (v) => ConfigPanel.prototype._normalizeObjectName.call(panel, v);

    it('simple ASCII name passes through', () => {
      expect(normalize('my-object')).toBe('my-object');
    });

    it('slashes become hyphens', () => {
      expect(normalize('path/to/object')).toBe('path-to-object');
    });

    it('spaces become hyphens', () => {
      expect(normalize('my cool object')).toBe('my-cool-object');
    });

    it('special characters removed', () => {
      expect(normalize('obj@#$ect!')).toBe('obj-ect');
    });

    it('consecutive hyphens compressed', () => {
      expect(normalize('a---b')).toBe('a-b');
    });

    it('leading dots/hyphens removed', () => {
      expect(normalize('..--name')).toBe('name');
    });

    it('trailing dots/hyphens removed', () => {
      expect(normalize('name..-')).toBe('name');
    });

    it('truncated to 80 chars', () => {
      const long = 'a'.repeat(100);
      expect(normalize(long).length).toBe(80);
    });

    it('empty string returns empty', () => {
      expect(normalize('')).toBe('');
    });

    it('Japanese characters preserved', () => {
      expect(normalize('テスト')).toBe('テスト');
    });

    it('Korean characters preserved', () => {
      expect(normalize('테스트')).toBe('테스트');
    });

    it('mixed ASCII + Japanese', () => {
      expect(normalize('test-テスト')).toBe('test-テスト');
    });
  });

  // ── _parsePositiveInt ───────────────────────────────────────

  describe('_parsePositiveInt', () => {
    const parse = (v, fb) => ConfigPanel.prototype._parsePositiveInt.call(panel, v, fb);

    it('valid positive returns value', () => {
      expect(parse('42', 10)).toBe(42);
    });

    it('zero returns fallback', () => {
      expect(parse('0', 10)).toBe(10);
    });

    it('negative returns fallback', () => {
      expect(parse('-5', 10)).toBe(10);
    });

    it('NaN string returns fallback', () => {
      expect(parse('abc', 10)).toBe(10);
    });
  });

  // ── _parsePositiveFloat ─────────────────────────────────────

  describe('_parsePositiveFloat', () => {
    const parse = (v, fb) => ConfigPanel.prototype._parsePositiveFloat.call(panel, v, fb);

    it('valid positive returns value', () => {
      expect(parse('3.14', 1.0)).toBe(3.14);
    });

    it('zero returns fallback', () => {
      expect(parse('0', 1.0)).toBe(1.0);
    });

    it('negative returns fallback', () => {
      expect(parse('-2.5', 1.0)).toBe(1.0);
    });
  });

  // ── _parseNonNegativeFloat ──────────────────────────────────

  describe('_parseNonNegativeFloat', () => {
    const parse = (v, fb) => ConfigPanel.prototype._parseNonNegativeFloat.call(panel, v, fb);

    it('zero accepted (returns 0)', () => {
      expect(parse('0', 1.0)).toBe(0);
    });

    it('positive returns value', () => {
      expect(parse('2.5', 1.0)).toBe(2.5);
    });

    it('negative returns fallback', () => {
      expect(parse('-1.0', 1.0)).toBe(1.0);
    });
  });

  // ── _valuesAlmostEqual ──────────────────────────────────────

  describe('_valuesAlmostEqual', () => {
    const eq = (a, b) => ConfigPanel.prototype._valuesAlmostEqual.call(panel, a, b);

    it('identical values return true', () => {
      expect(eq(5.0, 5.0)).toBe(true);
    });

    it('difference within 1e-9 returns true', () => {
      expect(eq(1.0, 1.0 + 1e-10)).toBe(true);
    });

    it('difference > 1e-9 returns false', () => {
      expect(eq(1.0, 1.0 + 1e-8)).toBe(false);
    });

    it('non-finite returns false', () => {
      expect(eq(NaN, NaN)).toBe(false);
      expect(eq(Infinity, Infinity)).toBe(false);
    });
  });

  // ── setMeshMethod ───────────────────────────────────────────

  describe('setMeshMethod', () => {
    it('poisson: shows classical controls, hides diffcd', () => {
      panel.setMeshMethod('poisson');
      const poissonSummary = document.getElementById('cfg-poisson-summary');
      const diffcdControls = document.getElementById('cfg-diffcd-controls');
      const classicalControls = document.getElementById('cfg-classical-controls');
      expect(poissonSummary.style.display).toBe('');
      expect(diffcdControls.style.display).toBe('none');
      expect(classicalControls.style.display).toBe('');
    });

    it('diffcd: shows diffcd controls, hides classical', () => {
      panel.setMeshMethod('diffcd');
      const poissonSummary = document.getElementById('cfg-poisson-summary');
      const diffcdControls = document.getElementById('cfg-diffcd-controls');
      const classicalControls = document.getElementById('cfg-classical-controls');
      expect(poissonSummary.style.display).toBe('none');
      expect(diffcdControls.style.display).toBe('');
      expect(classicalControls.style.display).toBe('none');
    });
  });

  // ── getConfig ───────────────────────────────────────────────

  describe('getConfig', () => {
    it('returns object with video_path key', () => {
      const config = panel.getConfig();
      expect(config).toHaveProperty('video_path');
    });

    it('returns object with object_name key', () => {
      const config = panel.getConfig();
      expect(config).toHaveProperty('object_name');
    });

    it('all expected keys present', () => {
      const config = panel.getConfig();
      const expectedKeys = [
        'video_path', 'object_name', 'resume_from_stage',
        'frame_interval', 'max_frames', 'pixel_limit',
        'pi3x_frame_target', 'confidence_threshold', 'edge_rtol',
        'sam2_model', 'denoise_preset', 'denoise_algorithm',
        'mesh_method', 'diffcd_batch_size', 'diffcd_n_batches',
        'diffcd_resolution', 'texture_size', 'texture_view_assign_mode',
        'meshwrap_poisson_depth', 'meshwrap_poisson_scale',
        'meshwrap_iterations', 'meshwrap_crop_scale',
        'classical_preset', 'classical_poisson_depth',
        'mesh_repair_enabled',
      ];
      for (const key of expectedKeys) {
        expect(config).toHaveProperty(key);
      }
    });

    it('empty inputs produce default values', () => {
      // Clear all number inputs to empty
      const inputs = document.querySelectorAll('#config-panel input[type="number"]');
      inputs.forEach(inp => { inp.value = ''; });

      const config = panel.getConfig();
      // object_name should fall back to 'object' when input is empty
      expect(config.object_name).toBe('object');
      // Numeric fields should have fallback defaults (positive numbers)
      expect(config.frame_interval).toBeGreaterThan(0);
      expect(config.max_frames).toBeGreaterThanOrEqual(2);
      expect(config.texture_view_assign_mode).toBe('legacy');
    });

    it('reads the selected texture view assignment mode', () => {
      document.getElementById('cfg-texture-view-assign-mode').value = 'region_gc';
      const config = panel.getConfig();
      expect(config.texture_view_assign_mode).toBe('region_gc');
    });
  });
});
