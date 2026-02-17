import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { buildSAM2VerificationDOM } from './helpers/dom-factory.js';
import { useFetchMock } from './helpers/fetch-mock.js';
import { SAM2Verification } from '../../../scripts/dashboard/static/js/sam2-verification.js';

const { installFetch, restoreFetch, addRoute, clearRoutes, jsonResponse } = useFetchMock();

beforeEach(() => {
  installFetch();
  clearRoutes();
});

afterEach(() => {
  restoreFetch();
});

describe('SAM2Verification', () => {
  let verification;

  beforeEach(() => {
    buildSAM2VerificationDOM();
    verification = new SAM2Verification();
  });

  // --- _pickIndices ---

  describe('_pickIndices', () => {
    it('total=0 returns empty array', () => {
      const result = SAM2Verification.prototype._pickIndices.call(verification, 0, 10);
      expect(result).toEqual([]);
    });

    it('total=5, count=10 returns all indices [0..4]', () => {
      const result = SAM2Verification.prototype._pickIndices.call(verification, 5, 10);
      expect(result).toEqual([0, 1, 2, 3, 4]);
    });

    it('total=100, count=10 returns 10 evenly-spaced indices', () => {
      const result = SAM2Verification.prototype._pickIndices.call(verification, 100, 10);
      expect(result).toHaveLength(10);
      expect(result[0]).toBe(0);
      expect(result[9]).toBe(99);
      // All indices should be unique and sorted ascending
      for (let i = 1; i < result.length; i++) {
        expect(result[i]).toBeGreaterThan(result[i - 1]);
      }
    });

    it('total=1 returns [0]', () => {
      const result = SAM2Verification.prototype._pickIndices.call(verification, 1, 10);
      expect(result).toEqual([0]);
    });

    it('total=2, count=10 returns [0, 1]', () => {
      const result = SAM2Verification.prototype._pickIndices.call(verification, 2, 10);
      expect(result).toEqual([0, 1]);
    });
  });

  // --- show ---

  describe('show', () => {
    it('creates 10 frame elements in strip for totalFrames=100', () => {
      verification.show(100);
      const strip = document.getElementById('sam2-verification-strip');
      expect(strip.children.length).toBe(10);
    });

    it('img src contains correct frame index', () => {
      verification.show(100);
      const strip = document.getElementById('sam2-verification-strip');
      const imgs = strip.querySelectorAll('img');
      // First frame index should be 0
      expect(imgs[0].src).toContain('/api/verification/frame/0');
      // Last frame index should be 99
      expect(imgs[9].src).toContain('/api/verification/frame/99');
    });

    it('container becomes visible (display:block)', () => {
      verification.show(50);
      const container = document.getElementById('sam2-verification');
      expect(container.style.display).toBe('block');
    });
  });

  // --- hide ---

  describe('hide', () => {
    it('sets display:none and clears strip', () => {
      verification.show(50);
      verification.hide();
      const container = document.getElementById('sam2-verification');
      const strip = document.getElementById('sam2-verification-strip');
      expect(container.style.display).toBe('none');
      expect(strip.innerHTML).toBe('');
    });
  });

  // --- visible getter ---

  describe('visible getter', () => {
    it('true after show, false after hide', () => {
      expect(verification.visible).toBe(false);
      verification.show(20);
      expect(verification.visible).toBe(true);
      verification.hide();
      expect(verification.visible).toBe(false);
    });
  });

  // --- approve button ---

  describe('approve button', () => {
    it('POSTs to /api/sam2/approve and hides', async () => {
      addRoute('/api/sam2/approve', jsonResponse({ ok: true }));
      verification.show(20);

      const btn = document.getElementById('sam2-approve');
      btn.click();

      // Wait for the async handler to complete
      await vi.waitFor(() => {
        expect(verification.visible).toBe(false);
      });

      expect(globalThis.fetch).toHaveBeenCalledWith(
        '/api/sam2/approve',
        expect.objectContaining({ method: 'POST' }),
      );
    });
  });

  // --- redo button ---

  describe('redo button', () => {
    it('POSTs to /api/sam2/redo and hides', async () => {
      addRoute('/api/sam2/redo', jsonResponse({ ok: true }));
      verification.show(20);

      const btn = document.getElementById('sam2-redo');
      btn.click();

      await vi.waitFor(() => {
        expect(verification.visible).toBe(false);
      });

      expect(globalThis.fetch).toHaveBeenCalledWith(
        '/api/sam2/redo',
        expect.objectContaining({ method: 'POST' }),
      );
    });
  });
});
