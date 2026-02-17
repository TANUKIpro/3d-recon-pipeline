import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { buildSAM2CanvasDOM } from './helpers/dom-factory.js';
import { useFetchMock } from './helpers/fetch-mock.js';
import { SAM2Canvas } from '../../../scripts/dashboard/static/js/sam2-canvas.js';

const { installFetch, restoreFetch, addRoute, clearRoutes, blobResponse } = useFetchMock();

// Mock Image constructor so _loadFrame resolves synchronously-ish
class MockImage {
  constructor() {
    this.naturalWidth = 640;
    this.naturalHeight = 480;
    this.crossOrigin = '';
  }
  set src(val) {
    this._src = val;
    if (this.onload) setTimeout(() => this.onload(), 0);
  }
  get src() {
    return this._src || '';
  }
}

// Mock URL.createObjectURL / revokeObjectURL
if (typeof URL.createObjectURL !== 'function') {
  URL.createObjectURL = vi.fn(() => 'blob:mock-url');
}
if (typeof URL.revokeObjectURL !== 'function') {
  URL.revokeObjectURL = vi.fn();
}

beforeEach(() => {
  installFetch();
  clearRoutes();
  globalThis.Image = MockImage;
  URL.createObjectURL = vi.fn(() => 'blob:mock-url');
  URL.revokeObjectURL = vi.fn();
});

afterEach(() => {
  restoreFetch();
});

function mockCanvasContext() {
  const canvas = document.getElementById('sam2-canvas');
  const mockCtx = {
    drawImage: vi.fn(),
    clearRect: vi.fn(),
  };
  canvas.getContext = vi.fn(() => mockCtx);
  return mockCtx;
}

describe('SAM2Canvas', () => {
  let canvas;

  beforeEach(() => {
    buildSAM2CanvasDOM();
    mockCanvasContext();
    // Add a route for the initial frame load in activate()
    addRoute('/api/sam2/frame', blobResponse('png-data', 'image/png'));
    canvas = new SAM2Canvas();
  });

  // --- activate ---

  describe('activate', () => {
    it('canvas becomes visible (display:block)', () => {
      canvas.activate(30, 640, 480);
      const el = document.getElementById('sam2-canvas');
      expect(el.style.display).toBe('block');
    });

    it('placeholder hidden (display:none)', () => {
      canvas.activate(30, 640, 480);
      const placeholder = document.getElementById('sam2-placeholder');
      expect(placeholder.style.display).toBe('none');
    });

    it('canvas dimensions set to width/height', () => {
      canvas.activate(30, 800, 600);
      const el = document.getElementById('sam2-canvas');
      expect(el.width).toBe(800);
      expect(el.height).toBe(600);
    });

    it('counters reset to 0', () => {
      canvas._positiveCount = 5;
      canvas._negativeCount = 3;
      canvas.activate(30, 640, 480);
      expect(canvas._positiveCount).toBe(0);
      expect(canvas._negativeCount).toBe(0);
    });
  });

  // --- deactivate ---

  describe('deactivate', () => {
    it('active becomes false', () => {
      canvas.activate(30, 640, 480);
      canvas.deactivate();
      expect(canvas._active).toBe(false);
    });

    it('buttons disabled', () => {
      canvas.activate(30, 640, 480);
      canvas.deactivate();
      expect(document.getElementById('sam2-undo').disabled).toBe(true);
      expect(document.getElementById('sam2-clear').disabled).toBe(true);
      expect(document.getElementById('sam2-confirm').disabled).toBe(true);
    });
  });

  // --- _handleClick ---

  describe('_handleClick', () => {
    it('positive label increments positiveCount', async () => {
      canvas.activate(30, 640, 480);
      canvas._loading = false;
      addRoute('/api/sam2/click', blobResponse('png-data', 'image/png'));

      const fakeEvent = {
        clientX: 100,
        clientY: 100,
      };
      // Mock getBoundingClientRect
      const canvasEl = document.getElementById('sam2-canvas');
      canvasEl.getBoundingClientRect = () => ({ left: 0, top: 0, width: 640, height: 480 });

      await canvas._handleClick(fakeEvent, 1);
      expect(canvas._positiveCount).toBe(1);
    });

    it('negative label increments negativeCount', async () => {
      canvas.activate(30, 640, 480);
      canvas._loading = false;
      addRoute('/api/sam2/click', blobResponse('png-data', 'image/png'));

      const fakeEvent = { clientX: 100, clientY: 100 };
      const canvasEl = document.getElementById('sam2-canvas');
      canvasEl.getBoundingClientRect = () => ({ left: 0, top: 0, width: 640, height: 480 });

      await canvas._handleClick(fakeEvent, 0);
      expect(canvas._negativeCount).toBe(1);
    });

    it('POSTs to /api/sam2/click with normalized coords', async () => {
      canvas.activate(30, 640, 480);
      canvas._loading = false;
      addRoute('/api/sam2/click', blobResponse('png-data', 'image/png'));

      const canvasEl = document.getElementById('sam2-canvas');
      canvasEl.getBoundingClientRect = () => ({ left: 0, top: 0, width: 640, height: 480 });

      const fakeEvent = { clientX: 320, clientY: 240 };
      await canvas._handleClick(fakeEvent, 1);

      expect(globalThis.fetch).toHaveBeenCalledWith(
        '/api/sam2/click',
        expect.objectContaining({
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
        }),
      );
      // Parse the body to check normalized coords
      const callArgs = globalThis.fetch.mock.calls.find(c => c[0] === '/api/sam2/click');
      const body = JSON.parse(callArgs[1].body);
      expect(body.x).toBeCloseTo(0.5, 1);
      expect(body.y).toBeCloseTo(0.5, 1);
      expect(body.label).toBe(1);
    });

    it('ignored when not active', async () => {
      canvas._active = false;
      canvas._loading = false;
      const fakeEvent = { clientX: 100, clientY: 100 };

      await canvas._handleClick(fakeEvent, 1);
      expect(canvas._positiveCount).toBe(0);
    });

    it('ignored when loading', async () => {
      canvas.activate(30, 640, 480);
      canvas._loading = true;
      const fakeEvent = { clientX: 100, clientY: 100 };

      await canvas._handleClick(fakeEvent, 1);
      // positiveCount was reset to 0 by activate, and _handleClick should not increment it
      expect(canvas._positiveCount).toBe(0);
    });
  });

  // --- _undo ---

  describe('_undo', () => {
    it('POSTs to /api/sam2/undo and decrements count', async () => {
      canvas.activate(30, 640, 480);
      canvas._loading = false;
      canvas._positiveCount = 3;
      addRoute('/api/sam2/undo', blobResponse('png-data', 'image/png'));

      await canvas._undo();

      expect(globalThis.fetch).toHaveBeenCalledWith(
        '/api/sam2/undo',
        expect.objectContaining({ method: 'POST' }),
      );
      expect(canvas._positiveCount).toBe(2);
    });
  });

  // --- _clear ---

  describe('_clear', () => {
    it('POSTs to /api/sam2/clear and resets counts to 0', async () => {
      canvas.activate(30, 640, 480);
      canvas._loading = false;
      canvas._positiveCount = 3;
      canvas._negativeCount = 2;
      addRoute('/api/sam2/clear', blobResponse('png-data', 'image/png'));

      await canvas._clear();

      expect(globalThis.fetch).toHaveBeenCalledWith(
        '/api/sam2/clear',
        expect.objectContaining({ method: 'POST' }),
      );
      expect(canvas._positiveCount).toBe(0);
      expect(canvas._negativeCount).toBe(0);
    });
  });

  // --- _confirm ---

  describe('_confirm', () => {
    it('POSTs to /api/sam2/confirm and disables confirm button', async () => {
      canvas.activate(30, 640, 480);
      canvas._loading = false;
      addRoute('/api/sam2/confirm', {
        ok: true,
        status: 200,
        json: async () => ({}),
        text: async () => '{}',
        blob: async () => new Blob(),
      });

      await canvas._confirm();

      expect(globalThis.fetch).toHaveBeenCalledWith(
        '/api/sam2/confirm',
        expect.objectContaining({ method: 'POST' }),
      );
      expect(document.getElementById('sam2-confirm').disabled).toBe(true);
    });
  });
});
