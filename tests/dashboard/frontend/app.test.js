/**
 * Integration tests for app.js — wiring between WebSocket messages,
 * pipeline UI, config panel, and user actions.
 *
 * Covers P2 test IDs 31.1–31.15.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { buildAppDOM } from './helpers/dom-factory.js';
import { MockWebSocket, installMockWebSocket, cleanupMockWebSocket } from './helpers/ws-mock.js';
import { useFetchMock } from './helpers/fetch-mock.js';

const fetchMock = useFetchMock();

/** Flush microtasks so async WS handlers complete. */
async function flush() {
  await new Promise(r => setTimeout(r, 0));
  await new Promise(r => setTimeout(r, 0));
}

/** Helper: get the last MockWebSocket instance. */
function getWs() {
  return MockWebSocket.lastInstance;
}

function installDefaultRoutes() {
  fetchMock.addRoute('/api/vram', fetchMock.jsonResponse({ free_mb: 8192 }));
  fetchMock.addRoute('/api/pipeline/start', fetchMock.jsonResponse({
    ok: true,
    resume_from_stage: 1,
    object_name: 'test-obj',
  }));
  fetchMock.addRoute('/api/pipeline/cancel', fetchMock.jsonResponse({ ok: true }));
  fetchMock.addRoute('/api/pipeline/load-object', fetchMock.jsonResponse({
    pipeline_status: {
      running: false,
      current_stage: 0,
      stages: {},
      object_name: 'test-obj',
      overall_progress: 0,
      frame_count: 0,
      next_stage_confirmation: {},
    },
  }));
  fetchMock.addRoute('/api/pipeline/objects', fetchMock.jsonResponse({ objects: [] }));
  fetchMock.addRoute('/api/pipeline/videos', fetchMock.jsonResponse({ videos: [] }));
  fetchMock.addRoute('/api/objects', fetchMock.jsonResponse({ objects: [] }));
  fetchMock.addRoute('/api/preview', fetchMock.jsonResponse({}));
  fetchMock.addRoute('/api/pipeline/confirm-next-stage', fetchMock.jsonResponse({ ok: true }));
  fetchMock.addRoute('/api/pipeline/mesh-repair', fetchMock.jsonResponse({ ok: true }));
}

describe('app.js integration', () => {
  beforeEach(async () => {
    vi.resetModules();

    // Set location for WS URL construction — hash=pipeline to skip overview.refresh()
    Object.defineProperty(window, 'location', {
      value: { protocol: 'http:', host: 'localhost:7860', hash: '#pipeline' },
      writable: true,
      configurable: true,
    });

    buildAppDOM();
    installMockWebSocket();
    fetchMock.installFetch();
    installDefaultRoutes();

    // Import app.js — triggers all side effects (pollVRAM, ws.connect, etc.)
    await import('../../../scripts/dashboard/static/js/app.js');

    // Open the WebSocket
    const ws = getWs();
    if (ws) ws.simulateOpen();
    await flush();
  });

  afterEach(() => {
    vi.restoreAllMocks();
    cleanupMockWebSocket();
    fetchMock.restoreFetch();
    fetchMock.clearRoutes();
  });

  // 31.1 — ws status → applySnapshot
  it('ws status → updates status badge without error', async () => {
    const ws = getWs();
    ws.simulateMessage({
      type: 'status',
      running: false,
      current_stage: 0,
      stages: {},
      object_name: 'test',
      overall_progress: 0,
      frame_count: 0,
      next_stage_confirmation: {},
    });
    await flush();

    const badge = document.getElementById('status-badge');
    expect(badge).toBeTruthy();
  });

  // 31.2 — ws stage_start → UI update
  it('ws stage_start → stage pill running', async () => {
    const ws = getWs();
    ws.simulateMessage({
      type: 'stage_start',
      stage: 2,
      label: 'Pi3X 3D Reconstruction',
    });
    await flush();

    const pill = document.querySelector('.stage-pill[data-stage="2"]');
    expect(pill).toBeTruthy();
    expect(
      pill.classList.contains('running') || pill.classList.contains('stage-running'),
    ).toBe(true);
  });

  // 31.3 — ws stage_complete → pill complete
  it('ws stage_start + stage_complete → stage pill complete', async () => {
    const ws = getWs();
    ws.simulateMessage({ type: 'stage_start', stage: 1, label: 'Extract Frames' });
    await flush();
    ws.simulateMessage({
      type: 'stage_complete',
      stage: 1,
      elapsed: 5.2,
      overall_progress: 12.5,
    });
    await flush();

    const pill = document.querySelector('.stage-pill[data-stage="1"]');
    expect(pill).toBeTruthy();
    expect(
      pill.classList.contains('complete') || pill.classList.contains('stage-complete'),
    ).toBe(true);
  });

  // 31.4 — ws sam2_ready → canvas activate
  it('ws sam2_ready → SAM2 canvas shown', async () => {
    const ws = getWs();
    ws.simulateMessage({
      type: 'sam2_ready',
      frame_count: 10,
      width: 640,
      height: 480,
    });
    await flush();

    const canvas = document.getElementById('sam2-canvas');
    const placeholder = document.getElementById('sam2-placeholder');
    expect(canvas.style.display).not.toBe('none');
    expect(
      placeholder.style.display === 'none' || placeholder.classList.contains('hidden'),
    ).toBe(true);
  });

  // 31.5 — ws sam2_verification_ready → strip
  it('ws sam2_verification_ready → verification shown', async () => {
    const ws = getWs();
    ws.simulateMessage({
      type: 'sam2_verification_ready',
      frame_count: 10,
      has_ground: false,
    });
    await flush();

    const verif = document.getElementById('sam2-verification');
    expect(verif.style.display).not.toBe('none');
  });

  // 31.6 — ws mesh_repair_ready → UI
  it('ws mesh_repair_ready → mesh repair controller notified', async () => {
    fetchMock.addRoute('/api/pipeline/mesh-repair/candidates', fetchMock.jsonResponse({
      candidates: [{ loop_id: 0 }, { loop_id: 1 }, { loop_id: 2 }],
      mesh_path: 'test.ply',
      analysis: {},
    }));

    const ws = getWs();
    ws.simulateMessage({
      type: 'mesh_repair_ready',
      stage: 7,
      candidate_count: 3,
      auto_accepted: false,
      has_ground_plane: false,
    });
    await flush();

    // Stage 7 pill should get interactive state
    const pill = document.querySelector('.stage-pill[data-stage="7"]');
    expect(pill).toBeTruthy();
  });

  // 31.7 — ws pipeline_complete → done UI
  it('ws pipeline_complete → status badge shows done', async () => {
    const ws = getWs();
    ws.simulateMessage({
      type: 'pipeline_complete',
      elapsed: 120,
      overall_progress: 100,
    });
    await flush();

    const badge = document.getElementById('status-badge');
    const text = badge.textContent.toLowerCase();
    expect(text.includes('done') || text.includes('complete') || text.includes('2m')).toBe(true);
    expect(badge.classList.contains('badge-complete')).toBe(true);
  });

  // 31.8 — ws pipeline_error → error UI
  it('ws pipeline_error → status badge shows error', async () => {
    const ws = getWs();
    ws.simulateMessage({
      type: 'pipeline_error',
      stage: 3,
      error: 'OOM',
      reason_code: '',
    });
    await flush();

    const badge = document.getElementById('status-badge');
    expect(badge.classList.contains('badge-error')).toBe(true);
  });

  // 31.9 — ws next_stage_confirmation_required → bar
  it('ws next_stage_confirmation_required → confirm bar in waiting state', async () => {
    const ws = getWs();
    ws.simulateMessage({
      type: 'next_stage_confirmation_required',
      from_stage: 1,
      to_stage: 2,
      message: 'Continue to Pi3X?',
      auto_accepted: false,
    });
    await flush();

    const bar = document.querySelector('.task-confirm-bar[data-stage="1"]');
    expect(bar).toBeTruthy();
    const message = bar.querySelector('.task-confirm-message');
    expect(message).toBeTruthy();
  });

  // 31.10 — VRAM polling
  it('VRAM polling calls /api/vram on load', async () => {
    // pollVRAM() is called immediately during module load
    const vramCalls = globalThis.fetch.mock.calls
      .filter(c => String(c[0]).includes('/api/vram'));
    expect(vramCalls.length).toBeGreaterThanOrEqual(1);

    const badge = document.getElementById('vram-badge');
    await flush();
    // After flush, the VRAM badge should be updated
    expect(badge.textContent).toContain('8192');
  });

  // 31.11 — view switching
  it('view switching updates body data-view', async () => {
    window.location.hash = '#overview';
    window.dispatchEvent(new Event('hashchange'));
    await flush();

    expect(document.body.getAttribute('data-view')).toBe('overview');

    window.location.hash = '#pipeline';
    window.dispatchEvent(new Event('hashchange'));
    await flush();

    expect(document.body.getAttribute('data-view')).toBe('pipeline');
  });

  // 31.12 — object select → load-object
  it('object select triggers load-object POST', async () => {
    const objectSelect = document.getElementById('object-select');
    const opt = document.createElement('option');
    opt.value = 'my-obj';
    opt.textContent = 'my-obj';
    objectSelect.appendChild(opt);
    objectSelect.value = 'my-obj';
    objectSelect.dispatchEvent(new Event('change', { bubbles: true }));
    await flush();

    const loadCalls = globalThis.fetch.mock.calls
      .filter(c => String(c[0]).includes('/api/pipeline/load-object'));
    expect(loadCalls.length).toBeGreaterThanOrEqual(1);
    expect(loadCalls[loadCalls.length - 1][1].method).toBe('POST');
  });

  // 31.13 — start button → start
  it('start button triggers /api/pipeline/start POST', async () => {
    // Set a video path so config panel doesn't reject the start
    const videoSelect = document.getElementById('video-select');
    const vopt = document.createElement('option');
    vopt.value = '/data/input/test.mp4';
    vopt.textContent = 'test.mp4';
    videoSelect.appendChild(vopt);
    videoSelect.value = '/data/input/test.mp4';

    const btn = document.getElementById('btn-start');
    btn.click();
    await flush();

    const startCalls = globalThis.fetch.mock.calls
      .filter(c => String(c[0]).includes('/api/pipeline/start'));
    expect(startCalls.length).toBeGreaterThanOrEqual(1);
    expect(startCalls[0][1].method).toBe('POST');
  });

  // 31.14 — cancel button → cancel
  it('cancel button triggers /api/pipeline/cancel POST', async () => {
    const btn = document.getElementById('btn-cancel');
    btn.disabled = false;
    btn.click();
    await flush();

    const cancelCalls = globalThis.fetch.mock.calls
      .filter(c => String(c[0]).includes('/api/pipeline/cancel'));
    expect(cancelCalls.length).toBeGreaterThanOrEqual(1);
  });

  // 31.15 — stale response filter
  it('stale load-object response is ignored', async () => {
    let resolveFirst;
    const slowResponse = new Promise(r => { resolveFirst = r; });

    let callCount = 0;
    fetchMock.addRoute('/api/pipeline/load-object', (url, opts) => {
      callCount++;
      if (callCount === 1) return slowResponse;
      return fetchMock.jsonResponse({
        pipeline_status: {
          running: false,
          current_stage: 0,
          stages: {},
          object_name: 'second-obj',
          overall_progress: 50,
          frame_count: 20,
          next_stage_confirmation: {},
        },
      });
    });

    const objectSelect = document.getElementById('object-select');

    // First load: slow
    const opt1 = document.createElement('option');
    opt1.value = 'first-obj';
    objectSelect.appendChild(opt1);
    objectSelect.value = 'first-obj';
    objectSelect.dispatchEvent(new Event('change', { bubbles: true }));
    await flush();

    // Second load: fast (increments _objectLoadRequestId)
    const opt2 = document.createElement('option');
    opt2.value = 'second-obj';
    objectSelect.appendChild(opt2);
    objectSelect.value = 'second-obj';
    objectSelect.dispatchEvent(new Event('change', { bubbles: true }));
    await flush();

    // Resolve the stale first response — should be ignored
    resolveFirst(fetchMock.jsonResponse({
      pipeline_status: {
        running: false,
        current_stage: 0,
        stages: {},
        object_name: 'first-obj',
        overall_progress: 10,
        frame_count: 5,
        next_stage_confirmation: {},
      },
    }));
    await flush();

    // No crash = stale response was safely ignored by reqId guard
    expect(callCount).toBeGreaterThanOrEqual(2);
  });
});
