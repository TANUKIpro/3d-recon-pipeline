import { describe, it, expect, vi, beforeEach } from 'vitest';
import { buildOverviewDOM } from './helpers/dom-factory.js';
import { useFetchMock } from './helpers/fetch-mock.js';
import { OverviewPanel } from '../../../scripts/dashboard/static/js/overview.js';

describe('OverviewPanel', () => {
  let panel;
  const fetchMock = useFetchMock();

  const fakeObjects = [
    {
      name: 'coffee-mug',
      video_name: 'coffee.mp4',
      video_path: '/data/input/coffee.mp4',
      stages: { '1': true, '2': true, '3': false },
      file_count: 5,
      size_mb: 12.3,
      updated_at: new Date().toISOString(),
      artifacts: [],
    },
    {
      name: 'chair-01',
      video_name: 'chair.mp4',
      video_path: '/data/input/chair.mp4',
      stages: { '1': true },
      file_count: 2,
      size_mb: 3.1,
      updated_at: new Date().toISOString(),
      artifacts: [],
    },
  ];

  beforeEach(() => {
    buildOverviewDOM();
    fetchMock.installFetch();
    fetchMock.addRoute('/api/pipeline/objects', fetchMock.jsonResponse({
      objects: fakeObjects,
      active_object: null,
    }));
    panel = new OverviewPanel();
  });

  // 30.1 — refresh() fetches /api/pipeline/objects
  it('refresh() fetches /api/pipeline/objects', async () => {
    await panel.refresh();

    const calls = globalThis.fetch.mock.calls
      .filter(c => String(c[0]).includes('/api/pipeline/objects'));
    expect(calls.length).toBeGreaterThanOrEqual(1);
  });

  // 30.2 — Card has name and stage dots
  it('card has name and stage dots', async () => {
    await panel.refresh();

    const grid = document.getElementById('overview-grid');
    const cards = grid.querySelectorAll('.overview-card');
    expect(cards.length).toBe(2);

    const name = cards[0].querySelector('.overview-card-name');
    expect(name.textContent).toBe('coffee-mug');

    const dots = cards[0].querySelectorAll('.overview-dot');
    expect(dots.length).toBe(8);
  });

  // 30.3 — Card click → onOpenObject callback
  it('card click triggers onOpenObject callback', async () => {
    const cb = vi.fn();
    panel.onOpenObject = cb;
    await panel.refresh();

    const card = document.querySelector('.overview-card');
    card.click();

    expect(cb).toHaveBeenCalledWith('coffee-mug');
  });

  // 30.4 — New button → onNewPipeline callback
  it('new button triggers onNewPipeline callback', () => {
    const cb = vi.fn();
    panel.onNewPipeline = cb;

    const btn = document.getElementById('overview-new-btn');
    btn.click();

    expect(cb).toHaveBeenCalled();
  });

  // 30.5 — markStale + refreshIfStale re-fetches
  it('markStale + refreshIfStale re-fetches', async () => {
    await panel.refresh();
    const callsBefore = globalThis.fetch.mock.calls.length;

    // Should not re-fetch (not stale)
    await panel.refreshIfStale();
    expect(globalThis.fetch.mock.calls.length).toBe(callsBefore);

    // Mark stale → should re-fetch
    panel.markStale();
    await panel.refreshIfStale();
    expect(globalThis.fetch.mock.calls.length).toBeGreaterThan(callsBefore);
  });

  // 30.6 — setActiveObject sets .overview-card-active class
  it('setActiveObject sets active class', async () => {
    await panel.refresh();

    panel.setActiveObject('chair-01');

    const cards = document.querySelectorAll('.overview-card');
    const active = document.querySelectorAll('.overview-card-active');
    expect(active.length).toBe(1);
    expect(active[0].dataset.objectName).toBe('chair-01');
  });

  // 30.7 — Empty objects → #overview-empty shown
  it('empty objects shows empty state', async () => {
    fetchMock.addRoute('/api/pipeline/objects', fetchMock.jsonResponse({
      objects: [],
      active_object: null,
    }));

    await panel.refresh();

    const empty = document.getElementById('overview-empty');
    expect(empty.style.display).not.toBe('none');
  });
});
