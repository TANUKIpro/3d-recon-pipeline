import { describe, it, expect, vi, beforeEach } from 'vitest';
import { buildOverviewDOM } from './helpers/dom-factory.js';
import { useFetchMock } from './helpers/fetch-mock.js';
import { OverviewPanel } from '../../../scripts/dashboard/static/js/overview.js';

const flushPromises = () => new Promise((resolve) => setTimeout(resolve, 0));

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

  function cardFor(name) {
    return document.querySelector(`.overview-card[data-object-name="${name}"]`);
  }

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

    const name = cardFor('coffee-mug').querySelector('.overview-card-name');
    expect(name.textContent).toBe('coffee-mug');

    const dots = cardFor('coffee-mug').querySelectorAll('.overview-dot');
    expect(dots.length).toBe(6);
  });

  // 30.3 — Card click → onOpenObject callback
  it('card click triggers onOpenObject callback', async () => {
    const cb = vi.fn();
    panel.onOpenObject = cb;
    await panel.refresh();

    const card = cardFor('coffee-mug');
    card.click();

    expect(cb).toHaveBeenCalledWith('coffee-mug', undefined);
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
    expect(empty.classList.contains('hidden')).toBe(false);
  });

  it('shows delete menu for unlocked cards', async () => {
    await panel.refresh();

    const card = cardFor('coffee-mug');

    expect(card.querySelector('.overview-menu-trigger')).not.toBeNull();
    expect(card.querySelector('.overview-menu-dropdown')).not.toBeNull();
  });

  it('menu click does not open object', async () => {
    const cb = vi.fn();
    panel.onOpenObject = cb;
    await panel.refresh();

    document.querySelector('.overview-menu-trigger').click();

    expect(cb).not.toHaveBeenCalled();
    expect(document.querySelector('.overview-card-menu').classList.contains('open')).toBe(true);
  });

  it('confirming delete calls endpoint and removes card', async () => {
    vi.spyOn(window, 'confirm').mockReturnValue(true);
    await panel.refresh();
    fetchMock.clearRoutes();
    fetchMock.addRoute('/api/pipeline/objects/coffee-mug', (url, options) => {
      expect(options.method).toBe('DELETE');
      return fetchMock.jsonResponse({ status: 'deleted', object_name: 'coffee-mug' });
    });

    cardFor('coffee-mug').querySelector('.overview-menu-trigger').click();
    await cardFor('coffee-mug').querySelector('.overview-menu-danger').click();
    await flushPromises();

    const cards = [...document.querySelectorAll('.overview-card')];
    expect(cards.map((c) => c.dataset.objectName)).toEqual(['chair-01']);
  });

  it('cancelling delete does not call delete endpoint', async () => {
    vi.spyOn(window, 'confirm').mockReturnValue(false);
    await panel.refresh();
    const before = globalThis.fetch.mock.calls.length;

    cardFor('coffee-mug').querySelector('.overview-menu-trigger').click();
    await cardFor('coffee-mug').querySelector('.overview-menu-danger').click();
    await flushPromises();

    const deleteCalls = globalThis.fetch.mock.calls.slice(before)
      .filter((c) => String(c[0]).includes('/api/pipeline/objects/'));
    expect(deleteCalls.length).toBe(0);
    expect(document.querySelectorAll('.overview-card').length).toBe(2);
  });

  it('locked cards do not expose delete menu', async () => {
    fetchMock.addRoute('/api/pipeline/objects', fetchMock.jsonResponse({
      objects: [{ ...fakeObjects[0], locked: true, branch: 'other' }],
      active_object: null,
    }));

    await panel.refresh();

    expect(document.querySelector('.overview-card-locked')).not.toBeNull();
    expect(document.querySelector('.overview-menu-trigger')).toBeNull();
  });

  it('failed delete leaves card visible', async () => {
    vi.spyOn(window, 'confirm').mockReturnValue(true);
    vi.spyOn(window, 'alert').mockImplementation(() => {});
    await panel.refresh();
    fetchMock.clearRoutes();
    fetchMock.addRoute('/api/pipeline/objects/coffee-mug', fetchMock.jsonResponse({
      error: 'Cannot delete objects while pipeline is running',
    }, 409));

    cardFor('coffee-mug').querySelector('.overview-menu-trigger').click();
    await cardFor('coffee-mug').querySelector('.overview-menu-danger').click();
    await flushPromises();

    expect(window.alert).toHaveBeenCalledWith('Cannot delete objects while pipeline is running');
    expect(document.querySelectorAll('.overview-card').length).toBe(2);
  });
});
