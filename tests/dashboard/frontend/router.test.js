import { describe, it, expect, vi, beforeEach } from 'vitest';
import { buildRouterDOM } from './helpers/dom-factory.js';
import { ViewRouter } from '../../../scripts/dashboard/static/js/router.js';

describe('ViewRouter', () => {
  let router;

  beforeEach(() => {
    buildRouterDOM();
    window.location.hash = '';
    router = new ViewRouter();
  });

  // 13.1 — Hash #pipeline → view === 'pipeline'
  it('hash #pipeline → view is pipeline', () => {
    window.location.hash = '#pipeline';
    window.dispatchEvent(new Event('hashchange'));
    expect(router.view).toBe('pipeline');
  });

  // 13.2 — Only overview/pipeline valid
  it('only overview and pipeline are valid views', () => {
    // Default is overview
    expect(router.view).toBe('overview');

    window.location.hash = '#pipeline';
    window.dispatchEvent(new Event('hashchange'));
    expect(router.view).toBe('pipeline');
  });

  // 13.3 — Unknown hash → defaults to overview
  it('unknown hash defaults to overview', () => {
    window.location.hash = '#settings';
    window.dispatchEvent(new Event('hashchange'));
    expect(router.view).toBe('overview');
  });

  // 13.4 — onChange callback fires on view change
  it('onChange callback fires on view change', () => {
    const cb = vi.fn();
    router.onChange = cb;

    window.location.hash = '#pipeline';
    window.dispatchEvent(new Event('hashchange'));

    expect(cb).toHaveBeenCalledWith('pipeline');
  });

  // 13.5 — body[data-view] set correctly
  it('body[data-view] attribute set correctly', () => {
    expect(document.body.getAttribute('data-view')).toBe('overview');

    window.location.hash = '#pipeline';
    window.dispatchEvent(new Event('hashchange'));

    expect(document.body.getAttribute('data-view')).toBe('pipeline');
  });
});
