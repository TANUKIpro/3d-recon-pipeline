/**
 * Minimal hash-based view router.
 * Sets `data-view` attribute on <body> and fires a callback on view change.
 */

const VIEWS = new Set(['overview', 'pipeline']);
const DEFAULT_VIEW = 'overview';

export class ViewRouter {
  constructor() {
    this._view = null;
    this._onChange = null;
    window.addEventListener('hashchange', () => this._apply());
    this._apply();
  }

  get view() { return this._view; }

  set onChange(fn) { this._onChange = fn; }

  navigate(view) {
    if (!VIEWS.has(view)) return;
    window.location.hash = view;
  }

  _apply() {
    const raw = window.location.hash.replace('#', '').toLowerCase();
    const view = VIEWS.has(raw) ? raw : DEFAULT_VIEW;
    if (view === this._view) return;
    this._view = view;
    document.body.setAttribute('data-view', view);
    if (this._onChange) this._onChange(view);
  }
}
