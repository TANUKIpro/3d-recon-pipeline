/**
 * Overview page controller — displays a responsive card grid of all
 * processed objects with thumbnails, stage-dot progress, and actions.
 */

import { STAGE_COUNT } from './constants.js';

const STAGE_LABELS = [
  'Frames', 'Pi3X', 'SAM2', 'Denoise', 'Mesh', 'Wrap', 'Repair', 'Texture',
];

export class OverviewPanel {
  constructor() {
    this._grid = document.getElementById('overview-grid');
    this._empty = document.getElementById('overview-empty');
    this._objects = [];
    this._stale = true;
    this._activeObject = null;

    /** @type {((name: string) => void) | null} */
    this.onOpenObject = null;
    /** @type {(() => void) | null} */
    this.onNewPipeline = null;

    // Wire "New Pipeline" button
    const newBtn = document.getElementById('overview-new-btn');
    if (newBtn) {
      newBtn.addEventListener('click', () => {
        if (this.onNewPipeline) this.onNewPipeline();
      });
    }
  }

  markStale() { this._stale = true; }

  async refresh() {
    try {
      const res = await fetch('/api/pipeline/objects');
      const data = await res.json();
      this._objects = data.objects || [];
      this._activeObject = data.active_object || null;
      this._render();
      this._stale = false;
    } catch (e) {
      console.error('Overview refresh failed:', e);
    }
  }

  async refreshIfStale() {
    if (this._stale) await this.refresh();
  }

  setActiveObject(name) {
    this._activeObject = name || null;
    if (!this._grid) return;
    for (const card of this._grid.children) {
      card.classList.toggle('overview-card-active', card.dataset.objectName === name);
    }
  }

  // ── Rendering ────────────────────────────────────────────

  _render() {
    if (!this._grid || !this._empty) return;
    this._grid.innerHTML = '';

    if (this._objects.length === 0) {
      this._empty.classList.remove('hidden');
      return;
    }
    this._empty.classList.add('hidden');

    for (const obj of this._objects) {
      this._grid.appendChild(this._createCard(obj));
    }
  }

  _createCard(obj) {
    const card = document.createElement('div');
    card.className = 'overview-card';
    card.dataset.objectName = obj.name;
    if (obj.name === this._activeObject) card.classList.add('overview-card-active');

    // ── Thumbnail ──
    const thumbWrap = document.createElement('div');
    thumbWrap.className = 'overview-thumb';
    const img = document.createElement('img');
    img.src = `/api/preview/object-file/${encodeURIComponent(obj.name)}/frames/00000.jpg`;
    img.alt = obj.name;
    img.loading = 'lazy';
    img.onerror = () => {
      img.style.display = 'none';
      thumbWrap.classList.add('overview-thumb-placeholder');
    };
    thumbWrap.appendChild(img);
    card.appendChild(thumbWrap);

    // ── Card body ──
    const body = document.createElement('div');
    body.className = 'overview-card-body';

    // Name
    const nameEl = document.createElement('div');
    nameEl.className = 'overview-card-name';
    nameEl.textContent = obj.name;
    nameEl.title = obj.name;
    body.appendChild(nameEl);

    // Video name
    if (obj.video_name) {
      const video = document.createElement('div');
      video.className = 'overview-card-video';
      video.textContent = obj.video_name;
      video.title = obj.video_path || '';
      body.appendChild(video);
    }

    // Stage dots
    const dotsRow = document.createElement('div');
    dotsRow.className = 'overview-stage-dots';
    for (let i = 1; i <= STAGE_COUNT; i++) {
      const done = obj.stages && obj.stages[String(i)];
      const dot = document.createElement('span');
      dot.className = 'overview-dot' + (done ? ' overview-dot-complete' : '');
      dot.title = `${STAGE_LABELS[i - 1]}: ${done ? 'Complete' : 'Pending'}`;
      dotsRow.appendChild(dot);
    }
    body.appendChild(dotsRow);

    // Meta (files · size)
    const meta = document.createElement('div');
    meta.className = 'overview-card-meta';
    const parts = [];
    if (obj.file_count != null) parts.push(`${obj.file_count} files`);
    if (obj.size_mb != null) parts.push(`${obj.size_mb} MB`);
    meta.textContent = parts.join(' \u00b7 ');
    body.appendChild(meta);

    // Updated timestamp
    if (obj.updated_at) {
      const ts = document.createElement('div');
      ts.className = 'overview-card-ts';
      ts.textContent = this._fmtDate(obj.updated_at);
      ts.title = obj.updated_at;
      body.appendChild(ts);
    }

    // Artifacts (expandable)
    if (obj.artifacts && obj.artifacts.length > 0) {
      const det = document.createElement('details');
      det.className = 'overview-artifacts';
      const sum = document.createElement('summary');
      sum.textContent = `${obj.artifacts.length} artifact${obj.artifacts.length !== 1 ? 's' : ''}`;
      det.appendChild(sum);
      const ul = document.createElement('ul');
      ul.className = 'overview-artifact-list';
      for (const a of obj.artifacts) {
        const li = document.createElement('li');
        li.className = 'overview-artifact-item';
        const n = document.createElement('span');
        n.className = 'overview-artifact-name';
        n.textContent = a.name;
        li.appendChild(n);
        const s = document.createElement('span');
        s.className = 'overview-artifact-size';
        s.textContent = `${a.size_mb} MB`;
        li.appendChild(s);
        ul.appendChild(li);
      }
      det.appendChild(ul);
      det.addEventListener('click', (e) => e.stopPropagation());
      body.appendChild(det);
    }

    card.appendChild(body);

    // ── Actions ──
    const actions = document.createElement('div');
    actions.className = 'overview-card-actions';
    const openBtn = document.createElement('button');
    openBtn.className = 'btn-primary btn-small';
    openBtn.textContent = 'Open';
    openBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      if (this.onOpenObject) this.onOpenObject(obj.name);
    });
    actions.appendChild(openBtn);
    card.appendChild(actions);

    // Entire card clickable
    card.addEventListener('click', () => {
      if (this.onOpenObject) this.onOpenObject(obj.name);
    });

    return card;
  }

  _fmtDate(iso) {
    try {
      const d = new Date(iso);
      const diff = Date.now() - d.getTime();
      if (diff < 60_000) return 'just now';
      if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
      if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`;
      if (diff < 604_800_000) return `${Math.floor(diff / 86_400_000)}d ago`;
      return d.toLocaleDateString();
    } catch { return iso; }
  }
}
