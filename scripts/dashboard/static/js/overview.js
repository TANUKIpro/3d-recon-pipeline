/**
 * Overview page controller — displays a responsive card grid of all
 * processed objects with thumbnails, stage-dot progress, and actions.
 */

import { STAGE_COUNT, STORAGE_KEYS } from './constants.js';

const STAGE_LABELS = [
  'Frames', 'COLMAP', 'SAM2', 'gs2mesh', 'Texture', 'Post Cleanup',
];

export class OverviewPanel {
  constructor() {
    this._grid = document.getElementById('overview-grid');
    this._empty = document.getElementById('overview-empty');
    this._objects = [];
    this._stale = true;
    this._activeObject = null;
    this._sortOrder = localStorage.getItem(STORAGE_KEYS.sortOrder) || 'asc';
    this.currentBranchSlug = null;

    /** @type {((name: string, branch?: string) => void) | null} */
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

    // Wire sort selector
    this._sortSelect = document.getElementById('overview-sort');
    if (this._sortSelect) {
      this._sortSelect.value = this._sortOrder;
      this._sortSelect.addEventListener('change', () => {
        this._sortOrder = this._sortSelect.value;
        localStorage.setItem(STORAGE_KEYS.sortOrder, this._sortOrder);
        this._render();
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

    const sorted = this._sortedObjects();
    for (const obj of sorted) {
      this._grid.appendChild(this._createCard(obj));
    }
  }

  _sortedObjects() {
    const list = [...this._objects];
    switch (this._sortOrder) {
      case 'asc':
        list.sort((a, b) => a.name.localeCompare(b.name));
        break;
      case 'desc':
        list.sort((a, b) => b.name.localeCompare(a.name));
        break;
      case 'updated':
        list.sort((a, b) => (b.updated_at || '').localeCompare(a.updated_at || ''));
        break;
    }
    return list;
  }

  _createCard(obj) {
    const card = document.createElement('div');
    card.className = 'overview-card';
    card.dataset.objectName = obj.name;
    if (obj.name === this._activeObject) card.classList.add('overview-card-active');

    if (obj.locked) card.classList.add('overview-card-locked');

    // ── Thumbnail ──
    const thumbWrap = document.createElement('div');
    thumbWrap.className = 'overview-thumb';
    const img = document.createElement('img');
    const branchParam = obj.branch ? `?branch=${encodeURIComponent(obj.branch)}` : '';
    img.src = `/api/preview/object-file/${encodeURIComponent(obj.name)}/p1_frames/00000.jpg${branchParam}`;
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

    // Branch badge
    if (obj.branch) {
      const branchBadge = document.createElement('span');
      branchBadge.className = 'overview-branch-badge';
      if (obj.locked) branchBadge.classList.add('overview-branch-locked');
      branchBadge.textContent = obj.branch;
      branchBadge.title = obj.locked
        ? `Read-only (branch: ${obj.branch})`
        : `Branch: ${obj.branch}`;
      body.appendChild(branchBadge);
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

    if (obj.locked) {
      openBtn.className = 'btn-small btn-locked';
      openBtn.textContent = 'Locked';
      openBtn.disabled = true;
      // Prevent all interaction on locked cards
      card.style.cursor = 'default';
      card.addEventListener('click', (e) => e.stopPropagation());
    } else {
      openBtn.className = 'btn-primary btn-small';
      openBtn.textContent = 'Open';
      openBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (this.onOpenObject) this.onOpenObject(obj.name, obj.branch);
      });
      // Entire card clickable
      card.addEventListener('click', () => {
        if (this.onOpenObject) this.onOpenObject(obj.name, obj.branch);
      });
    }

    actions.appendChild(openBtn);
    card.appendChild(actions);

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
