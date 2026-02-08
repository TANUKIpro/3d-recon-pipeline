/**
 * Configuration panel: video/object selection and parameter inputs.
 */

const NEW_OBJECT_VALUE = '__new__';
const STAGE_LABELS = {
  1: 'Extract Frames',
  2: 'Pi3X',
  3: 'SAM2',
  4: 'Denoise',
  5: 'DiffCD Mesh',
  6: 'Texture Bake',
};

export class ConfigPanel {
  constructor() {
    this._panel = document.getElementById('config-panel');
    this._title = document.getElementById('config-title');
    this._sections = document.querySelectorAll('.config-section[data-stages]');

    this._videoSelect = document.getElementById('video-select');
    this._videoInfo = document.getElementById('video-info');
    this._objectSelect = document.getElementById('object-select');
    this._objectNameInput = document.getElementById('cfg-object-name');
    this._objectInfo = document.getElementById('object-info');
    this._objectArtifacts = document.getElementById('object-artifacts');
    this._objectArtifactsEmpty = document.getElementById('object-artifacts-empty');
    this._resumeStageInfo = document.getElementById('cfg-resume-stage-info');
    this._refreshObjectsBtn = document.getElementById('btn-refresh-objects');
    this._startBtn = document.getElementById('btn-start');
    this._cancelBtn = document.getElementById('btn-cancel');

    this._inputs = {
      frame_interval: document.getElementById('cfg-frame-interval'),
      max_frames: document.getElementById('cfg-max-frames'),
      pixel_limit: document.getElementById('cfg-pixel-limit'),
      confidence_threshold: document.getElementById('cfg-conf-threshold'),
      edge_rtol: document.getElementById('cfg-edge-rtol'),
      sam2_model: document.getElementById('cfg-sam2-model'),
      diffcd_batch_size: document.getElementById('cfg-diffcd-batch'),
      diffcd_n_batches: document.getElementById('cfg-diffcd-nbatches'),
      diffcd_resolution: document.getElementById('cfg-diffcd-res'),
      texture_size: document.getElementById('cfg-texture-size'),
    };

    this.onStart = null;  // callback(config)
    this.onCancel = null; // callback()
    this.onObjectSelected = null; // callback(objectName|null)
    this._videoMeta = null;
    this._maxFramesAuto = true;
    this._extractDefaults = { frame_interval: 10, max_frames: 50 };
    this._objects = [];
    this._objectInfoRequestId = 0;
    this._objectNameDirty = false;
    this._running = false;
    this._selectedObjectSummary = null;
    this._startStage = 1;

    this._bindEvents();
    this._loadVideos();
    this.refreshObjects();
    this._updateResumeHint();
  }

  setRunning(running) {
    this._running = running;
    this._startBtn.disabled = running;
    this._cancelBtn.disabled = !running;
    this._videoSelect.disabled = running;
    this._objectSelect.disabled = running;
    this._objectNameInput.disabled = running;
    this._refreshObjectsBtn.disabled = running;
    for (const inp of Object.values(this._inputs)) {
      inp.disabled = running;
    }
  }

  setActiveStage(stage) {
    if (stage === null) {
      this._panel.classList.remove('stage-filtered');
      this._sections.forEach(s => s.classList.remove('stage-visible'));
      this._title.innerHTML = 'Configuration';
      this._updateResumeHint();
      return;
    }

    this._panel.classList.add('stage-filtered');
    const stageStr = String(stage);

    this._sections.forEach(s => {
      const stages = s.dataset.stages.split(/\s+/);
      s.classList.toggle('stage-visible', stages.includes(stageStr));
    });

    this._title.innerHTML = `Configuration <span class="config-stage-name">\u2014 ${STAGE_LABELS[stage] || 'Stage '+stage}</span>`;
    this._startStage = this._clampStage(stage);
    this._updateResumeHint();
  }

  setObjectName(name) {
    const normalized = this._normalizeObjectName(name);
    if (!normalized) return;
    this._objectNameInput.value = normalized;
    this._objectNameDirty = false;
    const matched = this._objects.find(o => o.name === normalized);
    if (matched) {
      this._objectSelect.value = normalized;
      this._selectedObjectSummary = matched;
      this._renderObjectSummary(matched);
      this._refreshObjectInfo(normalized);
    } else {
      this._objectSelect.value = NEW_OBJECT_VALUE;
      this._selectedObjectSummary = null;
      this._renderObjectSummary(null, normalized);
      this._renderArtifacts(null);
    }
    this._updateResumeHint();
  }

  getConfig() {
    const suggestedObject = this._suggestObjectNameFromVideo();
    const objectName = this._normalizeObjectName(this._objectNameInput.value || suggestedObject || 'object');
    this._objectNameInput.value = objectName;
    const resumeFromStage = this._resolveResumeStage();

    return {
      video_path: this._videoSelect.value,
      object_name: objectName,
      resume_from_stage: resumeFromStage,
      frame_interval: this._parsePositiveInt(
        this._inputs.frame_interval.value,
        this._extractDefaults.frame_interval,
      ),
      max_frames: this._parsePositiveInt(
        this._inputs.max_frames.value,
        this._extractDefaults.max_frames,
      ),
      pixel_limit: parseInt(this._inputs.pixel_limit.value) || 255000,
      confidence_threshold: parseFloat(this._inputs.confidence_threshold.value) || 0.1,
      edge_rtol: parseFloat(this._inputs.edge_rtol.value) || 0.03,
      sam2_model: this._inputs.sam2_model.value,
      diffcd_batch_size: parseInt(this._inputs.diffcd_batch_size.value) || 3000,
      diffcd_n_batches: parseInt(this._inputs.diffcd_n_batches.value) || 25000,
      diffcd_resolution: parseInt(this._inputs.diffcd_resolution.value) || 384,
      texture_size: parseInt(this._inputs.texture_size.value) || 2048,
    };
  }

  async refreshObjects() {
    try {
      const currentSelect = this._objectSelect.value;
      const currentInput = this._normalizeObjectName(this._objectNameInput.value);
      const res = await fetch('/api/pipeline/objects');
      const data = await res.json();
      const objects = Array.isArray(data.objects) ? data.objects : [];
      this._objects = objects;
      this._populateObjectSelect(objects);

      let target = NEW_OBJECT_VALUE;
      if (currentSelect && currentSelect !== NEW_OBJECT_VALUE && objects.some(o => o.name === currentSelect)) {
        target = currentSelect;
      } else if (currentInput && objects.some(o => o.name === currentInput)) {
        target = currentInput;
      } else if (data.active_object && objects.some(o => o.name === data.active_object)) {
        target = data.active_object;
      }
      this._selectObject(target, { keepInput: true, notify: false });
    } catch (e) {
      this._objectInfo.textContent = 'Failed to load objects';
      this._selectedObjectSummary = null;
      this._renderArtifacts(null);
      this._updateResumeHint();
    }
  }

  async _loadVideos() {
    try {
      const res = await fetch('/api/pipeline/videos');
      const data = await res.json();
      this._videoSelect.innerHTML = '';

      if (data.videos.length === 0) {
        const opt = document.createElement('option');
        opt.value = '';
        opt.textContent = 'No videos found in /data/input/';
        this._videoSelect.appendChild(opt);
        this._applySuggestedObjectName();
        return;
      }

      for (const v of data.videos) {
        const opt = document.createElement('option');
        opt.value = v.path;
        opt.textContent = `${v.name} (${v.size_mb} MB)`;
        opt.dataset.suggestedObjectName = v.suggested_object_name || '';
        this._videoSelect.appendChild(opt);
      }
      this._onVideoChange();
    } catch (e) {
      this._videoSelect.innerHTML = '<option value="">Error loading videos</option>';
    }
  }

  _bindEvents() {
    this._startBtn.addEventListener('click', () => {
      if (this.onStart) {
        const cfg = this.getConfig();
        if (!cfg.video_path && cfg.resume_from_stage === 1) {
          alert('Please select a video file for stage 1 restart.');
          return;
        }
        this.onStart(cfg);
      }
    });

    this._cancelBtn.addEventListener('click', () => {
      if (this.onCancel) this.onCancel();
    });

    this._videoSelect.addEventListener('change', () => this._onVideoChange());
    this._inputs.frame_interval.addEventListener('input', () => this._onFrameIntervalInput());
    this._inputs.max_frames.addEventListener('input', () => this._onMaxFramesInput());

    this._objectSelect.addEventListener('change', () => {
      this._selectObject(this._objectSelect.value);
    });

    this._objectNameInput.addEventListener('input', () => {
      this._objectNameDirty = true;
      const normalized = this._normalizeObjectName(this._objectNameInput.value);
      const matched = this._objects.find(o => o.name === normalized);
      if (matched) {
        this._objectSelect.value = matched.name;
        this._selectedObjectSummary = matched;
        this._renderObjectSummary(matched);
        this._renderArtifacts(matched);
      } else {
        this._objectSelect.value = NEW_OBJECT_VALUE;
        this._selectedObjectSummary = null;
        this._renderObjectSummary(null, normalized);
        this._renderArtifacts(null);
      }
      this._updateResumeHint();
    });

    this._objectNameInput.addEventListener('change', () => {
      const normalized = this._normalizeObjectName(this._objectNameInput.value || this._suggestObjectNameFromVideo() || 'object');
      this._objectNameInput.value = normalized;
      const matched = this._objects.find(o => o.name === normalized);
      if (matched) {
        this._objectSelect.value = matched.name;
        this._objectNameDirty = false;
        this._selectedObjectSummary = matched;
        this._refreshObjectInfo(matched.name);
      } else {
        this._objectSelect.value = NEW_OBJECT_VALUE;
        this._selectedObjectSummary = null;
        this._renderObjectSummary(null, normalized);
      }
      this._updateResumeHint();
    });

    this._refreshObjectsBtn.addEventListener('click', () => {
      this.refreshObjects();
    });
  }

  _populateObjectSelect(objects) {
    this._objectSelect.innerHTML = '';
    const createOpt = document.createElement('option');
    createOpt.value = NEW_OBJECT_VALUE;
    createOpt.textContent = 'Create New Object';
    this._objectSelect.appendChild(createOpt);

    for (const o of objects) {
      const opt = document.createElement('option');
      opt.value = o.name;
      opt.textContent = `${o.name} (${o.complete_stages || 0}/6)`;
      this._objectSelect.appendChild(opt);
    }
  }

  _selectObject(name, opts = {}) {
    const notify = opts.notify !== false;
    if (name && name !== NEW_OBJECT_VALUE) {
      this._objectSelect.value = name;
      this._objectNameInput.value = name;
      this._objectNameDirty = false;
      const summary = this._objects.find(o => o.name === name) || null;
      this._selectedObjectSummary = summary;
      this._renderObjectSummary(summary, name);
      this._renderArtifacts(summary);
      this._refreshObjectInfo(name);
      this._updateResumeHint();
      if (notify && !this._running && this.onObjectSelected) {
        this.onObjectSelected(name);
      }
      return;
    }

    this._objectSelect.value = NEW_OBJECT_VALUE;
    this._selectedObjectSummary = null;
    if (!opts.keepInput || !this._objectNameInput.value.trim()) {
      this._applySuggestedObjectName(true);
    }

    const normalized = this._normalizeObjectName(this._objectNameInput.value);
    if (normalized) {
      this._objectNameInput.value = normalized;
      this._renderObjectSummary(null, normalized);
    } else {
      this._renderObjectSummary(null, '');
    }
    this._renderArtifacts(null);
    this._updateResumeHint();
    if (notify && !this._running && this.onObjectSelected) {
      this.onObjectSelected(null);
    }
  }

  async _refreshObjectInfo(name) {
    if (!name) return;
    const reqId = ++this._objectInfoRequestId;
    try {
      const res = await fetch(`/api/pipeline/object-info?name=${encodeURIComponent(name)}`);
      const data = await res.json();
      if (reqId !== this._objectInfoRequestId) return;
      if (!res.ok || data.error || !data.object) return;
      const object = data.object;
      if (this._objectSelect.value !== object.name) return;
      this._selectedObjectSummary = object;
      if (object.video_path) {
        this._applyObjectVideoPath(object.video_path);
      }
      this._renderObjectSummary(object, object.name);
      this._renderArtifacts(object);
      this._updateResumeHint();
    } catch (e) {
      // Keep previously rendered summary.
    }
  }

  _renderObjectSummary(object, fallbackName = '') {
    if (object) {
      const details = [
        `${object.complete_stages || 0}/6 stages`,
        `${object.file_count || 0} files`,
        `${this._formatSize(object.size_mb)}`,
      ];
      if (object.video_name) details.push(`video: ${object.video_name}`);
      this._objectInfo.textContent = details.join(' | ');
      return;
    }
    if (fallbackName) {
      this._objectInfo.textContent = `New object: ${fallbackName}`;
      return;
    }
    this._objectInfo.textContent = '';
  }

  _renderArtifacts(object) {
    this._objectArtifacts.innerHTML = '';

    let artifacts = [];
    if (object && Array.isArray(object.files)) {
      artifacts = object.files.filter(
        f => !f.path.startsWith('frames/') && !f.path.startsWith('masks/')
      );
    } else if (object && Array.isArray(object.artifacts)) {
      artifacts = object.artifacts;
    }

    if (object?.frame_count > 0) {
      artifacts.unshift({
        name: `frames/ (${object.frame_count} jpg)`,
        size_mb: null,
      });
    }
    if (object?.mask_count > 0) {
      artifacts.unshift({
        name: `masks/ (${object.mask_count} png)`,
        size_mb: null,
      });
    }

    const displayItems = artifacts.slice(0, 16);
    this._objectArtifactsEmpty.style.display = displayItems.length ? 'none' : '';

    for (const f of displayItems) {
      const li = document.createElement('li');
      li.className = 'object-artifact-item';

      const name = document.createElement('span');
      name.className = 'object-artifact-name';
      name.textContent = f.path || f.name || '';
      name.title = f.path || f.name || '';

      const size = document.createElement('span');
      size.className = 'object-artifact-size';
      size.textContent = this._formatSize(f.size_mb);

      li.appendChild(name);
      li.appendChild(size);
      this._objectArtifacts.appendChild(li);
    }
  }

  async _onVideoChange() {
    const path = this._videoSelect.value;
    if (!path) {
      this._videoInfo.textContent = '';
      this._videoMeta = null;
      this._applySuggestedObjectName();
      return;
    }
    this._videoInfo.textContent = 'Loading...';
    try {
      const res = await fetch(`/api/pipeline/video-info?path=${encodeURIComponent(path)}`);
      if (!res.ok) {
        this._videoInfo.textContent = 'Failed to load video info';
        this._videoMeta = null;
        return;
      }
      const data = await res.json();
      if (data.error) {
        this._videoInfo.textContent = data.error;
        this._videoMeta = null;
        return;
      }
      this._videoMeta = data;

      const suggestedInterval = this._parsePositiveInt(
        data.suggested_frame_interval,
        Math.max(1, Math.round((this._parsePositiveInt(data.fps, 30)) / 2)),
      );
      const suggestedMaxFrames = this._parsePositiveInt(
        data.suggested_max_frames,
        this._estimateMaxFrames(data.total_frames, suggestedInterval),
      );
      this._extractDefaults = {
        frame_interval: suggestedInterval,
        max_frames: suggestedMaxFrames,
      };
      this._inputs.frame_interval.value = String(suggestedInterval);
      this._inputs.max_frames.value = String(suggestedMaxFrames);
      this._maxFramesAuto = true;

      const mins = Math.floor(data.duration / 60);
      const secs = (data.duration % 60).toFixed(1);
      const dur = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
      this._videoInfo.textContent =
        `${data.width}x${data.height} | ${data.fps} fps | ${data.total_frames} frames | ${dur}`;
      this._applySuggestedObjectName();
    } catch {
      this._videoInfo.textContent = 'Failed to load video info';
      this._videoMeta = null;
    }
  }

  _applySuggestedObjectName(force = false) {
    const selectedExisting = this._objectSelect.value && this._objectSelect.value !== NEW_OBJECT_VALUE;
    if (selectedExisting && !force) return;
    if (this._objectNameDirty && !force) return;
    const suggested = this._suggestObjectNameFromVideo();
    if (!suggested) return;
    this._objectNameInput.value = suggested;
    this._objectNameDirty = false;
    this._selectedObjectSummary = null;
    this._renderObjectSummary(null, suggested);
    this._renderArtifacts(null);
    this._updateResumeHint();
  }

  _suggestObjectNameFromVideo() {
    const option = this._videoSelect.selectedOptions?.[0];
    const raw = option?.dataset?.suggestedObjectName || '';
    return this._normalizeObjectName(raw);
  }

  _resolveResumeStage() {
    return this._clampStage(this._startStage);
  }

  _clampStage(stage) {
    const n = Number(stage) || 1;
    return Math.max(1, Math.min(6, Math.round(n)));
  }

  _updateResumeHint() {
    if (!this._resumeStageInfo) return;
    const manualStage = this._resolveResumeStage();
    const manualLabel = STAGE_LABELS[manualStage] || `Stage ${manualStage}`;
    this._resumeStageInfo.textContent =
      `Start Pipeline from selected task: Stage ${manualStage} (${manualLabel}).`;
  }

  _applyObjectVideoPath(path) {
    if (!path) return;
    const target = String(path);
    const matched = Array.from(this._videoSelect.options || []).some((opt) => opt.value === target);
    if (!matched) return;
    if (this._videoSelect.value === target) return;
    this._videoSelect.value = target;
    this._onVideoChange();
  }

  _onFrameIntervalInput() {
    if (!this._videoMeta) return;
    const interval = this._parsePositiveInt(
      this._inputs.frame_interval.value,
      this._extractDefaults.frame_interval,
    );
    this._extractDefaults.frame_interval = interval;
    if (!this._maxFramesAuto) return;
    const maxFrames = this._estimateMaxFrames(this._videoMeta.total_frames, interval);
    this._extractDefaults.max_frames = maxFrames;
    this._inputs.max_frames.value = String(maxFrames);
  }

  _onMaxFramesInput() {
    const raw = (this._inputs.max_frames.value || '').trim();
    if (!raw) {
      this._maxFramesAuto = true;
      this._onFrameIntervalInput();
      return;
    }
    const parsed = Number.parseInt(raw, 10);
    if (Number.isFinite(parsed) && parsed > 0) {
      this._maxFramesAuto = false;
    }
  }

  _estimateMaxFrames(totalFrames, frameInterval) {
    const total = this._parsePositiveInt(totalFrames, 0);
    const interval = this._parsePositiveInt(frameInterval, 1);
    if (total <= 0) return this._extractDefaults.max_frames;
    return Math.max(1, Math.ceil(total / interval));
  }

  _normalizeObjectName(value) {
    const raw = String(value || '').trim();
    if (!raw) return '';
    let normalized = raw
      .replace(/[\\/]/g, '-')
      .replace(/\s+/g, '-')
      .replace(/[^\p{Letter}\p{Number}_.-]/gu, '-')
      .replace(/-+/g, '-')
      .replace(/^[.-]+/, '')
      .replace(/[.-]+$/, '');
    if (!normalized) normalized = 'object';
    return normalized.slice(0, 80);
  }

  _formatSize(sizeMb) {
    if (!Number.isFinite(sizeMb)) return '';
    if (sizeMb < 0.01) return '<0.01 MB';
    return `${Number(sizeMb).toFixed(2)} MB`;
  }

  _parsePositiveInt(value, fallback) {
    const n = Number.parseInt(value, 10);
    if (!Number.isFinite(n) || n <= 0) return fallback;
    return n;
  }
}
