/**
 * Configuration panel: video/object selection and parameter inputs.
 *
 * 6-stage gs2mesh pipeline:
 *   1. Extract Frames
 *   2. COLMAP SfM
 *   3. SAM2 Segmentation
 *   4. gs2mesh Reconstruction
 *   5. Texture Bake
 *   6. Post-texture Cleanup
 */

import {
  GS2MESH_PRESET_CUSTOM,
  GS2MESH_PRESET_DEFAULT,
  GS2MESH_PUBLIC_PRESETS,
  NEW_OBJECT_VALUE,
  STAGE_LABELS,
} from './config/presets.js';
import * as FormHelpers from './config/form-helpers.js';

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
      // Stage 1: Extract Frames
      frame_interval: document.getElementById('cfg-frame-interval'),
      max_frames: document.getElementById('cfg-max-frames'),

      // Stage 2: COLMAP SfM
      colmap_matcher: document.getElementById('cfg-colmap-matcher'),
      colmap_max_features: document.getElementById('cfg-colmap-max-features'),
      colmap_image_size: document.getElementById('cfg-colmap-image-size'),
      colmap_use_gpu: document.getElementById('cfg-colmap-use-gpu'),
      colmap_dsp_sift: document.getElementById('cfg-colmap-dsp-sift'),
      colmap_first_octave: document.getElementById('cfg-colmap-first-octave'),

      // Stage 3: SAM2 Segmentation
      sam2_model: document.getElementById('cfg-sam2-model'),
      ground_plane_enabled: document.getElementById('cfg-ground-plane-enabled'),

      // Stage 4: gs2mesh Reconstruction
      gs2mesh_preset: document.getElementById('cfg-gs2mesh-preset'),
      gs2mesh_gs_iterations: document.getElementById('cfg-gs2mesh-gs-iterations'),
      gs2mesh_runtime_profile: document.getElementById('cfg-gs2mesh-runtime-profile'),
      gs2mesh_stereo_model: document.getElementById('cfg-gs2mesh-stereo-model'),
      gs2mesh_tsdf_voxel_size: document.getElementById('cfg-gs2mesh-tsdf-voxel-size'),
      gs2mesh_tsdf_depth_trunc: document.getElementById('cfg-gs2mesh-tsdf-depth-trunc'),
      gs2mesh_use_masks: document.getElementById('cfg-gs2mesh-use-masks'),

      // Stage 5: Texture Bake
      texture_mode: document.getElementById('cfg-texture-mode'),
      texture_size: document.getElementById('cfg-texture-size'),
      texture_view_assign_mode: document.getElementById('cfg-texture-view-assign-mode'),
      texture_quality_boost: document.getElementById('cfg-texture-quality-boost'),

      // Stage 6: Post-texture Cleanup
      post_texture_cleanup_enabled: document.getElementById('cfg-post-texture-cleanup-enabled'),
      cleanup_lower_half_threshold: document.getElementById('cfg-cleanup-lower-half-threshold'),
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
    this.currentBranchSlug = null;
    this._suppressGs2meshPresetSync = false;
    this._gs2meshPresetBase = GS2MESH_PRESET_DEFAULT;

    this._updateTextureAutoOption(null);
    this._syncTextureModeVisibility();
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
      if (inp) inp.disabled = running;
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
    const maxFrames = this._resolveMaxFrames();

    return {
      video_path: this._videoSelect.value,
      object_name: objectName,
      resume_from_stage: resumeFromStage,

      // Stage 1: Extract Frames
      frame_interval: this._parsePositiveInt(
        this._inputs.frame_interval.value,
        this._extractDefaults.frame_interval,
      ),
      max_frames: maxFrames,

      // Stage 2: COLMAP SfM
      colmap_matcher: this._inputs.colmap_matcher?.value || 'exhaustive',
      colmap_max_features: this._parsePositiveInt(
        this._inputs.colmap_max_features?.value,
        32768,
      ),
      colmap_image_size: this._parsePositiveInt(
        this._inputs.colmap_image_size?.value,
        2048,
      ),
      colmap_use_gpu: this._inputs.colmap_use_gpu?.checked ?? false,
      colmap_dsp_sift: this._inputs.colmap_dsp_sift?.checked ?? false,
      colmap_first_octave: (this._inputs.colmap_first_octave?.checked ?? true) ? -1 : 0,

      // Stage 3: SAM2 Segmentation
      sam2_model: this._inputs.sam2_model.value,
      ground_plane_enabled: this._inputs.ground_plane_enabled?.checked ?? true,

      // Stage 4: gs2mesh Reconstruction
      gs2mesh_preset: this._inputs.gs2mesh_preset?.value || GS2MESH_PRESET_DEFAULT,
      gs2mesh_preset_base: this._gs2meshPresetBase,
      gs2mesh_gs_iterations: this._parsePositiveInt(
        this._inputs.gs2mesh_gs_iterations?.value,
        5000,
      ),
      gs2mesh_runtime_profile: this._inputs.gs2mesh_runtime_profile?.value || 'auto',
      gs2mesh_stereo_model: this._inputs.gs2mesh_stereo_model?.value || 'DLNR',
      gs2mesh_tsdf_voxel_size: this._parsePositiveFloat(
        this._inputs.gs2mesh_tsdf_voxel_size?.value,
        0.005,
      ),
      gs2mesh_tsdf_depth_trunc: this._parsePositiveFloat(
        this._inputs.gs2mesh_tsdf_depth_trunc?.value,
        0.04,
      ),
      gs2mesh_use_masks: this._inputs.gs2mesh_use_masks?.checked ?? true,

      // Stage 5: Texture Bake
      texture_mode: this._inputs.texture_mode?.value || 'multi_view',
      texture_size: this._parseTextureSize(this._inputs.texture_size.value, 0),
      texture_view_assign_mode: this._inputs.texture_view_assign_mode?.value || 'legacy',
      texture_quality_boost: Boolean(this._inputs.texture_quality_boost?.checked),

      // Stage 6: Post-texture Cleanup
      post_texture_cleanup_enabled: this._inputs.post_texture_cleanup_enabled?.checked ?? true,
      cleanup_lower_half_threshold: this._parseNonNegativeFloat(
        this._inputs.cleanup_lower_half_threshold?.value,
        0.2,
      ),
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
        this._updateTextureAutoOption(null);
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
      this._updateTextureAutoOption(null);
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

    if (this._inputs.cleanup_lower_half_threshold) {
      this._inputs.cleanup_lower_half_threshold.addEventListener('input', () => {
        const display = document.getElementById('cfg-cleanup-lower-half-threshold-value');
        if (display) display.textContent = this._inputs.cleanup_lower_half_threshold.value;
      });
    }

    if (this._inputs.texture_mode) {
      this._inputs.texture_mode.addEventListener('change', () => this._syncTextureModeVisibility());
    }

    if (this._inputs.gs2mesh_preset) {
      this._inputs.gs2mesh_preset.addEventListener('change', () => {
        if (this._suppressGs2meshPresetSync) return;
        const preset = this._inputs.gs2mesh_preset.value || GS2MESH_PRESET_DEFAULT;
        this._applyGs2meshPreset(preset);
      });
    }
    for (const key of [
      'gs2mesh_gs_iterations',
      'gs2mesh_runtime_profile',
      'gs2mesh_stereo_model',
      'gs2mesh_tsdf_voxel_size',
      'gs2mesh_tsdf_depth_trunc',
      'gs2mesh_use_masks',
    ]) {
      const input = this._inputs[key];
      if (!input) continue;
      input.addEventListener('input', () => this._markGs2meshPresetCustom());
      input.addEventListener('change', () => this._markGs2meshPresetCustom());
    }

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

    // Group: current branch (selectable) first, then cross-branch (disabled)
    const current = objects.filter(o => !o.locked);
    const locked = objects.filter(o => o.locked);

    for (const o of current) {
      const opt = document.createElement('option');
      opt.value = o.name;
      opt.textContent = `${o.name} (${o.complete_stages || 0}/6)`;
      this._objectSelect.appendChild(opt);
    }

    if (locked.length > 0) {
      const sep = document.createElement('option');
      sep.disabled = true;
      sep.textContent = '\u2500\u2500 Other Branches \u2500\u2500';
      this._objectSelect.appendChild(sep);

      for (const o of locked) {
        const opt = document.createElement('option');
        opt.value = `locked:${o.branch}:${o.name}`;
        opt.textContent = `[${o.branch}] ${o.name} (${o.complete_stages || 0}/6)`;
        opt.disabled = true;
        this._objectSelect.appendChild(opt);
      }
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
        await this._applyObjectVideoPath(object.video_path);
      }
      if (object.config && typeof object.config === 'object') {
        this._applyConfig(object.config);
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
      this._updateTextureAutoOption(null);
      this._applySuggestedObjectName();
      return;
    }
    this._videoInfo.textContent = 'Loading...';
    try {
      const res = await fetch(`/api/pipeline/video-info?path=${encodeURIComponent(path)}`);
      if (!res.ok) {
        this._videoInfo.textContent = 'Failed to load video info';
        this._videoMeta = null;
        this._updateTextureAutoOption(null);
        return;
      }
      const data = await res.json();
      if (data.error) {
        this._videoInfo.textContent = data.error;
        this._videoMeta = null;
        this._updateTextureAutoOption(null);
        return;
      }
      this._videoMeta = data;
      this._updateTextureAutoOption(data);

      const suggestedInterval = this._parsePositiveInt(
        data.suggested_frame_interval,
        Math.max(1, Math.round((this._parsePositiveInt(data.fps, 30)) / 2)),
      );
      const suggestedMaxFrames = this._parsePositiveInt(
        data.suggested_max_frames,
        this._estimateMaxFrames(data.total_frames, suggestedInterval),
      );
      const clampedSuggestedMaxFrames = Math.max(2, suggestedMaxFrames);
      this._extractDefaults = {
        frame_interval: suggestedInterval,
        max_frames: clampedSuggestedMaxFrames,
      };
      this._inputs.frame_interval.value = String(suggestedInterval);
      this._inputs.max_frames.value = String(clampedSuggestedMaxFrames);
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
      this._updateTextureAutoOption(null);
    }
  }

  _applyConfig(rawCfg) {
    const cfg = rawCfg || {};

    // Stage 1: Extract Frames
    for (const key of ['frame_interval', 'max_frames']) {
      if (cfg[key] == null || !this._inputs[key]) continue;
      this._inputs[key].value = String(cfg[key]);
    }

    // Stage 2: COLMAP SfM
    if (cfg.colmap_matcher != null) {
      this._setSelectValue(this._inputs.colmap_matcher, String(cfg.colmap_matcher));
    }
    for (const key of ['colmap_max_features', 'colmap_image_size']) {
      if (cfg[key] == null || !this._inputs[key]) continue;
      this._inputs[key].value = String(cfg[key]);
    }
    if (cfg.colmap_use_gpu != null && this._inputs.colmap_use_gpu) {
      this._inputs.colmap_use_gpu.checked = cfg.colmap_use_gpu !== false;
    }
    if (cfg.colmap_dsp_sift != null && this._inputs.colmap_dsp_sift) {
      this._inputs.colmap_dsp_sift.checked = cfg.colmap_dsp_sift !== false;
    }
    if (cfg.colmap_first_octave != null && this._inputs.colmap_first_octave) {
      this._inputs.colmap_first_octave.checked = cfg.colmap_first_octave < 0;
    }

    // Stage 3: SAM2 Segmentation
    if (cfg.sam2_model != null) {
      this._setSelectValue(this._inputs.sam2_model, String(cfg.sam2_model));
    }
    if (cfg.ground_plane_enabled != null && this._inputs.ground_plane_enabled) {
      this._inputs.ground_plane_enabled.checked = cfg.ground_plane_enabled !== false;
    }

    // Stage 4: gs2mesh Reconstruction
    this._suppressGs2meshPresetSync = true;
    if (cfg.gs2mesh_preset != null && this._inputs.gs2mesh_preset) {
      this._setSelectValue(this._inputs.gs2mesh_preset, String(cfg.gs2mesh_preset));
    } else if (this._inputs.gs2mesh_preset) {
      this._inputs.gs2mesh_preset.value = GS2MESH_PRESET_DEFAULT;
    }
    const presetBase = String(
      cfg.gs2mesh_preset_base
      || (cfg.gs2mesh_preset && cfg.gs2mesh_preset !== GS2MESH_PRESET_CUSTOM
        ? cfg.gs2mesh_preset
        : GS2MESH_PRESET_DEFAULT)
    );
    this._gs2meshPresetBase = Object.prototype.hasOwnProperty.call(GS2MESH_PUBLIC_PRESETS, presetBase)
      ? presetBase
      : GS2MESH_PRESET_DEFAULT;
    for (const key of [
      'gs2mesh_gs_iterations',
      'gs2mesh_tsdf_voxel_size',
      'gs2mesh_tsdf_depth_trunc',
    ]) {
      if (cfg[key] == null || !this._inputs[key]) continue;
      this._inputs[key].value = String(cfg[key]);
    }
    if (cfg.gs2mesh_stereo_model != null) {
      const stereoModel = String(cfg.gs2mesh_stereo_model) === 'DLNR_Middlebury'
        ? 'DLNR'
        : String(cfg.gs2mesh_stereo_model);
      this._setSelectValue(this._inputs.gs2mesh_stereo_model, stereoModel);
    }
    if (cfg.gs2mesh_runtime_profile != null && this._inputs.gs2mesh_runtime_profile) {
      this._setSelectValue(this._inputs.gs2mesh_runtime_profile, String(cfg.gs2mesh_runtime_profile));
    }
    if (cfg.gs2mesh_use_masks != null && this._inputs.gs2mesh_use_masks) {
      this._inputs.gs2mesh_use_masks.checked = cfg.gs2mesh_use_masks !== false;
    }
    this._suppressGs2meshPresetSync = false;

    // Stage 5: Texture Bake
    if (cfg.texture_mode != null && this._inputs.texture_mode) {
      this._setSelectValue(this._inputs.texture_mode, String(cfg.texture_mode));
      this._syncTextureModeVisibility();
    }
    if (cfg.texture_size != null && this._inputs.texture_size) {
      this._inputs.texture_size.value = String(cfg.texture_size);
    }
    if (cfg.texture_view_assign_mode != null) {
      this._setSelectValue(this._inputs.texture_view_assign_mode, String(cfg.texture_view_assign_mode));
    }
    if (cfg.texture_quality_boost != null && this._inputs.texture_quality_boost) {
      this._inputs.texture_quality_boost.checked = Boolean(cfg.texture_quality_boost);
    }
    if (cfg.post_texture_cleanup_enabled != null && this._inputs.post_texture_cleanup_enabled) {
      this._inputs.post_texture_cleanup_enabled.checked = cfg.post_texture_cleanup_enabled !== false;
    }
    if (cfg.cleanup_lower_half_threshold != null && this._inputs.cleanup_lower_half_threshold) {
      this._inputs.cleanup_lower_half_threshold.value = String(cfg.cleanup_lower_half_threshold);
      const display = document.getElementById('cfg-cleanup-lower-half-threshold-value');
      if (display) display.textContent = String(cfg.cleanup_lower_half_threshold);
    }

    this._maxFramesAuto = false;
  }

  // ── Frame extraction helpers ─────────────────────────────────────

  _onFrameIntervalInput() {
    if (this._videoMeta) {
      const interval = this._parsePositiveInt(
        this._inputs.frame_interval.value,
        this._extractDefaults.frame_interval,
      );
      this._extractDefaults.frame_interval = interval;
      if (this._maxFramesAuto) {
        const maxFrames = this._estimateMaxFrames(this._videoMeta.total_frames, interval);
        this._extractDefaults.max_frames = maxFrames;
        this._inputs.max_frames.value = String(maxFrames);
      }
    }
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
    return Math.max(2, Math.ceil(total / interval));
  }

  _resolveMaxFrames() {
    return Math.max(
      2,
      this._parsePositiveInt(
        this._inputs.max_frames.value,
        this._extractDefaults.max_frames,
      ),
    );
  }

  _applyGs2meshPreset(preset) {
    const values = GS2MESH_PUBLIC_PRESETS[preset];
    if (!values) return;
    this._suppressGs2meshPresetSync = true;
    this._gs2meshPresetBase = preset;
    for (const [key, value] of Object.entries(values)) {
      const input = this._inputs[key];
      if (!input) continue;
      if (input.type === 'checkbox') {
        input.checked = Boolean(value);
      } else {
        input.value = String(value);
      }
    }
    if (this._inputs.gs2mesh_preset) {
      this._inputs.gs2mesh_preset.value = preset;
    }
    this._suppressGs2meshPresetSync = false;
  }

  _markGs2meshPresetCustom() {
    if (this._suppressGs2meshPresetSync || !this._inputs.gs2mesh_preset) return;
    this._inputs.gs2mesh_preset.value = GS2MESH_PRESET_CUSTOM;
  }
}

// ── Mixin: form helpers (from config/form-helpers.js) ─────────────
Object.assign(ConfigPanel.prototype, {
  _normalizeObjectName: FormHelpers._normalizeObjectName,
  _formatSize: FormHelpers._formatSize,
  _setSelectValue: FormHelpers._setSelectValue,
  _valuesAlmostEqual: FormHelpers._valuesAlmostEqual,
  _parseTextureSize: FormHelpers._parseTextureSize,
  _computeAutoTextureSize: FormHelpers._computeAutoTextureSize,
  _updateTextureAutoOption: FormHelpers._updateTextureAutoOption,
  _applySuggestedObjectName: FormHelpers._applySuggestedObjectName,
  _suggestObjectNameFromVideo: FormHelpers._suggestObjectNameFromVideo,
  _resolveResumeStage: FormHelpers._resolveResumeStage,
  _clampStage: FormHelpers._clampStage,
  _updateResumeHint: FormHelpers._updateResumeHint,
  _applyObjectVideoPath: FormHelpers._applyObjectVideoPath,
  _parsePositiveInt: FormHelpers._parsePositiveInt,
  _parseNonNegativeInt: FormHelpers._parseNonNegativeInt,
  _parsePositiveFloat: FormHelpers._parsePositiveFloat,
  _parseNonNegativeFloat: FormHelpers._parseNonNegativeFloat,

  _syncTextureModeVisibility() {
    const mode = this._inputs.texture_mode?.value || 'multi_view';
    const section = this._inputs.texture_mode?.closest('.config-section');
    if (!section) return;
    for (const group of section.querySelectorAll('[data-texture-mode]')) {
      group.style.display = group.dataset.textureMode === mode ? '' : 'none';
    }
  },
});
