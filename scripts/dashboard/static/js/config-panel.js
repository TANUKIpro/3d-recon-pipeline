/**
 * Configuration panel: video/object selection and parameter inputs.
 */

const NEW_OBJECT_VALUE = '__new__';
const STAGE_LABELS = {
  1: 'Extract Frames',
  2: 'Pi3X',
  3: 'SAM2',
  4: 'Denoise',
  5: 'Mesh Reconstruction',
  6: 'Mesh Wrap + Hole Repair',
  7: 'Texture Bake',
};
const DENOISE_CUSTOM_PRESET = 'custom';
const DENOISE_ALGO_LABELS = {
  dbscan_sor: 'DBSCAN + SOR',
  dbscan_only: 'DBSCAN',
  sor_only: 'SOR',
  radius_only: 'Radius Outlier Removal',
  dbscan_radius: 'DBSCAN + Radius Outlier Removal',
};
const DENOISE_ALGO_STEPS = {
  dbscan_sor: { dbscan: true, sor: true, radius: false },
  dbscan_only: { dbscan: true, sor: false, radius: false },
  sor_only: { dbscan: false, sor: true, radius: false },
  radius_only: { dbscan: false, sor: false, radius: true },
  dbscan_radius: { dbscan: true, sor: false, radius: true },
};
const DENOISE_PRESETS = {
  balanced: {
    denoise_algorithm: 'dbscan_sor',
    denoise_dbscan_eps: 0.0,
    denoise_dbscan_eps_ratio: 0.02,
    denoise_dbscan_min_samples: 10,
    denoise_dbscan_max_points: 500000,
    denoise_sor_neighbors: 20,
    denoise_sor_std_ratio: 2.0,
    denoise_radius_neighbors: 8,
    denoise_radius_radius_ratio: 0.015,
  },
  detail_preserving: {
    denoise_algorithm: 'sor_only',
    denoise_dbscan_eps: 0.0,
    denoise_dbscan_eps_ratio: 0.02,
    denoise_dbscan_min_samples: 10,
    denoise_dbscan_max_points: 500000,
    denoise_sor_neighbors: 16,
    denoise_sor_std_ratio: 2.6,
    denoise_radius_neighbors: 8,
    denoise_radius_radius_ratio: 0.015,
  },
  isolate_subject: {
    denoise_algorithm: 'dbscan_only',
    denoise_dbscan_eps: 0.0,
    denoise_dbscan_eps_ratio: 0.018,
    denoise_dbscan_min_samples: 8,
    denoise_dbscan_max_points: 500000,
    denoise_sor_neighbors: 20,
    denoise_sor_std_ratio: 2.0,
    denoise_radius_neighbors: 8,
    denoise_radius_radius_ratio: 0.015,
  },
  sparse_noise: {
    denoise_algorithm: 'radius_only',
    denoise_dbscan_eps: 0.0,
    denoise_dbscan_eps_ratio: 0.02,
    denoise_dbscan_min_samples: 10,
    denoise_dbscan_max_points: 500000,
    denoise_sor_neighbors: 20,
    denoise_sor_std_ratio: 2.0,
    denoise_radius_neighbors: 6,
    denoise_radius_radius_ratio: 0.012,
  },
  aggressive_cleanup: {
    denoise_algorithm: 'dbscan_radius',
    denoise_dbscan_eps: 0.0,
    denoise_dbscan_eps_ratio: 0.024,
    denoise_dbscan_min_samples: 14,
    denoise_dbscan_max_points: 500000,
    denoise_sor_neighbors: 20,
    denoise_sor_std_ratio: 2.0,
    denoise_radius_neighbors: 10,
    denoise_radius_radius_ratio: 0.02,
  },
};
const MESHWRAP_CUSTOM_PRESET = 'custom';
const MESHWRAP_PRESETS = {
  balanced: {
    meshwrap_poisson_depth: 9,
    meshwrap_poisson_scale: 1.18,
    meshwrap_density_trim_q: 0.06,
    meshwrap_target_face_ratio: 1.80,
    meshwrap_iterations: 2,
    meshwrap_crop_scale: 1.08,
    meshwrap_sample_points: 180000,
    meshwrap_normal_radius_ratio: 0.035,
  },
  quality: {
    meshwrap_poisson_depth: 10,
    meshwrap_poisson_scale: 1.22,
    meshwrap_density_trim_q: 0.08,
    meshwrap_target_face_ratio: 2.50,
    meshwrap_iterations: 3,
    meshwrap_crop_scale: 1.10,
    meshwrap_sample_points: 250000,
    meshwrap_normal_radius_ratio: 0.04,
  },
  fast: {
    meshwrap_poisson_depth: 8,
    meshwrap_poisson_scale: 1.15,
    meshwrap_density_trim_q: 0.05,
    meshwrap_target_face_ratio: 1.50,
    meshwrap_iterations: 1,
    meshwrap_crop_scale: 1.06,
    meshwrap_sample_points: 120000,
    meshwrap_normal_radius_ratio: 0.03,
  },
  minimal: {
    meshwrap_poisson_depth: 8,
    meshwrap_poisson_scale: 1.10,
    meshwrap_density_trim_q: 0.03,
    meshwrap_target_face_ratio: 1.20,
    meshwrap_iterations: 1,
    meshwrap_crop_scale: 1.04,
    meshwrap_sample_points: 100000,
    meshwrap_normal_radius_ratio: 0.025,
  },
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
    this._pi3xFrameTargetValue = document.getElementById('cfg-pi3x-frame-target-value');
    this._pi3xFrameTargetNote = document.getElementById('cfg-pi3x-frame-target-note');
    this._pi3xPlanNote = document.getElementById('cfg-pi3x-plan-note');
    this._refreshObjectsBtn = document.getElementById('btn-refresh-objects');
    this._startBtn = document.getElementById('btn-start');
    this._cancelBtn = document.getElementById('btn-cancel');

    this._inputs = {
      frame_interval: document.getElementById('cfg-frame-interval'),
      max_frames: document.getElementById('cfg-max-frames'),
      pixel_limit: document.getElementById('cfg-pixel-limit'),
      pi3x_frame_target: document.getElementById('cfg-pi3x-frame-target'),
      confidence_threshold: document.getElementById('cfg-conf-threshold'),
      edge_rtol: document.getElementById('cfg-edge-rtol'),
      sam2_model: document.getElementById('cfg-sam2-model'),
      denoise_preset: document.getElementById('cfg-denoise-preset'),
      denoise_algorithm: document.getElementById('cfg-denoise-algorithm'),
      denoise_dbscan_eps: document.getElementById('cfg-denoise-dbscan-eps'),
      denoise_dbscan_eps_ratio: document.getElementById('cfg-denoise-dbscan-eps-ratio'),
      denoise_dbscan_min_samples: document.getElementById('cfg-denoise-dbscan-min-samples'),
      denoise_dbscan_max_points: document.getElementById('cfg-denoise-dbscan-max-points'),
      denoise_sor_neighbors: document.getElementById('cfg-denoise-sor-neighbors'),
      denoise_sor_std_ratio: document.getElementById('cfg-denoise-sor-std'),
      denoise_radius_neighbors: document.getElementById('cfg-denoise-radius-neighbors'),
      denoise_radius_radius_ratio: document.getElementById('cfg-denoise-radius-ratio'),
      mesh_method: document.getElementById('cfg-mesh-method'),
      diffcd_batch_size: document.getElementById('cfg-diffcd-batch'),
      diffcd_n_batches: document.getElementById('cfg-diffcd-nbatches'),
      diffcd_resolution: document.getElementById('cfg-diffcd-res'),
      texture_size: document.getElementById('cfg-texture-size'),
      meshwrap_preset: document.getElementById('cfg-meshwrap-preset'),
      meshwrap_poisson_depth: document.getElementById('cfg-meshwrap-poisson-depth'),
      meshwrap_poisson_scale: document.getElementById('cfg-meshwrap-poisson-scale'),
      meshwrap_density_trim_q: document.getElementById('cfg-meshwrap-density-trim-q'),
      meshwrap_target_face_ratio: document.getElementById('cfg-meshwrap-face-ratio'),
      meshwrap_iterations: document.getElementById('cfg-meshwrap-iterations'),
      meshwrap_crop_scale: document.getElementById('cfg-meshwrap-crop-scale'),
      meshwrap_sample_points: document.getElementById('cfg-meshwrap-sample-points'),
      meshwrap_normal_radius_ratio: document.getElementById('cfg-meshwrap-normal-radius'),
      contact_hole_repair_enabled: document.getElementById('cfg-contact-hole-repair-enabled'),
      contact_hole_max_diameter_ratio: document.getElementById('cfg-contact-hole-max-diameter-ratio'),
      contact_hole_y_band_ratio: document.getElementById('cfg-contact-hole-y-band-ratio'),
      contact_hole_smooth_iters: document.getElementById('cfg-contact-hole-smooth-iters'),
    };
    this._meshwrapSummary = document.getElementById('cfg-meshwrap-summary');
    this._denoiseSummary = document.getElementById('cfg-denoise-summary');
    this._denoiseGroups = {
      dbscan: document.getElementById('cfg-denoise-dbscan'),
      sor: document.getElementById('cfg-denoise-sor'),
      radius: document.getElementById('cfg-denoise-radius'),
    };
    this._meshMethodSummary = document.getElementById('cfg-mesh-method-summary');
    this._poissonSummary = document.getElementById('cfg-poisson-summary');
    this._diffcdControls = document.getElementById('cfg-diffcd-controls');

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
    this._updatingDenoisePreset = false;
    this._updatingMeshWrapPreset = false;
    this._pi3xFrameTargetAuto = true;
    this._pi3xFrameTargetRecommended = null;
    this._pi3xPlanRequestId = 0;
    this._pi3xPlanDebounce = null;

    this._applyDenoisePreset(this._inputs.denoise_preset.value || 'balanced');
    this._applyMeshWrapCustomDefaults();
    this.setMeshMethod(this._inputs.mesh_method?.value || 'poisson');
    this._updateTextureAutoOption(null);
    this._bindEvents();
    this._loadVideos();
    this.refreshObjects();
    this._updateResumeHint();
    this._updateFrameBudgetPreview();
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
      this._updateFrameBudgetPreview();
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
    this._updateFrameBudgetPreview();
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
    this._updateFrameBudgetPreview();
  }

  setMeshMethod(method) {
    const normalized = String(method || '').trim().toLowerCase();
    const resolved = normalized === 'diffcd' ? 'diffcd' : 'poisson';
    if (this._inputs.mesh_method) {
      this._inputs.mesh_method.value = resolved;
    }

    const isDiffcd = resolved === 'diffcd';
    if (this._meshMethodSummary) {
      this._meshMethodSummary.textContent = isDiffcd
        ? 'Learning Mesh (DiffCD) is active.'
        : 'Classical Mesh pipeline is active.';
    }
    if (this._poissonSummary) {
      this._poissonSummary.style.display = isDiffcd ? 'none' : '';
    }
    if (this._diffcdControls) {
      this._diffcdControls.style.display = isDiffcd ? '' : 'none';
    }
  }

  getMeshMethod() {
    return this._inputs.mesh_method?.value || 'poisson';
  }

  getConfig() {
    const suggestedObject = this._suggestObjectNameFromVideo();
    const objectName = this._normalizeObjectName(this._objectNameInput.value || suggestedObject || 'object');
    this._objectNameInput.value = objectName;
    const resumeFromStage = this._resolveResumeStage();
    const maxFrames = this._resolveMaxFrames();
    const pi3xFrameTarget = this._resolvePi3xFrameTarget(maxFrames);

    return {
      video_path: this._videoSelect.value,
      object_name: objectName,
      resume_from_stage: resumeFromStage,
      frame_interval: this._parsePositiveInt(
        this._inputs.frame_interval.value,
        this._extractDefaults.frame_interval,
      ),
      max_frames: maxFrames,
      pixel_limit: parseInt(this._inputs.pixel_limit.value) || 255000,
      pi3x_frame_target: pi3xFrameTarget,
      confidence_threshold: parseFloat(this._inputs.confidence_threshold.value) || 0.2,
      edge_rtol: parseFloat(this._inputs.edge_rtol.value) || 0.03,
      sam2_model: this._inputs.sam2_model.value,
      denoise_preset: this._inputs.denoise_preset.value || 'balanced',
      denoise_algorithm: this._inputs.denoise_algorithm.value || 'dbscan_sor',
      denoise_dbscan_eps: this._parseNonNegativeFloat(this._inputs.denoise_dbscan_eps.value, 0.0),
      denoise_dbscan_eps_ratio: this._parsePositiveFloat(this._inputs.denoise_dbscan_eps_ratio.value, 0.02),
      denoise_dbscan_min_samples: this._parsePositiveInt(this._inputs.denoise_dbscan_min_samples.value, 10),
      denoise_dbscan_max_points: this._parsePositiveInt(this._inputs.denoise_dbscan_max_points.value, 500000),
      denoise_sor_neighbors: this._parsePositiveInt(this._inputs.denoise_sor_neighbors.value, 20),
      denoise_sor_std_ratio: this._parsePositiveFloat(this._inputs.denoise_sor_std_ratio.value, 2.0),
      denoise_radius_neighbors: this._parsePositiveInt(this._inputs.denoise_radius_neighbors.value, 8),
      denoise_radius_radius_ratio: this._parsePositiveFloat(this._inputs.denoise_radius_radius_ratio.value, 0.015),
      mesh_method: this.getMeshMethod(),
      diffcd_batch_size: parseInt(this._inputs.diffcd_batch_size.value) || 5000,
      diffcd_n_batches: parseInt(this._inputs.diffcd_n_batches.value) || 30000,
      diffcd_resolution: parseInt(this._inputs.diffcd_resolution.value) || 512,
      meshwrap_preset: this._inputs.meshwrap_preset?.value || 'custom',
      meshwrap_poisson_depth: this._parsePositiveInt(this._inputs.meshwrap_poisson_depth?.value, 6),
      meshwrap_poisson_scale: this._parsePositiveFloat(this._inputs.meshwrap_poisson_scale?.value, 1.18),
      meshwrap_density_trim_q: this._parseNonNegativeFloat(this._inputs.meshwrap_density_trim_q?.value, 0.06),
      meshwrap_target_face_ratio: this._parsePositiveFloat(this._inputs.meshwrap_target_face_ratio?.value, 1.80),
      meshwrap_iterations: this._parsePositiveInt(this._inputs.meshwrap_iterations?.value, 2),
      meshwrap_crop_scale: this._parsePositiveFloat(this._inputs.meshwrap_crop_scale?.value, 1.08),
      meshwrap_sample_points: this._parsePositiveInt(this._inputs.meshwrap_sample_points?.value, 180000),
      meshwrap_normal_radius_ratio: this._parsePositiveFloat(this._inputs.meshwrap_normal_radius_ratio?.value, 0.035),
      contact_hole_repair_enabled: Boolean(this._inputs.contact_hole_repair_enabled?.checked),
      contact_hole_max_diameter_ratio: this._parsePositiveFloat(this._inputs.contact_hole_max_diameter_ratio?.value, 0.08),
      contact_hole_y_band_ratio: this._parsePositiveFloat(this._inputs.contact_hole_y_band_ratio?.value, 0.06),
      contact_hole_smooth_iters: this._parseNonNegativeInt(this._inputs.contact_hole_smooth_iters?.value, 2),
      texture_size: this._parseTextureSize(this._inputs.texture_size.value, 0),
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
        this._updateFrameBudgetPreview();
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
      this._updateFrameBudgetPreview();
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
    this._inputs.pixel_limit.addEventListener('input', () => this._updateFrameBudgetPreview());
    this._inputs.pi3x_frame_target.addEventListener('input', () => this._onPi3xFrameTargetInput());
    this._inputs.denoise_preset.addEventListener('change', () => {
      const preset = this._inputs.denoise_preset.value;
      if (preset === DENOISE_CUSTOM_PRESET) {
        this._updateDenoiseControls();
        this._updateDenoiseSummary();
        return;
      }
      this._applyDenoisePreset(preset);
    });
    this._inputs.denoise_algorithm.addEventListener('change', () => {
      this._updateDenoiseControls();
      this._onDenoiseInputChanged();
    });
    for (const key of [
      'denoise_dbscan_eps',
      'denoise_dbscan_eps_ratio',
      'denoise_dbscan_min_samples',
      'denoise_dbscan_max_points',
      'denoise_sor_neighbors',
      'denoise_sor_std_ratio',
      'denoise_radius_neighbors',
      'denoise_radius_radius_ratio',
    ]) {
      this._inputs[key].addEventListener('input', () => this._onDenoiseInputChanged());
    }

    if (this._inputs.meshwrap_preset) {
      this._inputs.meshwrap_preset.addEventListener('change', () => {
        const preset = this._inputs.meshwrap_preset.value;
        if (preset === MESHWRAP_CUSTOM_PRESET) {
          this._updateMeshWrapSummary();
          return;
        }
        this._applyMeshWrapPreset(preset);
      });
    }
    for (const key of [
      'meshwrap_poisson_depth',
      'meshwrap_poisson_scale',
      'meshwrap_density_trim_q',
      'meshwrap_target_face_ratio',
      'meshwrap_iterations',
      'meshwrap_crop_scale',
      'meshwrap_sample_points',
      'meshwrap_normal_radius_ratio',
      'contact_hole_max_diameter_ratio',
      'contact_hole_y_band_ratio',
      'contact_hole_smooth_iters',
    ]) {
      if (this._inputs[key]) {
        this._inputs[key].addEventListener('input', () => this._onMeshWrapInputChanged());
      }
    }
    if (this._inputs.contact_hole_repair_enabled) {
      this._inputs.contact_hole_repair_enabled.addEventListener('change', () => this._onMeshWrapInputChanged());
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
      this._updateFrameBudgetPreview();
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
      this._updateFrameBudgetPreview();
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
      opt.textContent = `${o.name} (${o.complete_stages || 0}/7)`;
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
      this._updateFrameBudgetPreview();
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
    this._updateFrameBudgetPreview();
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
      this._updateFrameBudgetPreview();
    } catch (e) {
      // Keep previously rendered summary.
    }
  }

  _renderObjectSummary(object, fallbackName = '') {
    if (object) {
      const details = [
        `${object.complete_stages || 0}/7 stages`,
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
      this._updateFrameBudgetPreview();
      return;
    }
    this._videoInfo.textContent = 'Loading...';
    try {
      const res = await fetch(`/api/pipeline/video-info?path=${encodeURIComponent(path)}`);
      if (!res.ok) {
        this._videoInfo.textContent = 'Failed to load video info';
        this._videoMeta = null;
        this._updateTextureAutoOption(null);
        this._updateFrameBudgetPreview();
        return;
      }
      const data = await res.json();
      if (data.error) {
        this._videoInfo.textContent = data.error;
        this._videoMeta = null;
        this._updateTextureAutoOption(null);
        this._updateFrameBudgetPreview();
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
      this._pi3xFrameTargetAuto = true;
      this._pi3xFrameTargetRecommended = null;

      const mins = Math.floor(data.duration / 60);
      const secs = (data.duration % 60).toFixed(1);
      const dur = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
      this._videoInfo.textContent =
        `${data.width}x${data.height} | ${data.fps} fps | ${data.total_frames} frames | ${dur}`;
      this._applySuggestedObjectName();
      this._updateFrameBudgetPreview();
    } catch {
      this._videoInfo.textContent = 'Failed to load video info';
      this._videoMeta = null;
      this._updateTextureAutoOption(null);
      this._updateFrameBudgetPreview();
    }
  }

  _applyConfig(rawCfg) {
    const cfg = rawCfg || {};
    for (const key of [
      'frame_interval',
      'max_frames',
      'pixel_limit',
      'pi3x_frame_target',
      'confidence_threshold',
      'edge_rtol',
      'diffcd_batch_size',
      'diffcd_n_batches',
      'diffcd_resolution',
      'texture_size',
    ]) {
      if (cfg[key] == null || !this._inputs[key]) continue;
      this._inputs[key].value = String(cfg[key]);
    }
    if (cfg.sam2_model != null) {
      this._setSelectValue(this._inputs.sam2_model, String(cfg.sam2_model));
    }
    this.setMeshMethod(String(cfg.mesh_method || this.getMeshMethod() || 'poisson'));

    const preset = String(cfg.denoise_preset || '');
    if (preset && preset !== DENOISE_CUSTOM_PRESET && DENOISE_PRESETS[preset]) {
      this._applyDenoisePreset(preset);
    } else {
      this._inputs.denoise_preset.value = DENOISE_CUSTOM_PRESET;
    }

    if (cfg.denoise_algorithm != null) {
      this._setSelectValue(this._inputs.denoise_algorithm, String(cfg.denoise_algorithm));
    }
    for (const key of [
      'denoise_dbscan_eps',
      'denoise_dbscan_eps_ratio',
      'denoise_dbscan_min_samples',
      'denoise_dbscan_max_points',
      'denoise_sor_neighbors',
      'denoise_sor_std_ratio',
      'denoise_radius_neighbors',
      'denoise_radius_radius_ratio',
    ]) {
      if (cfg[key] == null || !this._inputs[key]) continue;
      this._inputs[key].value = String(cfg[key]);
    }

    const meshwrapPreset = String(cfg.meshwrap_preset || '');
    if (meshwrapPreset && meshwrapPreset !== MESHWRAP_CUSTOM_PRESET && MESHWRAP_PRESETS[meshwrapPreset]) {
      this._applyMeshWrapPreset(meshwrapPreset);
    } else {
      if (this._inputs.meshwrap_preset) {
        this._inputs.meshwrap_preset.value = MESHWRAP_CUSTOM_PRESET;
      }
    }
    for (const key of [
      'meshwrap_poisson_depth',
      'meshwrap_poisson_scale',
      'meshwrap_density_trim_q',
      'meshwrap_target_face_ratio',
      'meshwrap_iterations',
      'meshwrap_crop_scale',
      'meshwrap_sample_points',
      'meshwrap_normal_radius_ratio',
      'contact_hole_max_diameter_ratio',
      'contact_hole_y_band_ratio',
      'contact_hole_smooth_iters',
    ]) {
      if (cfg[key] != null && this._inputs[key]) {
        this._inputs[key].value = String(cfg[key]);
      }
    }
    if (cfg.contact_hole_repair_enabled != null && this._inputs.contact_hole_repair_enabled) {
      this._inputs.contact_hole_repair_enabled.checked = Boolean(cfg.contact_hole_repair_enabled);
    }

    this._maxFramesAuto = false;
    this._pi3xFrameTargetAuto = cfg.pi3x_frame_target == null;
    this._pi3xFrameTargetRecommended = null;
    this._updateDenoiseControls();
    this._onDenoiseInputChanged();
    this._onMeshWrapInputChanged();
    this._updateFrameBudgetPreview();
  }

  _applyDenoisePreset(presetName) {
    const preset = DENOISE_PRESETS[presetName] || DENOISE_PRESETS.balanced;
    const resolvedName = DENOISE_PRESETS[presetName] ? presetName : 'balanced';
    this._updatingDenoisePreset = true;
    try {
      this._inputs.denoise_preset.value = resolvedName;
      for (const [key, value] of Object.entries(preset)) {
        if (!this._inputs[key]) continue;
        this._inputs[key].value = String(value);
      }
    } finally {
      this._updatingDenoisePreset = false;
    }
    this._updateDenoiseControls();
    this._updateDenoiseSummary();
  }

  _onDenoiseInputChanged() {
    this._updateDenoiseControls();
    if (!this._updatingDenoisePreset) {
      const matched = this._findMatchingDenoisePreset();
      this._inputs.denoise_preset.value = matched || DENOISE_CUSTOM_PRESET;
    }
    this._updateDenoiseSummary();
  }

  _findMatchingDenoisePreset() {
    const current = this._readDenoiseConfig();
    for (const [name, preset] of Object.entries(DENOISE_PRESETS)) {
      if (preset.denoise_algorithm !== current.denoise_algorithm) continue;
      let same = true;
      for (const key of [
        'denoise_dbscan_eps',
        'denoise_dbscan_eps_ratio',
        'denoise_dbscan_min_samples',
        'denoise_dbscan_max_points',
        'denoise_sor_neighbors',
        'denoise_sor_std_ratio',
        'denoise_radius_neighbors',
        'denoise_radius_radius_ratio',
      ]) {
        if (!this._valuesAlmostEqual(current[key], preset[key])) {
          same = false;
          break;
        }
      }
      if (same) return name;
    }
    return null;
  }

  _readDenoiseConfig() {
    return {
      denoise_algorithm: this._inputs.denoise_algorithm.value || 'dbscan_sor',
      denoise_dbscan_eps: this._parseNonNegativeFloat(this._inputs.denoise_dbscan_eps.value, 0.0),
      denoise_dbscan_eps_ratio: this._parsePositiveFloat(this._inputs.denoise_dbscan_eps_ratio.value, 0.02),
      denoise_dbscan_min_samples: this._parsePositiveInt(this._inputs.denoise_dbscan_min_samples.value, 10),
      denoise_dbscan_max_points: this._parsePositiveInt(this._inputs.denoise_dbscan_max_points.value, 500000),
      denoise_sor_neighbors: this._parsePositiveInt(this._inputs.denoise_sor_neighbors.value, 20),
      denoise_sor_std_ratio: this._parsePositiveFloat(this._inputs.denoise_sor_std_ratio.value, 2.0),
      denoise_radius_neighbors: this._parsePositiveInt(this._inputs.denoise_radius_neighbors.value, 8),
      denoise_radius_radius_ratio: this._parsePositiveFloat(this._inputs.denoise_radius_radius_ratio.value, 0.015),
    };
  }

  _updateDenoiseControls() {
    const algo = this._inputs.denoise_algorithm.value || 'dbscan_sor';
    const flags = DENOISE_ALGO_STEPS[algo] || DENOISE_ALGO_STEPS.dbscan_sor;
    this._denoiseGroups.dbscan.hidden = !flags.dbscan;
    this._denoiseGroups.sor.hidden = !flags.sor;
    this._denoiseGroups.radius.hidden = !flags.radius;
  }

  _updateDenoiseSummary() {
    if (!this._denoiseSummary) return;
    const cfg = this._readDenoiseConfig();
    const algo = DENOISE_ALGO_LABELS[cfg.denoise_algorithm] || cfg.denoise_algorithm;
    const flags = DENOISE_ALGO_STEPS[cfg.denoise_algorithm] || DENOISE_ALGO_STEPS.dbscan_sor;

    const details = [];
    if (flags.dbscan) {
      const epsText = cfg.denoise_dbscan_eps > 0
        ? `eps=${cfg.denoise_dbscan_eps.toFixed(4)}`
        : `eps=auto(${cfg.denoise_dbscan_eps_ratio.toFixed(3)}*bbox)`;
      details.push(`DBSCAN ${epsText}, min_samples=${cfg.denoise_dbscan_min_samples}`);
    }
    if (flags.sor) {
      details.push(`SOR k=${cfg.denoise_sor_neighbors}, std=${cfg.denoise_sor_std_ratio.toFixed(2)}`);
    }
    if (flags.radius) {
      details.push(`Radius k=${cfg.denoise_radius_neighbors}, ratio=${cfg.denoise_radius_radius_ratio.toFixed(3)}`);
    }

    this._denoiseSummary.textContent = `Pipeline: ${algo}${details.length ? ` | ${details.join(' | ')}` : ''}`;
  }

  _applyMeshWrapCustomDefaults() {
    const customDefaults = {
      meshwrap_poisson_depth: 6,
      meshwrap_poisson_scale: 1.18,
      meshwrap_density_trim_q: 0.06,
      meshwrap_target_face_ratio: 1.80,
      meshwrap_iterations: 2,
      meshwrap_crop_scale: 2.0,
      meshwrap_sample_points: 180000,
      meshwrap_normal_radius_ratio: 0.035,
    };
    this._updatingMeshWrapPreset = true;
    try {
      if (this._inputs.meshwrap_preset) {
        this._inputs.meshwrap_preset.value = MESHWRAP_CUSTOM_PRESET;
      }
      for (const [key, value] of Object.entries(customDefaults)) {
        if (!this._inputs[key]) continue;
        this._inputs[key].value = String(value);
      }
    } finally {
      this._updatingMeshWrapPreset = false;
    }
    this._updateMeshWrapSummary();
  }

  _applyMeshWrapPreset(presetName) {
    const preset = MESHWRAP_PRESETS[presetName] || MESHWRAP_PRESETS.balanced;
    const resolvedName = MESHWRAP_PRESETS[presetName] ? presetName : 'balanced';
    this._updatingMeshWrapPreset = true;
    try {
      if (this._inputs.meshwrap_preset) {
        this._inputs.meshwrap_preset.value = resolvedName;
      }
      for (const [key, value] of Object.entries(preset)) {
        if (!this._inputs[key]) continue;
        this._inputs[key].value = String(value);
      }
    } finally {
      this._updatingMeshWrapPreset = false;
    }
    this._updateMeshWrapSummary();
  }

  _onMeshWrapInputChanged() {
    if (!this._updatingMeshWrapPreset) {
      const matched = this._findMatchingMeshWrapPreset();
      if (this._inputs.meshwrap_preset) {
        this._inputs.meshwrap_preset.value = matched || MESHWRAP_CUSTOM_PRESET;
      }
    }
    this._updateMeshWrapSummary();
  }

  _findMatchingMeshWrapPreset() {
    const current = {};
    for (const key of [
      'meshwrap_poisson_depth',
      'meshwrap_poisson_scale',
      'meshwrap_density_trim_q',
      'meshwrap_target_face_ratio',
      'meshwrap_iterations',
      'meshwrap_crop_scale',
      'meshwrap_sample_points',
      'meshwrap_normal_radius_ratio',
    ]) {
      current[key] = this._inputs[key] ? Number(this._inputs[key].value) : 0;
    }
    for (const [name, preset] of Object.entries(MESHWRAP_PRESETS)) {
      let same = true;
      for (const key of Object.keys(preset)) {
        if (!this._valuesAlmostEqual(current[key], preset[key])) {
          same = false;
          break;
        }
      }
      if (same) return name;
    }
    return null;
  }

  _updateMeshWrapSummary() {
    if (!this._meshwrapSummary) return;
    const depth = this._inputs.meshwrap_poisson_depth?.value || '9';
    const scale = this._inputs.meshwrap_poisson_scale?.value || '1.18';
    const iters = this._inputs.meshwrap_iterations?.value || '2';
    const ratio = this._inputs.meshwrap_target_face_ratio?.value || '1.80';
    const holeEnabled = this._inputs.contact_hole_repair_enabled?.checked ? 'on' : 'off';
    const holeDiameter = this._inputs.contact_hole_max_diameter_ratio?.value || '0.08';
    const holeBand = this._inputs.contact_hole_y_band_ratio?.value || '0.06';
    this._meshwrapSummary.textContent =
      `depth=${depth}, scale=${scale}, iters=${iters}, face_ratio=${ratio}, hole_repair=${holeEnabled}, hole_diam=${holeDiameter}, hole_band=${holeBand}`;
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
    return Math.max(1, Math.min(7, Math.round(n)));
  }

  _updateResumeHint() {
    if (!this._resumeStageInfo) return;
    const manualStage = this._resolveResumeStage();
    const manualLabel = STAGE_LABELS[manualStage] || `Stage ${manualStage}`;
    this._resumeStageInfo.textContent =
      `Start Pipeline from selected task: Stage ${manualStage} (${manualLabel}).`;
  }

  async _applyObjectVideoPath(path) {
    if (!path) return;
    const target = String(path);
    const matched = Array.from(this._videoSelect.options || []).some((opt) => opt.value === target);
    if (!matched) return;
    if (this._videoSelect.value === target) return;
    this._videoSelect.value = target;
    await this._onVideoChange();
  }

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
    this._updateFrameBudgetPreview();
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
    this._updateFrameBudgetPreview();
  }

  _onPi3xFrameTargetInput() {
    this._pi3xFrameTargetAuto = false;
    this._syncPi3xFrameTargetInput();
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

  _clampPi3xFrameTarget(targetFrames, maxFrames) {
    const maxAllowed = Math.max(2, this._parsePositiveInt(maxFrames, this._extractDefaults.max_frames));
    const parsed = this._parsePositiveInt(targetFrames, maxAllowed);
    return Math.max(2, Math.min(parsed, maxAllowed));
  }

  _resolvePi3xFrameTarget(maxFrames = this._resolveMaxFrames()) {
    const fallback = this._clampPi3xFrameTarget(
      this._pi3xFrameTargetRecommended ?? maxFrames,
      maxFrames,
    );
    const parsed = this._parsePositiveInt(this._inputs.pi3x_frame_target.value, fallback);
    return this._clampPi3xFrameTarget(parsed, maxFrames);
  }

  _syncPi3xFrameTargetInput(opts = {}) {
    const maxFrames = this._resolveMaxFrames();
    const recommendedRaw = opts.recommendedFrames;
    if (recommendedRaw != null) {
      this._pi3xFrameTargetRecommended = this._clampPi3xFrameTarget(recommendedRaw, maxFrames);
    } else if (this._pi3xFrameTargetRecommended == null) {
      this._pi3xFrameTargetRecommended = maxFrames;
    } else {
      this._pi3xFrameTargetRecommended = this._clampPi3xFrameTarget(this._pi3xFrameTargetRecommended, maxFrames);
    }

    this._inputs.pi3x_frame_target.min = '2';
    this._inputs.pi3x_frame_target.max = String(maxFrames);

    const fallback = this._pi3xFrameTargetRecommended;
    let selectedFrames = this._resolvePi3xFrameTarget(maxFrames);
    if (this._pi3xFrameTargetAuto || !(this._inputs.pi3x_frame_target.value || '').trim()) {
      selectedFrames = fallback;
      this._inputs.pi3x_frame_target.value = String(selectedFrames);
    } else {
      this._inputs.pi3x_frame_target.value = String(selectedFrames);
    }

    if (this._pi3xFrameTargetValue) {
      this._pi3xFrameTargetValue.textContent = String(selectedFrames);
    }
    if (this._pi3xFrameTargetNote) {
      this._pi3xFrameTargetNote.textContent =
        `AutoTarget recommendation: ${fallback} frames (max ${maxFrames})`;
    }

    return {
      requestedFrames: maxFrames,
      selectedFrames,
      recommendedFrames: fallback,
    };
  }

  _resolveRequestedPi3xFrames() {
    return this._resolveMaxFrames();
  }

  _updateFrameBudgetPreview() {
    const requestedFrames = this._resolveRequestedPi3xFrames();
    this._syncPi3xFrameTargetInput();
    if (!this._pi3xPlanNote) return;
    const pixelLimit = this._parsePositiveInt(this._inputs.pixel_limit.value, 255000);
    this._pi3xPlanNote.textContent = 'Pi3X VRAM plan: estimating...';
    if (this._pi3xPlanDebounce) {
      clearTimeout(this._pi3xPlanDebounce);
    }
    this._pi3xPlanDebounce = setTimeout(() => {
      this._updatePi3xPlanPreview(requestedFrames, pixelLimit);
    }, 120);
  }

  async _updatePi3xPlanPreview(requestedFrames, pixelLimit) {
    const reqId = ++this._pi3xPlanRequestId;
    try {
      const res = await fetch(
        `/api/pipeline/pi3x-plan?requested_frames=${requestedFrames}&pixel_limit=${pixelLimit}`,
      );
      if (reqId !== this._pi3xPlanRequestId) return;
      if (!res.ok) {
        this._syncPi3xFrameTargetInput({ recommendedFrames: requestedFrames });
        this._pi3xPlanNote.textContent = 'Pi3X VRAM plan: unavailable';
        return;
      }
      const data = await res.json();
      const autoFrames = this._parsePositiveInt(data.auto_target_frames, requestedFrames);
      const synced = this._syncPi3xFrameTargetInput({ recommendedFrames: autoFrames });
      const targetPct = Number(data.target_vram_utilization) * 100;
      const usedPct = Number(data.predicted_used_pct);
      const parts = [`Auto target: ${synced.recommendedFrames}/${synced.requestedFrames} frames`];
      if (synced.selectedFrames !== synced.recommendedFrames) {
        parts.push(`selected ${synced.selectedFrames}`);
      }
      if (Number.isFinite(targetPct) && targetPct > 0) {
        parts.push(`target VRAM ${targetPct.toFixed(0)}%`);
      }
      if (Number.isFinite(usedPct) && usedPct > 0) {
        parts.push(`estimated ${usedPct.toFixed(1)}%`);
      }
      if (data.auto_reduced) {
        parts.push('auto-reduced');
      }
      this._pi3xPlanNote.textContent = parts.join(' | ');
    } catch {
      if (reqId !== this._pi3xPlanRequestId) return;
      this._syncPi3xFrameTargetInput({ recommendedFrames: requestedFrames });
      this._pi3xPlanNote.textContent = 'Pi3X VRAM plan: unavailable';
    }
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

  _setSelectValue(select, value) {
    if (!select) return;
    if (!Array.from(select.options || []).some(opt => opt.value === value)) return;
    select.value = value;
  }

  _valuesAlmostEqual(a, b) {
    const aNum = Number(a);
    const bNum = Number(b);
    if (!Number.isFinite(aNum) || !Number.isFinite(bNum)) return false;
    return Math.abs(aNum - bNum) <= 1e-9;
  }

  _parseTextureSize(value, fallback = 0) {
    const n = Number.parseInt(value, 10);
    if (!Number.isFinite(n)) return fallback;
    if (n <= 0) return 0;
    return n;
  }

  _computeAutoTextureSize(width, height) {
    const w = Number.parseInt(width, 10);
    const h = Number.parseInt(height, 10);
    if (!Number.isFinite(w) || !Number.isFinite(h) || w <= 0 || h <= 0) {
      return null;
    }
    return Math.max(1, Math.round(Math.sqrt(w * h)));
  }

  _updateTextureAutoOption(videoMeta) {
    const textureInput = this._inputs.texture_size;
    if (!textureInput) return;
    const autoOption = Array.from(textureInput.options || []).find((opt) => opt.value === '0');
    if (!autoOption) return;
    const autoSize = this._computeAutoTextureSize(videoMeta?.width, videoMeta?.height);
    autoOption.textContent = autoSize == null ? 'Auto' : `Auto (~${autoSize})`;
  }

  _parsePositiveInt(value, fallback) {
    const n = Number.parseInt(value, 10);
    if (!Number.isFinite(n) || n <= 0) return fallback;
    return n;
  }

  _parseNonNegativeInt(value, fallback) {
    const n = Number.parseInt(value, 10);
    if (!Number.isFinite(n) || n < 0) return fallback;
    return n;
  }

  _parsePositiveFloat(value, fallback) {
    const n = Number.parseFloat(value);
    if (!Number.isFinite(n) || n <= 0) return fallback;
    return n;
  }

  _parseNonNegativeFloat(value, fallback) {
    const n = Number.parseFloat(value);
    if (!Number.isFinite(n) || n < 0) return fallback;
    return n;
  }
}
