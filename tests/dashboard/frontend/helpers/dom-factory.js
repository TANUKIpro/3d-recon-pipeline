/**
 * Minimal DOM builders for each module under test.
 * Each function injects the minimum HTML into document.body.
 */

export function buildI18nDOM() {
  document.body.innerHTML = `
    <h1 data-i18n="header.title">clip2mesh</h1>
    <p data-i18n="header.subtitle">RGB Video → Textured 3D Mesh</p>
    <span data-i18n="stage.extract_frames">Extract Frames</span>
  `;
}

export function buildLogViewerDOM() {
  document.body.innerHTML = `
    <div id="log-container" style="height:300px;overflow:auto;">
      <div id="log-content"></div>
    </div>
    <button id="log-clear">Clear</button>
  `;
}

export function buildCheckpointPanelDOM() {
  document.body.innerHTML = `
    <div id="checkpoint-panel">
      <div id="checkpoint-stage-label"></div>
      <ul id="checkpoint-list"></ul>
    </div>
  `;
}

export function buildPipelineDOM() {
  let pills = '';
  for (let i = 1; i <= 8; i++) {
    pills += `
      <div class="stage-pill" data-stage="${i}">
        <span class="stage-progress">0%</span>
        <span class="stage-time"></span>
      </div>
    `;
  }

  // Main connectors (7 between 8 stages)
  let connectors = '';
  for (let i = 0; i < 7; i++) {
    connectors += `<div class="stage-connector-main"></div>`;
  }

  document.body.innerHTML = `
    ${pills}
    ${connectors}
    <div id="mesh-branch-slot">
      <div id="mesh-pill-poisson" class="mesh-method-pill">Poisson</div>
      <div id="mesh-pill-diffcd" class="mesh-method-pill">DiffCD</div>
    </div>
    <div id="mesh-connector-poisson-left"></div>
    <div id="mesh-connector-poisson-right"></div>
    <div id="mesh-connector-diffcd-left"></div>
    <div id="mesh-connector-diffcd-right"></div>
  `;
}

export function buildStageControllerDOM() {
  let pills = '';
  let panels = '';
  for (let i = 1; i <= 8; i++) {
    pills += `<div class="stage-pill" data-stage="${i}" tabindex="0"></div>`;
    panels += `<div id="stage-panel-${i}" class="stage-panel"></div>`;
  }

  document.body.innerHTML = `
    ${pills}
    ${panels}
    <div class="mesh-stage-pill mesh-method-pill active" data-stage="5"></div>
    <div class="mesh-stage-pill" data-stage="5-alt"></div>
  `;
}

export function buildSettingsPanelDOM() {
  document.body.innerHTML = `
    <button id="settings-btn">Settings</button>
  `;
}

export function buildSAM2VerificationDOM() {
  document.body.innerHTML = `
    <div id="sam2-verification" class="hidden">
      <div id="sam2-verification-strip"></div>
      <button id="sam2-approve">Approve & Continue</button>
      <button id="sam2-redo">Redo</button>
    </div>
  `;
}

export function buildSAM2CanvasDOM() {
  document.body.innerHTML = `
    <div id="sam2-placeholder">Click to start</div>
    <canvas id="sam2-canvas" style="display:none"></canvas>
    <button id="sam2-undo" disabled>Undo</button>
    <button id="sam2-clear" disabled>Clear</button>
    <button id="sam2-confirm" disabled>Confirm & Propagate</button>
    <span id="sam2-click-info">Positive: 0  Negative: 0</span>
  `;
}

export function buildMeshRepairDOM() {
  document.body.innerHTML = `
    <div id="mesh-repair-toolbar" class="hidden">
      <span id="mesh-repair-status"></span>
      <input type="range" id="mesh-repair-threshold" min="-1" max="0" step="0.01" value="-0.25">
      <span id="mesh-repair-threshold-value"></span>
      <button id="mesh-repair-clear">Clear</button>
      <button id="mesh-repair-apply">Apply</button>
    </div>
  `;
}

export function buildConfigPanelDOM() {
  // Build the massive config panel DOM with all required inputs
  const inputIds = [
    'cfg-frame-interval', 'cfg-max-frames', 'cfg-pixel-limit',
    'cfg-pi3x-frame-target', 'cfg-conf-threshold', 'cfg-edge-rtol',
    'cfg-denoise-dbscan-eps', 'cfg-denoise-dbscan-eps-ratio',
    'cfg-denoise-dbscan-min-samples', 'cfg-denoise-dbscan-max-points',
    'cfg-denoise-sor-neighbors', 'cfg-denoise-sor-std',
    'cfg-denoise-radius-neighbors', 'cfg-denoise-radius-ratio',
    'cfg-diffcd-batch', 'cfg-diffcd-nbatches', 'cfg-diffcd-res',
    'cfg-meshwrap-poisson-depth', 'cfg-meshwrap-poisson-scale',
    'cfg-meshwrap-density-trim-q', 'cfg-meshwrap-face-ratio',
    'cfg-meshwrap-iterations', 'cfg-meshwrap-crop-scale',
    'cfg-meshwrap-sample-points', 'cfg-meshwrap-normal-radius',
    'cfg-meshwrap-smooth-iterations', 'cfg-meshwrap-quality-threshold',
    'cfg-meshwrap-alpha-ratio', 'cfg-meshwrap-offset-ratio',
    'cfg-classical-poisson-depth', 'cfg-classical-density-trim-q',
    'cfg-classical-smooth-iters', 'cfg-classical-target-faces',
    'cfg-mesh-repair-max-diameter-ratio', 'cfg-mesh-repair-y-band-ratio',
    'cfg-mesh-repair-smooth-iters',
  ];

  const numberInputs = inputIds.map(id => `<input type="number" id="${id}" value="0">`).join('\n');

  document.body.innerHTML = `
    <div id="config-panel">
      <div id="config-title">Configuration</div>
      <div class="config-section" data-stages="1 2">Section 1-2</div>
      <div class="config-section" data-stages="3">Section 3</div>
      <div class="config-section" data-stages="4 5">Section 4-5</div>

      <select id="video-select"></select>
      <div id="video-info"></div>
      <select id="object-select">
        <option value="__new__">Create New Object</option>
      </select>
      <input type="text" id="cfg-object-name" value="">
      <div id="object-info"></div>
      <ul id="object-artifacts"></ul>
      <div id="object-artifacts-empty"></div>
      <div id="cfg-resume-stage-info"></div>
      <div id="cfg-pi3x-frame-target-value"></div>
      <div id="cfg-pi3x-frame-target-note"></div>
      <div id="cfg-pi3x-plan-note"></div>
      <div id="cfg-pi3x-frame-target-marker"></div>
      <button id="btn-refresh-objects">Refresh</button>
      <button id="btn-start">Start</button>
      <button id="btn-cancel" disabled>Cancel</button>

      ${numberInputs}

      <select id="cfg-sam2-model">
        <option value="large">large</option>
        <option value="base_plus">base_plus</option>
      </select>
      <input type="checkbox" id="cfg-ground-plane-enabled" checked>

      <select id="cfg-denoise-preset">
        <option value="balanced">Balanced</option>
        <option value="detail_preserving">Detail Preserving</option>
        <option value="isolate_subject">Isolate Subject</option>
        <option value="sparse_noise">Sparse Noise</option>
        <option value="aggressive_cleanup">Aggressive</option>
        <option value="custom">Custom</option>
      </select>

      <select id="cfg-denoise-algorithm">
        <option value="dbscan_sor">DBSCAN + SOR</option>
        <option value="dbscan_only">DBSCAN</option>
        <option value="sor_only">SOR</option>
        <option value="radius_only">Radius</option>
        <option value="dbscan_radius">DBSCAN + Radius</option>
      </select>

      <div id="cfg-denoise-dbscan"></div>
      <div id="cfg-denoise-sor"></div>
      <div id="cfg-denoise-radius"></div>
      <div id="cfg-denoise-summary"></div>

      <select id="cfg-mesh-method">
        <option value="poisson">Classical Mesh (Poisson)</option>
        <option value="diffcd">Learning Mesh (DiffCD)</option>
      </select>
      <div id="cfg-mesh-method-summary"></div>
      <div id="cfg-poisson-summary"></div>
      <div id="cfg-diffcd-controls"></div>
      <div id="cfg-classical-controls"></div>
      <div id="cfg-meshwrap-summary"></div>
      <button id="btn-meshwrap-reset">Reset</button>
      <div id="cfg-meshwrap-poisson-group"></div>
      <div id="cfg-meshwrap-alpha-group"></div>
      <select id="cfg-meshwrap-method">
        <option value="poisson_iterative" selected>Poisson Iterative</option>
      </select>
      <div id="cfg-classical-summary"></div>
      <button id="btn-classical-reset">Reset</button>

      <select id="cfg-classical-preset">
        <option value="lightweight">Lightweight</option>
        <option value="trust_point_cloud" selected>Trust Point Cloud</option>
        <option value="custom">Custom</option>
      </select>
      <input type="checkbox" id="cfg-classical-preprocess" checked>
      <input type="checkbox" id="cfg-classical-auto-smooth">
      <input type="checkbox" id="cfg-classical-downsample" checked>
      <input type="checkbox" id="cfg-mesh-repair-enabled" checked>

      <select id="cfg-milo-preset">
        <option value="default" selected>Default</option>
        <option value="high">High</option>
        <option value="fast">Fast</option>
        <option value="custom">Custom</option>
      </select>
      <input type="number" id="cfg-milo-iterations" value="18000">
      <select id="cfg-milo-scene-type">
        <option value="indoor" selected>Indoor</option>
        <option value="outdoor">Outdoor</option>
      </select>
      <select id="cfg-milo-mesh-config">
        <option value="default" selected>Default</option>
        <option value="highres">High Res</option>
      </select>
      <select id="cfg-milo-extraction-method">
        <option value="sdf" selected>SDF</option>
        <option value="integration">Integration</option>
        <option value="regular_tsdf">Regular TSDF</option>
      </select>
      <input type="checkbox" id="cfg-milo-use-masks" checked>
      <input type="checkbox" id="cfg-colmap-filter-enabled" checked>
      <input type="number" id="cfg-colmap-filter-min-inside-views" value="2">
      <input type="number" id="cfg-colmap-filter-min-inside-ratio" value="0.6">
      <input type="number" id="cfg-colmap-filter-mask-dilate" value="1">
      <input type="number" id="cfg-colmap-filter-min-points" value="200">

      <select id="cfg-texture-size">
        <option value="0">Auto</option>
        <option value="512">512</option>
        <option value="1024">1024</option>
        <option value="2048">2048</option>
      </select>
      <select id="cfg-texture-view-assign-mode">
        <option value="legacy">Legacy Blend</option>
        <option value="region_gc">Region Optimized</option>
      </select>
      <input type="checkbox" id="cfg-texture-quality-boost">
    </div>
  `;
}

export function buildPreviewDOM() {
  document.body.innerHTML = `
    <div id="gallery-grid"></div>
  `;
}

export function buildRouterDOM() {
  document.body.innerHTML = `
    <div id="overview-grid"></div>
    <div id="overview-empty"></div>
  `;
}

export function buildOverviewDOM() {
  document.body.innerHTML = `
    <div id="overview-grid"></div>
    <div id="overview-empty" class="hidden"></div>
    <button id="overview-new-btn">New Pipeline</button>
  `;
}

/**
 * Comprehensive DOM for app.js integration tests.
 * Merges all module DOM requirements into a single tree.
 */
export function buildAppDOM() {
  // ── Pipeline pills + connectors ────────────────────────────
  let pills = '';
  for (let i = 1; i <= 8; i++) {
    pills += `
      <div class="stage-pill" data-stage="${i}" tabindex="0">
        <span class="stage-progress">0%</span>
        <span class="stage-time"></span>
      </div>`;
  }
  let connectors = '';
  for (let i = 0; i < 7; i++) {
    connectors += '<div class="stage-connector-main"></div>';
  }

  // ── Stage panels ───────────────────────────────────────────
  let panels = '';
  for (let i = 1; i <= 8; i++) {
    panels += `
      <div id="stage-panel-${i}" class="stage-panel">
        <div class="stage-panel-empty"></div>
        <div id="stage-${i}-empty"></div>
      </div>`;
  }

  // ── Task confirm bars (stages 1–7) ─────────────────────────
  let confirmBars = '';
  for (let s = 1; s <= 7; s++) {
    confirmBars += `
      <div class="task-confirm-bar" data-stage="${s}">
        <span class="task-confirm-message"></span>
        <button class="task-confirm-btn">Continue</button>
      </div>`;
  }

  // ── Config panel numeric inputs ────────────────────────────
  const numberIds = [
    'cfg-frame-interval', 'cfg-max-frames', 'cfg-pixel-limit',
    'cfg-pi3x-frame-target', 'cfg-conf-threshold', 'cfg-edge-rtol',
    'cfg-denoise-dbscan-eps', 'cfg-denoise-dbscan-eps-ratio',
    'cfg-denoise-dbscan-min-samples', 'cfg-denoise-dbscan-max-points',
    'cfg-denoise-sor-neighbors', 'cfg-denoise-sor-std',
    'cfg-denoise-radius-neighbors', 'cfg-denoise-radius-ratio',
    'cfg-diffcd-batch', 'cfg-diffcd-nbatches', 'cfg-diffcd-res',
    'cfg-meshwrap-poisson-depth', 'cfg-meshwrap-poisson-scale',
    'cfg-meshwrap-density-trim-q', 'cfg-meshwrap-face-ratio',
    'cfg-meshwrap-iterations', 'cfg-meshwrap-crop-scale',
    'cfg-meshwrap-sample-points', 'cfg-meshwrap-normal-radius',
    'cfg-meshwrap-smooth-iterations', 'cfg-meshwrap-quality-threshold',
    'cfg-meshwrap-alpha-ratio', 'cfg-meshwrap-offset-ratio',
    'cfg-classical-poisson-depth', 'cfg-classical-density-trim-q',
    'cfg-classical-smooth-iters', 'cfg-classical-target-faces',
    'cfg-mesh-repair-max-diameter-ratio', 'cfg-mesh-repair-y-band-ratio',
    'cfg-mesh-repair-smooth-iters',
  ];
  const numInputs = numberIds.map(id => `<input type="number" id="${id}" value="0">`).join('\n');

  document.body.innerHTML = `
    <!-- App-level badges -->
    <span id="status-badge" class="badge badge-idle">Idle</span>
    <span id="vram-badge">VRAM: --</span>
    <span id="overall-progress-badge">Overall: 0%</span>
    <div id="mesh-phase-status" class="hidden"><span id="mesh-phase-status-text"></span></div>
    <h1 id="header-title" data-i18n="header.title">clip2mesh</h1>
    <span id="breadcrumb-text"></span>

    <!-- Overview -->
    <div id="overview-grid"></div>
    <div id="overview-empty"></div>
    <button id="overview-new-btn">New Pipeline</button>

    <!-- Pipeline pills + connectors -->
    ${pills}
    ${connectors}
    <div id="mesh-branch-slot">
      <div id="mesh-pill-poisson" class="mesh-method-pill">Poisson</div>
      <div id="mesh-pill-diffcd" class="mesh-method-pill">DiffCD</div>
    </div>
    <div class="mesh-stage-pill mesh-method-pill active" data-stage="5"></div>
    <div class="mesh-stage-pill" data-stage="5-alt"></div>
    <div id="mesh-connector-poisson-left"></div>
    <div id="mesh-connector-poisson-right"></div>
    <div id="mesh-connector-diffcd-left"></div>
    <div id="mesh-connector-diffcd-right"></div>

    <!-- Stage panels -->
    ${panels}

    <!-- Config panel -->
    <div id="config-panel">
      <div id="config-title">Configuration</div>
      <div class="config-section" data-stages="1 2">Section 1-2</div>
      <div class="config-section" data-stages="3">Section 3</div>
      <div class="config-section" data-stages="4 5">Section 4-5</div>
      <div class="config-section" data-stages="6 7 8">Section 6-8</div>

      <select id="video-select"></select>
      <div id="video-info"></div>
      <select id="object-select">
        <option value="__new__">Create New Object</option>
      </select>
      <input type="text" id="cfg-object-name" value="">
      <div id="object-info"></div>
      <ul id="object-artifacts"></ul>
      <div id="object-artifacts-empty"></div>
      <div id="cfg-resume-stage-info"></div>
      <div id="cfg-pi3x-frame-target-value"></div>
      <div id="cfg-pi3x-frame-target-note"></div>
      <div id="cfg-pi3x-plan-note"></div>
      <div id="cfg-pi3x-frame-target-marker"></div>
      <button id="btn-refresh-objects">Refresh</button>
      <button id="btn-start">Start</button>
      <button id="btn-cancel" disabled>Cancel</button>

      ${numInputs}

      <select id="cfg-sam2-model">
        <option value="large">large</option>
        <option value="base_plus">base_plus</option>
      </select>
      <input type="checkbox" id="cfg-ground-plane-enabled">

      <select id="cfg-denoise-preset">
        <option value="balanced" selected>Balanced</option>
        <option value="custom">Custom</option>
      </select>
      <select id="cfg-denoise-algorithm">
        <option value="dbscan_sor" selected>DBSCAN + SOR</option>
      </select>
      <div id="cfg-denoise-dbscan"></div>
      <div id="cfg-denoise-sor"></div>
      <div id="cfg-denoise-radius"></div>
      <div id="cfg-denoise-summary"></div>

      <select id="cfg-mesh-method">
        <option value="poisson" selected>Classical Mesh (Poisson)</option>
        <option value="diffcd">Learning Mesh (DiffCD)</option>
      </select>
      <div id="cfg-mesh-method-summary"></div>
      <div id="cfg-poisson-summary"></div>
      <div id="cfg-diffcd-controls"></div>
      <div id="cfg-classical-controls"></div>
      <div id="cfg-meshwrap-summary"></div>
      <button id="btn-meshwrap-reset">Reset</button>
      <div id="cfg-meshwrap-poisson-group"></div>
      <div id="cfg-meshwrap-alpha-group"></div>
      <select id="cfg-meshwrap-method">
        <option value="poisson" selected>Poisson</option>
      </select>
      <div id="cfg-classical-summary"></div>
      <button id="btn-classical-reset">Reset</button>

      <select id="cfg-classical-preset">
        <option value="trust_point_cloud" selected>Trust Point Cloud</option>
        <option value="custom">Custom</option>
      </select>
      <input type="checkbox" id="cfg-classical-preprocess" checked>
      <input type="checkbox" id="cfg-classical-auto-smooth">
      <input type="checkbox" id="cfg-classical-downsample" checked>
      <input type="checkbox" id="cfg-mesh-repair-enabled" checked>

      <select id="cfg-texture-size">
        <option value="0" selected>Auto</option>
      </select>
      <select id="cfg-texture-view-assign-mode">
        <option value="legacy" selected>Legacy Blend</option>
      </select>
      <input type="checkbox" id="cfg-texture-quality-boost">
    </div>

    <!-- SAM2 canvas -->
    <div id="sam2-placeholder">Click to start</div>
    <canvas id="sam2-canvas" style="display:none"></canvas>
    <button id="sam2-undo" disabled>Undo</button>
    <button id="sam2-clear" disabled>Clear</button>
    <button id="sam2-confirm" disabled>Confirm & Propagate</button>
    <span id="sam2-click-info">Positive: 0  Negative: 0</span>

    <!-- SAM2 verification -->
    <div id="sam2-verification" class="hidden">
      <div id="sam2-verification-strip"></div>
      <button id="sam2-approve">Approve & Continue</button>
      <button id="sam2-redo">Redo</button>
    </div>

    <!-- Checkpoint panel -->
    <div id="checkpoint-panel">
      <div id="checkpoint-stage-label"></div>
      <ul id="checkpoint-list"></ul>
    </div>

    <!-- Log viewer -->
    <div id="log-container" style="height:300px;overflow:auto;">
      <div id="log-content"></div>
    </div>
    <button id="log-clear">Clear</button>

    <!-- Settings -->
    <button id="settings-btn">Settings</button>

    <!-- Preview -->
    <div id="gallery-grid"></div>
    <div id="frame-count-header"></div>
    <div id="frame-count-text"></div>
    <div id="pi3x-toolbar" class="hidden"></div>
    <span id="pi3x-point-count"></span>
    <span id="pi3x-camera-count"></span>
    <button id="pi3x-cameras-toggle"></button>

    <!-- Task confirm bars -->
    ${confirmBars}

    <!-- Mesh repair toolbar -->
    <div id="mesh-repair-toolbar" class="hidden">
      <span id="mesh-repair-status"></span>
      <input type="range" id="mesh-repair-threshold" min="-1" max="0" step="0.01" value="-0.25">
      <span id="mesh-repair-threshold-value">-0.25</span>
      <button id="mesh-repair-clear">Clear</button>
      <button id="mesh-repair-apply">Apply</button>
    </div>

    <!-- Mesh post toolbar -->
    <div id="mesh-post-toolbar" class="hidden">
      <select id="mesh-post-method"><option value="taubin" selected>Taubin</option></select>
      <input type="number" id="mesh-post-iterations" value="10">
      <input type="number" id="mesh-post-lambda" value="0.5">
      <input type="checkbox" id="mesh-post-downsample-enabled">
      <input type="number" id="mesh-post-target-faces" value="50000">
      <input type="number" id="mesh-post-trigger-faces" value="100000">
      <button id="mesh-post-apply">Apply</button>
      <button id="mesh-post-reset">Reset</button>
      <span id="mesh-post-status"></span>
    </div>
  `;
}
