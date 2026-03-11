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
    <div id="sam2-verification" style="display:none">
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
    </div>
  `;
}

export function buildPreviewDOM() {
  document.body.innerHTML = `
    <div id="gallery-grid"></div>
  `;
}
