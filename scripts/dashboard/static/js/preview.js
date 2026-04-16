/**
 * 3D preview manager — per-stage scene support for PLY/OBJ.
 *
 * Uses a single shared renderer that moves between stage containers.
 * Stage 2 has a dedicated scene for camera overlay support.
 *
 * 6-stage pipeline:
 *   1. Extract Frames
 *   2. COLMAP SfM (camera_poses.json + colmap_sparse_points.ply)
 *   3. SAM2 Segmentation (masks only)
 *   4. gs2mesh Reconstruction (object_mesh.ply)
 *   5. Texture Bake (textured_mesh.obj)
 *   6. Post-texture Cleanup (textured_mesh_cleaned.obj)
 */

import { SCENE_THEMES } from './preview/constants.js';
import * as PoseUtils from './preview/pose-utils.js';
import * as SceneHelpers from './preview/scene-helpers.js';

let THREE;
let OrbitControls;
let PLYLoader;
let OBJLoader;
let MTLLoader;

export class PreviewPanel {
  constructor() {
    this._galleryGrid = document.getElementById('gallery-grid');

    // Per-stage scene state: { scene, camera, controls, container, currentObject, initialized }
    this._stages = {};
    this._renderer = null;
    this._threeLoaded = false;
    this._activeStage = null;
    this._animating = false;
    this._currentTheme = 'dark';
    this._sceneFlipX = null;
    this._previewAssetRevision = 0;
    this._groundPlane = {
      meshByStage: {},   // stageNum -> THREE.Mesh
      data: null,        // parsed ground_plane.json
      visible: true,
    };
    // Object name of the currently-hydrated run. Used to resolve the stage-6
    // "final deliverable" subfolder at ``{object_name}/textured_mesh_cleaned.obj``.
    this._activeObjectName = '';
  }

  setActiveObjectName(name) {
    this._activeObjectName = String(name || '').trim();
  }

  _syncViewportSize(stageNum, retries = 0) {
    const stage = this._stages[stageNum];
    if (!stage || !this._renderer) return false;

    const w = stage.container.clientWidth;
    const h = stage.container.clientHeight;
    if (w > 0 && h > 0) {
      this._renderer.setSize(w, h);
      stage.camera.aspect = w / h;
      stage.camera.updateProjectionMatrix();
      return true;
    }

    if (retries > 0) {
      requestAnimationFrame(() => {
        if (this._activeStage !== stageNum) return;
        this._syncViewportSize(stageNum, retries - 1);
      });
    }
    return false;
  }

  reset() {
    this._sceneFlipX = null;
    this.clearFromStage(1);
  }

  clearFromStage(startStage = 1) {
    const start = Math.max(1, Math.min(6, Number(startStage) || 1));

    if (start <= 1) {
      const empty = document.querySelector('#stage-panel-1 .stage-panel-empty');
      if (empty) empty.classList.remove('hidden');
      const headerEl = document.getElementById('frame-count-header');
      if (headerEl) headerEl.classList.add('hidden');
      if (this._galleryGrid) this._galleryGrid.innerHTML = '';
    }

    if (start <= 2) {
      this._clearStageScene(2);
      const empty = document.getElementById('stage-2-empty');
      if (empty) empty.classList.remove('hidden');
      const toolbar = document.getElementById('colmap-toolbar');
      if (toolbar) toolbar.classList.add('hidden');
      const pointCount = document.getElementById('colmap-point-count');
      if (pointCount) pointCount.textContent = '';
      const cameraCount = document.getElementById('colmap-camera-count');
      if (cameraCount) cameraCount.textContent = '';
    }

    if (start <= 4) {
      this._clearStageScene(4);
      const empty = document.getElementById('stage-4-empty');
      if (empty) empty.classList.remove('hidden');
    }

    if (start <= 5) {
      this._clearStageScene(5);
      const empty = document.getElementById('stage-5-empty');
      if (empty) empty.classList.remove('hidden');
    }

    if (start <= 6) {
      this._clearStageScene(6);
      const empty = document.getElementById('stage-6-empty');
      if (empty) empty.classList.remove('hidden');
    }
  }

  /**
   * Load three.js and addons (once).
   */
  async _loadThree() {
    if (this._threeLoaded) return;
    try {
      THREE = await import('three');
      const addons = await Promise.all([
        import('three/addons/controls/OrbitControls.js'),
        import('three/addons/loaders/PLYLoader.js'),
        import('three/addons/loaders/OBJLoader.js'),
        import('three/addons/loaders/MTLLoader.js'),
      ]);
      OrbitControls = addons[0].OrbitControls;
      PLYLoader = addons[1].PLYLoader;
      OBJLoader = addons[2].OBJLoader;
      MTLLoader = addons[3].MTLLoader;
      // Initialize THREE in submodules
      PoseUtils.initThree(THREE);
      SceneHelpers.initThree(THREE);
      this._threeLoaded = true;
    } catch (e) {
      console.error('Failed to load three.js:', e);
    }
  }

  /**
   * Get THREE module reference (for camera overlay).
   */
  get THREE() { return THREE; }

  /**
   * Lazy-init a 3D scene for a given stage.
   */
  async initSceneForStage(stageNum) {
    if (this._stages[stageNum]?.initialized) return;

    await this._loadThree();
    if (!this._threeLoaded) return;

    const containerId = `three-container-${stageNum}`;
    const container = document.getElementById(containerId);
    if (!container) return;

    // Create shared renderer once
    if (!this._renderer) {
      this._renderer = new THREE.WebGLRenderer({ antialias: true });
      this._renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      this._renderer.shadowMap.enabled = true;
      this._renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    }

    const w = container.clientWidth || 640;
    const h = container.clientHeight || 480;

    const palette = SCENE_THEMES[this._currentTheme];

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(palette.background);

    const camera = new THREE.PerspectiveCamera(60, w / h, 0.01, 100);
    camera.position.set(0, 0, 2);

    const controls = new OrbitControls(camera, this._renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.1;

    // Lights
    const ambient = new THREE.AmbientLight(0xffffff, palette.ambientInt);
    scene.add(ambient);
    const dirLight = new THREE.DirectionalLight(0xffffff, palette.dirInt);
    dirLight.position.set(2, 3, 4);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.set(1024, 1024);
    dirLight.shadow.camera.near = 0.1;
    dirLight.shadow.camera.far = 40;
    dirLight.shadow.camera.left = -6;
    dirLight.shadow.camera.right = 6;
    dirLight.shadow.camera.top = 6;
    dirLight.shadow.camera.bottom = -6;
    dirLight.shadow.bias = -0.0006;
    dirLight.shadow.normalBias = 0.02;
    scene.add(dirLight);

    // Grid
    const grid = new THREE.GridHelper(4, 20, palette.gridPrimary, palette.gridSecondary);
    scene.add(grid);

    // Stage 2 starts with OpenCV->OpenGL flip. Stages 4-5 use inferred flip when available.
    const sceneRoot = new THREE.Group();
    sceneRoot.rotation.x = this._defaultSceneFlipX(stageNum) ? Math.PI : 0;
    scene.add(sceneRoot);

    // Hidden by default. Enabled only when mesh-shadow profile is active.
    const shadowFloor = new THREE.Mesh(
      new THREE.PlaneGeometry(1, 1),
      new THREE.ShadowMaterial({ color: palette.shadowColor, opacity: palette.shadowOpacity }),
    );
    shadowFloor.material.side = THREE.DoubleSide;
    shadowFloor.rotation.x = -Math.PI / 2;
    shadowFloor.receiveShadow = true;
    shadowFloor.visible = false;
    sceneRoot.add(shadowFloor);

    // Show container
    container.classList.add('visible');

    this._stages[stageNum] = {
      scene, sceneRoot, camera, controls, container,
      currentObject: null,
      overlayObject: null,
      _loadGeneration: 0,
      grid,
      ambientLight: ambient,
      keyLight: dirLight,
      shadowFloor,
      initialized: true,
    };

    // Resize observer
    const ro = new ResizeObserver(() => {
      const nw = container.clientWidth;
      const nh = container.clientHeight;
      if (nw > 0 && nh > 0) {
        if (this._activeStage === stageNum && this._renderer) {
          this._renderer.setSize(nw, nh);
        }
        camera.aspect = nw / nh;
        camera.updateProjectionMatrix();
      }
    });
    ro.observe(container);

    // Start animation loop if not running
    if (!this._animating) {
      this._animating = true;
      this._animate();
    }
  }

  /**
   * Activate the renderer for a specific stage (moves canvas to container).
   */
  activateStage(stageNum) {
    const stage = this._stages[stageNum];
    if (!stage || !this._renderer) return;

    this._activeStage = stageNum;
    stage.container.classList.add('visible');

    // Move renderer canvas to this container
    const canvas = this._renderer.domElement;
    const needsRebind = canvas.parentElement !== stage.container;

    if (needsRebind) {
      stage.container.appendChild(canvas);

      // Re-bind controls to the new parent
      stage.controls.dispose();
      stage.controls = new OrbitControls(stage.camera, canvas);
      stage.controls.enableDamping = true;
      stage.controls.dampingFactor = 0.1;
    }

    // Resize (retry on next frames to avoid zero-size reads right after panel switch).
    this._syncViewportSize(stageNum, 3);
  }

  /**
   * Animation loop — renders the active stage.
   */
  _animate() {
    requestAnimationFrame(() => this._animate());
    if (this._activeStage == null) return;
    const stage = this._stages[this._activeStage];
    if (!stage || !this._renderer) return;
    stage.controls.update();
    this._renderer.render(stage.scene, stage.camera);
  }

  _clearStageScene(stageNum) {
    const stage = this._stages[stageNum];
    if (!stage) return;
    if (stageNum >= 4) {
      this._removeGroundPlane(stageNum);
    }
    if (stage.currentObject) {
      this._disposeObject(stage.currentObject);
      stage.sceneRoot.remove(stage.currentObject);
      stage.currentObject = null;
    }
    stage.centerOffset = null;
    this._setMeshShadowProfile(stage, null, false);
    stage.container?.classList.remove('visible');
  }

  _defaultSceneFlipX(stageNum) {
    if (this._sceneFlipX != null) return this._sceneFlipX;
    return stageNum === 2;
  }

  /**
   * Apply a colour theme (light/dark) to all initialized 3D scenes.
   * Updates background, grid, lights, shadow floor, and object materials
   * without resetting camera or reloading geometry.
   */
  applyTheme(themeName) {
    const name = (themeName === 'light') ? 'light' : 'dark';
    this._currentTheme = name;
    const palette = SCENE_THEMES[name];

    for (const stage of Object.values(this._stages)) {
      if (!stage?.initialized) continue;

      // Scene background
      stage.scene.background.setHex(palette.background);

      // Grid: recreate (vertex-colored, can't repaint in place)
      if (stage.grid) {
        stage.scene.remove(stage.grid);
        stage.grid.geometry.dispose();
        stage.grid.material.dispose();
      }
      const newGrid = new THREE.GridHelper(4, 20, palette.gridPrimary, palette.gridSecondary);
      stage.scene.add(newGrid);
      stage.grid = newGrid;

      // Lights — respect active shadow profile
      const shadowActive = stage.shadowFloor?.visible === true;
      if (stage.ambientLight) {
        stage.ambientLight.intensity = shadowActive
          ? palette.shadowAmbientInt : palette.ambientInt;
      }
      if (stage.keyLight) {
        stage.keyLight.intensity = shadowActive
          ? palette.shadowDirInt : palette.dirInt;
      }

      // Shadow floor
      if (stage.shadowFloor?.material) {
        stage.shadowFloor.material.color.setHex(palette.shadowColor);
        stage.shadowFloor.material.opacity = palette.shadowOpacity;
      }

      // Update materials on current object (mesh / edges / points)
      if (stage.currentObject) {
        stage.currentObject.traverse((node) => {
          if (!node.material) return;
          const mat = node.material;
          if (mat.isMeshStandardMaterial && !mat.vertexColors) {
            mat.color.setHex(palette.meshColor);
          } else if (mat.isLineBasicMaterial && node.isLineSegments) {
            mat.color.setHex(palette.edgeColor);
            mat.opacity = palette.edgeOpacity;
          } else if (mat.isPointsMaterial && !mat.vertexColors) {
            mat.color.setHex(palette.pointColor);
          }
        });
      }
    }
  }

  /**
   * Load COLMAP results: sparse point cloud + camera poses.
   * @param {CameraOverlay} cameraOverlay - camera overlay instance
   * @param {string} [plyFile='colmap_sparse_points.ply'] - PLY filename to load
   */
  async loadColmapResults(cameraOverlay, plyFile = 'colmap_sparse_points.ply') {
    await this.initSceneForStage(2);
    this.activateStage(2);

    const empty = document.getElementById('stage-2-empty');
    const toolbar = document.getElementById('colmap-toolbar');
    if (toolbar) toolbar.classList.add('hidden');

    const stage = this._stages[2];
    if (!stage) return;
    stage.centerOffset = null;
    cameraOverlay.remove();
    const cacheToken = this._nextPreviewRevision();

    // Load point cloud
    const loaded = await this._loadPLYIntoStage(2, plyFile, { cacheToken, densityCenter: true, fitToStage: true });
    if (!loaded) {
      console.warn(`COLMAP sparse points not ready: ${plyFile}`);
      if (empty) empty.classList.remove('hidden');
      const pointCountEl = document.getElementById('colmap-point-count');
      if (pointCountEl) pointCountEl.textContent = '';
    } else {
      if (empty) empty.classList.add('hidden');
      if (toolbar) toolbar.classList.remove('hidden');
      if (stage.currentObject?.geometry) {
        const count = stage.currentObject.geometry.attributes.position?.count || 0;
        const el = document.getElementById('colmap-point-count');
        if (el) el.textContent = `${count.toLocaleString()} points`;
      }
    }

    // Load camera poses
    let cameraLoaded = false;
    try {
      const res = await fetch(this._buildPreviewFileUrl('camera_poses.json', cacheToken));
      if (res.ok) {
        const data = await res.json();

        const rawPoseArray = this._poseJsonToArray(data);

        const normalized = this._normalizePoseConvention(rawPoseArray);
        const poseArray = normalized.poseArray;
        const sceneFlipX = this._shouldApplySceneFlipX(poseArray, data.alignment);
        this._sceneFlipX = sceneFlipX;
        this._applySceneFlipToLoadedStages(sceneFlipX);

        const alignmentSign = this._forwardSignFromAlignment(data.alignment);
        let forwardSign = normalized.forwardSign;
        let forwardSignSource = `orbit-${normalized.convention}`;
        if (normalized.used < 3 || normalized.confidence < 0.05) {
          // If orbit-based estimate is weak, fallback to metadata.
          if (alignmentSign != null) {
            forwardSign = alignmentSign;
            forwardSignSource = data.alignment?.inferred_forward_axis || 'alignment-fallback';
          }
        }

        if (forwardSign == null) {
          const targetCenter = stage.centerOffset || new THREE.Vector3(0, 0, 0);
          forwardSign = this._inferForwardSignFromTarget(poseArray, targetCenter);
          forwardSignSource = 'target-center-fallback';
        }

        console.info(
          'Camera overlay pose convention:',
          normalized.convention,
          'confidence:',
          normalized.confidence.toFixed(3),
          'frames:',
          normalized.used,
        );
        console.info('COLMAP scene flip X(pi):', sceneFlipX);
        console.info('Camera overlay forwardSign:', forwardSign, forwardSignSource);

        cameraOverlay.create(THREE, stage.sceneRoot, poseArray, { forwardSign });
        cameraLoaded = true;
        stage.container?.classList.add('visible');

        // Apply same centering offset as point cloud
        if (stage.centerOffset) {
          cameraOverlay.applyOffset(stage.centerOffset);
        }

        // Display camera count
        const countEl = document.getElementById('colmap-camera-count');
        if (countEl) countEl.textContent = `${poseArray.length} cameras`;

        // Bind toggle
        const toggle = document.getElementById('colmap-cameras-toggle');
        if (toggle) {
          toggle.onchange = () => cameraOverlay.setVisible(toggle.checked);
        }
        if (empty) empty.classList.add('hidden');
        if (toolbar) toolbar.classList.remove('hidden');
      } else {
        cameraOverlay.remove();
      }
    } catch (e) {
      cameraOverlay.remove();
      console.error('Failed to load camera poses:', e);
    }

    if (!cameraLoaded) {
      const countEl = document.getElementById('colmap-camera-count');
      if (countEl) countEl.textContent = '';
    }
    if (!loaded && !cameraLoaded) {
      if (empty) empty.classList.remove('hidden');
      if (toolbar) toolbar.classList.add('hidden');
      stage.container?.classList.remove('visible');
    }
  }

  // ── Preview revision & URL helpers ────────────────────────────────

  _nextPreviewRevision() {
    this._previewAssetRevision += 1;
    return this._previewAssetRevision;
  }

  _buildPreviewFileUrl(relativePath, cacheToken = null) {
    const base = `/api/preview/file/${relativePath}`;
    if (cacheToken == null || cacheToken === '') return base;
    return `${base}?rev=${encodeURIComponent(String(cacheToken))}`;
  }

  // ── Stage result loading ──────────────────────────────────────────

  /**
   * Auto-load the appropriate result file for a stage.
   */
  async loadStageResult(stageNum, opts = {}) {
    // Stage 6 writes its cleaned textured mesh into the ``{object_name}/``
    // subfolder so the deliverable is a self-contained package. Resolve that
    // prefix at call time because the active object varies per run.
    const objectPrefix = this._activeObjectName ? `${this._activeObjectName}/` : '';
    const fileMap = {
      4: 'object_mesh.ply',
      5: 'textured_mesh.obj',
      6: `${objectPrefix}textured_mesh_cleaned.obj`,
    };

    const overrideFile = String(opts.file || '').trim();
    const defaultFile = fileMap[stageNum];
    const file = overrideFile || defaultFile;
    if (!file) return false;
    const fileName = file.split('/').pop()?.toLowerCase() || '';
    const isMesh = fileName.endsWith('.ply') && stageNum >= 4;

    const cacheToken = opts.cacheToken ?? this._nextPreviewRevision();
    await this.initSceneForStage(stageNum);
    await this._ensureSceneFlipForStage(stageNum, cacheToken);
    this.activateStage(stageNum);
    // Re-sync viewport after container is guaranteed visible
    requestAnimationFrame(() => this._syncViewportSize(stageNum, 2));

    // Toggle empty placeholder based on load result.
    const empty = document.getElementById(`stage-${stageNum}-empty`);

    const ext = file.split('.').pop().toLowerCase();
    const renderMode = String(opts.renderMode || (isMesh ? 'mesh' : '')).toLowerCase();
    const stripVertexColors = opts.stripVertexColors ?? false;
    const enableShadows = opts.enableShadows ?? isMesh;
    const loaded = await this._loadPreviewAssetForStage(stageNum, file, {
      ext,
      cacheToken,
      renderMode,
      stripVertexColors: stripVertexColors === true,
      enableShadows: enableShadows === true,
    });
    if (loaded && stageNum >= 4) {
      this.showGroundPlane(stageNum).catch(() => {});
    }
    if (empty) empty.classList.toggle('hidden', loaded);
    return loaded;
  }

  // ── Asset loaders ─────────────────────────────────────────────────

  async _loadPreviewAssetForStage(stageNum, file, opts = {}) {
    const ext = String(opts.ext || file.split('.').pop() || '').toLowerCase();
    if (ext === 'ply') {
      return this._loadPLYIntoStage(stageNum, file, {
        cacheToken: opts.cacheToken,
        renderMode: opts.renderMode,
        stripVertexColors: opts.stripVertexColors === true,
        enableShadows: opts.enableShadows === true,
      });
    }
    if (ext === 'obj') {
      return this._loadOBJIntoStage(stageNum, file, { cacheToken: opts.cacheToken });
    }
    return false;
  }

  /**
   * Estimate the densest region of a point cloud via a 32^3 voxel grid.
   * Returns the centroid of points falling in the densest voxel's 3x3x3
   * neighborhood, or null if the input is degenerate (caller should fall
   * back to the bbox center).
   */
  _computeDensityCenter(geometry, bbox) {
    const positions = geometry.attributes?.position;
    if (!positions) return null;
    const count = positions.count;
    if (count < 32) return null;

    const minX = bbox.min.x, minY = bbox.min.y, minZ = bbox.min.z;
    const extX = bbox.max.x - minX;
    const extY = bbox.max.y - minY;
    const extZ = bbox.max.z - minZ;
    if (!(extX > 0) || !(extY > 0) || !(extZ > 0)) return null;

    const GRID = 32;
    const counts = new Int32Array(GRID * GRID * GRID);
    const arr = positions.array;
    const stride = positions.itemSize || 3;

    const invX = GRID / extX;
    const invY = GRID / extY;
    const invZ = GRID / extZ;
    const maxIdx = GRID - 1;

    for (let i = 0; i < count; i++) {
      const b = i * stride;
      let ix = Math.floor((arr[b]     - minX) * invX);
      let iy = Math.floor((arr[b + 1] - minY) * invY);
      let iz = Math.floor((arr[b + 2] - minZ) * invZ);
      if (ix < 0) ix = 0; else if (ix > maxIdx) ix = maxIdx;
      if (iy < 0) iy = 0; else if (iy > maxIdx) iy = maxIdx;
      if (iz < 0) iz = 0; else if (iz > maxIdx) iz = maxIdx;
      counts[(iz * GRID + iy) * GRID + ix]++;
    }

    let best = -1, bx = 0, by = 0, bz = 0;
    for (let z = 0; z < GRID; z++) {
      for (let y = 0; y < GRID; y++) {
        const row = (z * GRID + y) * GRID;
        for (let x = 0; x < GRID; x++) {
          const c = counts[row + x];
          if (c > best) { best = c; bx = x; by = y; bz = z; }
        }
      }
    }
    if (best <= 0) return null;

    const loX = Math.max(0, bx - 1), hiX = Math.min(maxIdx, bx + 1);
    const loY = Math.max(0, by - 1), hiY = Math.min(maxIdx, by + 1);
    const loZ = Math.max(0, bz - 1), hiZ = Math.min(maxIdx, bz + 1);

    let sumX = 0, sumY = 0, sumZ = 0, n = 0;
    for (let i = 0; i < count; i++) {
      const b = i * stride;
      const px = arr[b], py = arr[b + 1], pz = arr[b + 2];
      let ix = Math.floor((px - minX) * invX);
      let iy = Math.floor((py - minY) * invY);
      let iz = Math.floor((pz - minZ) * invZ);
      if (ix < 0) ix = 0; else if (ix > maxIdx) ix = maxIdx;
      if (iy < 0) iy = 0; else if (iy > maxIdx) iy = maxIdx;
      if (iz < 0) iz = 0; else if (iz > maxIdx) iz = maxIdx;
      if (ix < loX || ix > hiX || iy < loY || iy > hiY || iz < loZ || iz > hiZ) continue;
      sumX += px; sumY += py; sumZ += pz; n++;
    }
    if (n === 0) return null;

    const cx = sumX / n, cy = sumY / n, cz = sumZ / n;
    if (!Number.isFinite(cx) || !Number.isFinite(cy) || !Number.isFinite(cz)) return null;
    return new THREE.Vector3(cx, cy, cz);
  }

  /**
   * Load a PLY file into a specific stage's scene.
   */
  async _loadPLYIntoStage(stageNum, relativePath, opts = {}) {
    const stage = this._stages[stageNum];
    if (!stage) return false;
    const gen = ++stage._loadGeneration;

    // Remove previous object
    this._cleanupCurrentObject(stage);

    const url = this._buildPreviewFileUrl(relativePath, opts.cacheToken);
    const loader = new PLYLoader();

    try {
      const geometry = await new Promise((resolve, reject) => {
        loader.load(url, resolve, undefined, reject);
      });

      // Stale load — a newer call has superseded this one
      if (stage._loadGeneration !== gen) {
        geometry.dispose();
        return false;
      }

      geometry.computeBoundingBox();
      let center = null;
      if (opts.densityCenter === true) {
        center = this._computeDensityCenter(geometry, geometry.boundingBox);
      }
      if (!center) {
        center = new THREE.Vector3();
        geometry.boundingBox.getCenter(center);
      }
      geometry.translate(-center.x, -center.y, -center.z);
      stage.centerOffset = center;
      // Recompute bounds after centering so the camera fit uses the updated box.
      geometry.computeBoundingBox();

      let obj;
      const mode = String(opts.renderMode || '').toLowerCase();
      const forcePoints = mode === 'points';
      const forceMesh = mode === 'mesh';
      const renderAsMesh = forceMesh || (!forcePoints && (stageNum >= 4 && geometry.index && geometry.index.count > 0));
      const stripVertexColors = renderAsMesh && opts.stripVertexColors === true;
      const enableShadows = renderAsMesh && opts.enableShadows === true;
      if (stripVertexColors && geometry.hasAttribute('color')) {
        geometry.deleteAttribute('color');
      }
      this._setMeshShadowProfile(stage, geometry.boundingBox, enableShadows);
      if (renderAsMesh) {
        // Mesh
        // Rebuild normals to avoid flat-looking shading from broken/stale attributes.
        geometry.computeVertexNormals();
        geometry.normalizeNormals();

        const hasColor = geometry.hasAttribute('color');
        const meshMat = new THREE.MeshStandardMaterial({
          vertexColors: hasColor,
          color: hasColor ? undefined : SCENE_THEMES[this._currentTheme].meshColor,
          roughness: 0.9,
          metalness: 0.02,
          side: THREE.DoubleSide,
        });

        const mesh = new THREE.Mesh(geometry, meshMat);
        mesh.castShadow = enableShadows;
        mesh.receiveShadow = enableShadows;

        // Lightweight edge overlay to reveal silhouette when no vertex color or texture is present.
        let overlay = null;
        if (!hasColor) {
          try {
            const edges = new THREE.EdgesGeometry(geometry, 25);
            const lineMat = new THREE.LineBasicMaterial({
              color: SCENE_THEMES[this._currentTheme].edgeColor,
              transparent: true,
              opacity: SCENE_THEMES[this._currentTheme].edgeOpacity,
            });
            overlay = new THREE.LineSegments(edges, lineMat);
          } catch (edgeErr) {
            console.warn('Edge overlay skipped:', edgeErr);
          }
        }

        obj = new THREE.Group();
        obj.add(mesh);
        if (overlay) obj.add(overlay);
      } else {
        // Point cloud
        const materialParams = {
          size: 0.005,
          vertexColors: geometry.hasAttribute('color'),
          sizeAttenuation: true,
        };
        if (!geometry.hasAttribute('color')) {
          materialParams.color = SCENE_THEMES[this._currentTheme].pointColor;
        }
        const material = new THREE.PointsMaterial(materialParams);
        obj = new THREE.Points(geometry, material);
      }

      this._cleanupCurrentObject(stage);
      stage.currentObject = obj;
      stage.sceneRoot.add(obj);
      // clearFromStage() can hide initialized containers; show it again on successful load.
      stage.container?.classList.add('visible');
      if (opts.fitToStage === true) {
        this._fitCameraToStage(stage);
      } else {
        this._fitCamera(stage, geometry.boundingBox);
      }
      return true;
    } catch (e) {
      if (stage._loadGeneration !== gen) return false;
      console.error(`Failed to load PLY (stage ${stageNum}):`, e);
      stage.centerOffset = null;
      this._setMeshShadowProfile(stage, null, false);
      return false;
    }
  }

  /**
   * Load an OBJ file into a specific stage's scene.
   */
  async _loadOBJIntoStage(stageNum, relativePath, opts = {}) {
    const stage = this._stages[stageNum];
    if (!stage) return false;
    const gen = ++stage._loadGeneration;

    this._cleanupCurrentObject(stage);

    const cacheToken = opts.cacheToken;
    const url = this._buildPreviewFileUrl(relativePath, cacheToken);
    const mtlPath = relativePath.replace(/\.obj$/i, '.mtl');
    const mtlPathWithRevision = cacheToken == null || cacheToken === ''
      ? mtlPath
      : `${mtlPath}?rev=${encodeURIComponent(String(cacheToken))}`;
    let materials = null;

    // Resolve texture references relative to the MTL's directory so bare
    // ``map_Kd texture.png`` entries in a subfolder MTL (e.g. the stage-6
    // ``{object_name}/textured_mesh_cleaned.mtl``) pick up the texture sitting
    // next to the MTL, not a same-named file at the output root.
    const lastSlash = mtlPath.lastIndexOf('/');
    const mtlDir = lastSlash >= 0 ? mtlPath.substring(0, lastSlash + 1) : '';

    try {
      const mtlLoader = new MTLLoader();
      mtlLoader.setPath('/api/preview/file/');
      mtlLoader.setResourcePath(`/api/preview/file/${mtlDir}`);
      materials = await new Promise((resolve, reject) => {
        mtlLoader.load(mtlPathWithRevision, resolve, undefined, reject);
      });
      if (stage._loadGeneration !== gen) return false;
      materials.preload();
    } catch (e) {
      if (stage._loadGeneration !== gen) return false;
      console.warn(`[preview] MTL load failed for stage ${stageNum}:`, e);
    }

    try {
      const objLoader = new OBJLoader();
      if (materials) objLoader.setMaterials(materials);

      const object = await new Promise((resolve, reject) => {
        objLoader.load(url, resolve, undefined, reject);
      });

      if (stage._loadGeneration !== gen) {
        // Dispose the loaded object to avoid memory leaks
        object.traverse((node) => {
          if (node.geometry) node.geometry.dispose();
          if (node.material) {
            if (Array.isArray(node.material)) node.material.forEach(m => m?.dispose());
            else node.material.dispose();
          }
        });
        return false;
      }

      const box = new THREE.Box3().setFromObject(object);
      const center = box.getCenter(new THREE.Vector3());
      stage.centerOffset = center.clone();
      object.position.sub(center);

      if (!materials) {
        const palette = SCENE_THEMES[this._currentTheme];
        object.traverse((node) => {
          if (node.isMesh) {
            node.material = new THREE.MeshStandardMaterial({
              color: palette.meshColor,
              roughness: 0.9,
              metalness: 0.02,
              side: THREE.DoubleSide,
            });
          }
        });
      }

      this._cleanupCurrentObject(stage);
      stage.currentObject = object;
      stage.sceneRoot.add(object);
      // clearFromStage() can hide initialized containers; show it again on successful load.
      stage.container?.classList.add('visible');
      this._fitCamera(stage, box);
      return true;
    } catch (e) {
      if (stage._loadGeneration !== gen) return false;
      console.error(`Failed to load OBJ (stage ${stageNum}):`, e);
      return false;
    }
  }

  /**
   * Load a translucent PLY overlay without replacing the current stage asset.
   */
  async loadStageOverlay(stageNum, relativePath, opts = {}) {
    await this.initSceneForStage(stageNum);
    await this._ensureSceneFlipForStage(stageNum, opts.cacheToken ?? null);
    this.activateStage(stageNum);

    const stage = this._stages[stageNum];
    if (!stage) return false;
    this._cleanupOverlayObject(stage);

    const url = this._buildPreviewFileUrl(relativePath, opts.cacheToken);
    const loader = new PLYLoader();

    try {
      const geometry = await new Promise((resolve, reject) => {
        loader.load(url, resolve, undefined, reject);
      });

      geometry.computeBoundingBox();
      if (stage.centerOffset) {
        geometry.translate(-stage.centerOffset.x, -stage.centerOffset.y, -stage.centerOffset.z);
      }
      geometry.computeBoundingBox();

      const hasColor = geometry.hasAttribute('color');
      let overlayObject;
      if (geometry.index && geometry.index.count > 0) {
        const material = new THREE.MeshStandardMaterial({
          vertexColors: hasColor,
          color: hasColor ? undefined : (opts.color ?? 0xff5533),
          transparent: true,
          opacity: Number.isFinite(opts.opacity) ? opts.opacity : 0.68,
          side: THREE.DoubleSide,
          depthWrite: false,
        });
        overlayObject = new THREE.Mesh(geometry, material);
      } else {
        const material = new THREE.PointsMaterial({
          vertexColors: hasColor,
          color: hasColor ? undefined : (opts.color ?? 0xff5533),
          transparent: true,
          opacity: Number.isFinite(opts.opacity) ? opts.opacity : 0.85,
          size: 0.006,
          sizeAttenuation: true,
          depthWrite: false,
        });
        overlayObject = new THREE.Points(geometry, material);
      }
      overlayObject.renderOrder = 10;
      stage.overlayObject = overlayObject;
      stage.sceneRoot.add(overlayObject);
      return true;
    } catch (e) {
      console.error(`Failed to load overlay PLY (stage ${stageNum}):`, e);
      return false;
    }
  }

  clearStageOverlay(stageNum) {
    const stage = this._stages[stageNum];
    if (!stage) return;
    this._cleanupOverlayObject(stage);
  }

  /**
   * Load extracted frames into Stage 1 gallery (representative 10 frames).
   * @param {number} frameCount - total extracted frame count from backend
   */
  async loadGallery(frameCount) {
    try {
      const res = await fetch('/api/preview/outputs', { cache: 'no-store' });
      const data = await res.json();
      this._galleryGrid.innerHTML = '';
      this._previewAssetRevision += 1;
      const rev = this._previewAssetRevision;

      const images = data.files.filter(
        f => ['.png', '.jpg'].includes(f.ext) && f.path.startsWith('frames/')
      );
      images.sort((a, b) => a.path.localeCompare(b.path));

      // Show frame count header
      const totalCount = frameCount || images.length;
      const headerEl = document.getElementById('frame-count-header');
      const textEl = document.getElementById('frame-count-text');
      if (headerEl && textEl) {
        textEl.textContent = `Extracted ${totalCount} frames`;
        headerEl.classList.remove('hidden');
      }

      // Select up to 10 representative frames (evenly spaced)
      const maxShow = 10;
      let selected = images;
      if (images.length > maxShow) {
        const step = (images.length - 1) / (maxShow - 1);
        selected = Array.from({length: maxShow}, (_, i) =>
          images[Math.round(i * step)]
        );
      }

      for (const f of selected) {
        const img = document.createElement('img');
        img.className = 'gallery-thumb';
        img.src = `/api/preview/file/${f.path}?rev=${rev}`;
        img.alt = f.name;
        img.title = f.path;
        img.loading = 'lazy';
        this._galleryGrid.appendChild(img);
      }

      if (images.length === 0) {
        this._galleryGrid.innerHTML = '<p style="padding:20px;color:#555">No frames yet.</p>';
      }
    } catch (e) {
      console.error('Failed to load gallery:', e);
    }
  }
}

// ── Mixin: pose utilities (from preview/pose-utils.js) ────────────
Object.assign(PreviewPanel.prototype, {
  _poseJsonToArray: PoseUtils._poseJsonToArray,
  _applySceneFlipToLoadedStages: PoseUtils._applySceneFlipToLoadedStages,
  _ensureSceneFlipForStage: PoseUtils._ensureSceneFlipForStage,
  _resolveSceneFlipFromCameraPoses: PoseUtils._resolveSceneFlipFromCameraPoses,
  _forwardSignFromAlignment: PoseUtils._forwardSignFromAlignment,
  _normalizePoseConvention: PoseUtils._normalizePoseConvention,
  _shouldApplySceneFlipX: PoseUtils._shouldApplySceneFlipX,
  _meanCameraDownY: PoseUtils._meanCameraDownY,
  _invertPoseArray: PoseUtils._invertPoseArray,
  _evaluateForwardFromOrbitCenter: PoseUtils._evaluateForwardFromOrbitCenter,
  _computePoseCentroid: PoseUtils._computePoseCentroid,
  _scoreForwardTowardTarget: PoseUtils._scoreForwardTowardTarget,
  _inferForwardSignFromTarget: PoseUtils._inferForwardSignFromTarget,
});

// ── Mixin: scene helpers (from preview/scene-helpers.js) ──────────
Object.assign(PreviewPanel.prototype, {
  _disposeObject: SceneHelpers._disposeObject,
  _cleanupCurrentObject: SceneHelpers._cleanupCurrentObject,
  _cleanupOverlayObject: SceneHelpers._cleanupOverlayObject,
  _fitCamera: SceneHelpers._fitCamera,
  _fitCameraToStage: SceneHelpers._fitCameraToStage,
  _setMeshShadowProfile: SceneHelpers._setMeshShadowProfile,
  _applyStageCenterOffset: SceneHelpers._applyStageCenterOffset,
  showGroundPlane: SceneHelpers.showGroundPlane,
  _createGroundPlaneMesh: SceneHelpers._createGroundPlaneMesh,
  _removeGroundPlane: SceneHelpers._removeGroundPlane,
  toggleGroundPlane: SceneHelpers.toggleGroundPlane,
});
