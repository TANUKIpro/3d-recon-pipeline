/**
 * 3D preview manager — per-stage scene support for PLY/OBJ.
 *
 * Uses a single shared renderer that moves between stage containers.
 * Stage 2 has a dedicated scene for camera overlay support.
 */

import { clampMeshRepairThreshold } from './utils.js';
import {
  MESH_REPAIR_THRESHOLD_MIN,
  MESH_REPAIR_THRESHOLD_MAX,
  MESH_REPAIR_THRESHOLD_DEFAULT,
} from './constants.js';

let THREE;
let OrbitControls;
let PLYLoader;
let OBJLoader;
let MTLLoader;

const FIRST_MESH_PREVIEW_FILES = new Set([
  'object_mesh_raw.ply',
  'object_mesh_preview.ply',
  'object_mesh_poisson_raw.ply',
  'object_mesh_wrapped.ply',
  'object_mesh_repaired.ply',
]);

const SCENE_THEMES = {
  dark: {
    background:    0x0a0a14,
    gridPrimary:   0x333355,
    gridSecondary: 0x222244,
    meshColor:     0xb3b3b3,
    edgeColor:     0x101820,
    edgeOpacity:   0.35,
    pointColor:    0x4a9eff,
    shadowColor:   0x000000,
    shadowOpacity: 0.28,
    ambientInt:    0.6,
    dirInt:        0.8,
    shadowAmbientInt: 0.36,
    shadowDirInt:     1.05,
  },
  light: {
    background:    0xe8e8ef,
    gridPrimary:   0xc0c0d0,
    gridSecondary: 0xd5d5e0,
    meshColor:     0x6a6a7a,
    edgeColor:     0x888899,
    edgeOpacity:   0.25,
    pointColor:    0x2d7cd6,
    shadowColor:   0x444466,
    shadowOpacity: 0.15,
    ambientInt:    0.7,
    dirInt:        0.7,
    shadowAmbientInt: 0.5,
    shadowDirInt:     0.9,
  },
};

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
    this.onMeshRepairSelectionChanged = null;
    this._meshRepair = {
      active: false,
      confirmed: false,
      group: null,
      loopObjects: new Map(),
      loopMeta: new Map(),
      selected: new Set(),
      threshold: MESH_REPAIR_THRESHOLD_DEFAULT,
      clickHandler: (event) => this._handleMeshRepairClick(event),
      clickBound: false,
      raycaster: null,
      pointer: null,
      colors: {
        candidate: 0xf4d03f,
        selected: 0xe74c3c,
        confirmed: 0x2ecc71,
      },
    };
    this._cropBbox = {
      helper: null,        // THREE.LineSegments (OBB wireframe)
      obbData: null,       // { center, extent, rotation, adjustedCenter } from API
    };
    this._groundPlane = {
      meshByStage: {},   // stageNum -> THREE.Mesh
      data: null,        // parsed ground_plane.json
      visible: true,
    };
    this._sectionLoop = {
      group: null,       // THREE.Group added to stage 7 scene
    };
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
    this._clearMeshRepairOverlay();
    this.clearFromStage(1);
  }

  clearFromStage(startStage = 1) {
    const start = Math.max(1, Math.min(8, Number(startStage) || 1));

    if (start <= 3) {
      this._groundPlane.data = null;
    }

    if (start <= 1) {
      const empty = document.querySelector('#stage-panel-1 .stage-panel-empty');
      if (empty) empty.classList.remove('hidden');
      const headerEl = document.getElementById('frame-count-header');
      if (headerEl) headerEl.style.display = 'none';
      if (this._galleryGrid) this._galleryGrid.innerHTML = '';
    }

    if (start <= 2) {
      this._clearStageScene(2);
      const empty = document.getElementById('stage-2-empty');
      if (empty) empty.classList.remove('hidden');
      const toolbar = document.getElementById('pi3x-toolbar');
      if (toolbar) toolbar.style.display = 'none';
      const pointCount = document.getElementById('pi3x-point-count');
      if (pointCount) pointCount.textContent = '';
      const cameraCount = document.getElementById('pi3x-camera-count');
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

    if (start <= 7) {
      this._clearStageScene(7);
      const empty = document.getElementById('stage-7-empty');
      if (empty) empty.classList.remove('hidden');
    }

    if (start <= 8) {
      this._clearStageScene(8);
      const empty = document.getElementById('stage-8-empty');
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
      if (!this._meshRepair.raycaster) {
        this._meshRepair.raycaster = new THREE.Raycaster();
      }
      if (!this._meshRepair.pointer) {
        this._meshRepair.pointer = new THREE.Vector2();
      }
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

    // Stage 2 starts with OpenCV->OpenGL flip. Stages 4-8 use inferred flip when available.
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
    if (stageNum === 6) {
      this.clearCropBbox();
    }
    if (stageNum === 7) {
      this._clearMeshRepairOverlay();
      this._clearSectionLoop();
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

  _disposeObject(object3d) {
    if (!object3d) return;
    object3d.traverse((node) => {
      if (node.geometry && typeof node.geometry.dispose === 'function') {
        node.geometry.dispose();
      }
      if (!node.material) return;
      if (Array.isArray(node.material)) {
        for (const material of node.material) {
          if (material && typeof material.dispose === 'function') material.dispose();
        }
        return;
      }
      if (typeof node.material.dispose === 'function') node.material.dispose();
    });
  }

  _cleanupCurrentObject(stage) {
    if (stage.currentObject) {
      this._disposeObject(stage.currentObject);
      stage.sceneRoot.remove(stage.currentObject);
      stage.currentObject = null;
    }
  }

  _setMeshRepairClickEnabled(enabled) {
    const stage = this._stages[7];
    const container = stage?.container;
    if (!container) return;
    if (enabled) {
      if (!this._meshRepair.clickBound) {
        container.addEventListener('click', this._meshRepair.clickHandler);
        this._meshRepair.clickBound = true;
      }
      return;
    }
    if (this._meshRepair.clickBound) {
      container.removeEventListener('click', this._meshRepair.clickHandler);
      this._meshRepair.clickBound = false;
    }
  }

  _handleMeshRepairClick(event) {
    if (!this._meshRepair.active || this._meshRepair.confirmed) return;
    if (this._activeStage !== 7) return;

    const stage = this._stages[7];
    if (!stage || !this._meshRepair.raycaster || !this._meshRepair.pointer) return;
    const objects = Array.from(this._meshRepair.loopObjects.values())
      .filter((obj) => obj?.visible !== false);
    if (objects.length === 0) return;

    const rect = stage.container.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;

    this._meshRepair.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this._meshRepair.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    this._meshRepair.raycaster.params.Line = this._meshRepair.raycaster.params.Line || {};
    this._meshRepair.raycaster.params.Line.threshold = 0.03;
    this._meshRepair.raycaster.setFromCamera(this._meshRepair.pointer, stage.camera);
    const hit = this._meshRepair.raycaster.intersectObjects(objects, false)
      .find((item) => item?.object?.userData?.loopId != null);
    if (!hit) return;

    const loopId = Number(hit.object.userData.loopId);
    if (!Number.isFinite(loopId)) return;
    if (this._meshRepair.selected.has(loopId)) {
      this._meshRepair.selected.delete(loopId);
    } else {
      this._meshRepair.selected.add(loopId);
    }
    this._applyMeshRepairVisibility();
    this._updateMeshRepairColors();
    this.onMeshRepairSelectionChanged?.(this.getMeshRepairSelectedLoopIds());
  }

  _isMeshRepairLoopVisible(loopId) {
    if (this._meshRepair.selected.has(loopId)) return true;
    const meta = this._meshRepair.loopMeta.get(loopId);
    const normalY = Number(meta?.normalY);
    if (!Number.isFinite(normalY)) return true;
    return normalY <= this._meshRepair.threshold;
  }

  _applyMeshRepairVisibility() {
    let visible = 0;
    for (const [loopId, obj] of this._meshRepair.loopObjects.entries()) {
      if (!obj) continue;
      const isVisible = this._isMeshRepairLoopVisible(loopId);
      obj.visible = isVisible;
      if (isVisible) visible += 1;
    }
    return visible;
  }

  _updateMeshRepairColors() {
    for (const [loopId, obj] of this._meshRepair.loopObjects.entries()) {
      if (!obj?.material) continue;
      let colorHex = this._meshRepair.colors.candidate;
      if (this._meshRepair.selected.has(loopId)) {
        colorHex = this._meshRepair.confirmed
          ? this._meshRepair.colors.confirmed
          : this._meshRepair.colors.selected;
      }
      obj.material.color.setHex(colorHex);
      obj.material.opacity = this._meshRepair.selected.has(loopId) ? 1.0 : 0.9;
    }
  }

  _clearMeshRepairOverlay() {
    this._setMeshRepairClickEnabled(false);
    if (this._meshRepair.group) {
      this._meshRepair.group.traverse((node) => {
        if (node?.geometry?.dispose) node.geometry.dispose();
        if (!node?.material) return;
        if (Array.isArray(node.material)) {
          for (const mat of node.material) {
            if (mat?.dispose) mat.dispose();
          }
          return;
        }
        if (node.material.dispose) node.material.dispose();
      });
      const stage = this._stages[7];
      if (stage?.sceneRoot) {
        stage.sceneRoot.remove(this._meshRepair.group);
      }
    }

    this._meshRepair.group = null;
    this._meshRepair.loopObjects = new Map();
    this._meshRepair.loopMeta = new Map();
    this._meshRepair.selected = new Set();
    this._meshRepair.active = false;
    this._meshRepair.confirmed = false;
    this.onMeshRepairSelectionChanged?.([]);
  }

  clearMeshRepairSelection() {
    if (!this._meshRepair.active) return;
    this._meshRepair.confirmed = false;
    this._meshRepair.selected.clear();
    this._applyMeshRepairVisibility();
    this._updateMeshRepairColors();
    this.onMeshRepairSelectionChanged?.(this.getMeshRepairSelectedLoopIds());
  }

  setMeshRepairConfirmed() {
    if (!this._meshRepair.active) return;
    this._meshRepair.confirmed = true;
    this._updateMeshRepairColors();
    this._setMeshRepairClickEnabled(false);
    this.onMeshRepairSelectionChanged?.(this.getMeshRepairSelectedLoopIds());
  }

  getMeshRepairSelectedLoopIds() {
    return Array.from(this._meshRepair.selected).sort((a, b) => a - b);
  }

  getMeshRepairVisibleLoopCount() {
    let visible = 0;
    for (const obj of this._meshRepair.loopObjects.values()) {
      if (obj?.visible !== false) visible += 1;
    }
    return visible;
  }

  getMeshRepairTotalLoopCount() {
    return this._meshRepair.loopObjects.size;
  }

  getMeshRepairThreshold() {
    return this._meshRepair.threshold;
  }

  setMeshRepairThreshold(value) {
    this._meshRepair.threshold = clampMeshRepairThreshold(value);
    if (!this._meshRepair.active) return;
    this._applyMeshRepairVisibility();
    this._updateMeshRepairColors();
    this.onMeshRepairSelectionChanged?.(this.getMeshRepairSelectedLoopIds());
  }

  /**
   * Show crop bounding box overlay on Stage 6 scene.
   * Loads Stage 5 output mesh into Stage 6 and draws an OBB wireframe.
   */
  async showCropBbox(cropScale, opts = {}) {
    const preferPreview = opts.preferPreview === true;
    const preferredFile = preferPreview ? 'object_mesh_preview.ply' : 'object_mesh.ply';
    let loaded = await this.loadStageResult(6, {
      file: preferredFile,
      renderMode: 'mesh',
      stripVertexColors: true,
      enableShadows: true,
    });
    if (!loaded && preferPreview) {
      loaded = await this.loadStageResult(6, {
        file: 'object_mesh.ply',
        renderMode: 'mesh',
        stripVertexColors: true,
        enableShadows: true,
      });
    }
    if (!loaded) return;

    const stage = this._stages[6];
    if (!stage?.currentObject) return;

    // Fetch OBB data from backend
    try {
      const resp = await fetch('/api/preview/crop-obb');
      if (!resp.ok) return;
      const obb = await resp.json();

      // Adjust OBB center by the stage centering offset
      const adjusted = this._applyStageCenterOffset(stage, obb.center);
      obb.adjustedCenter = adjusted || [0, 0, 0];
      this._cropBbox.obbData = obb;
    } catch (_) {
      return;
    }

    this._createCropBboxHelper(cropScale);

    const empty = document.getElementById('stage-6-empty');
    if (empty) empty.classList.add('hidden');
  }

  _createCropBboxHelper(cropScale) {
    const stage = this._stages[6];
    const obb = this._cropBbox.obbData;
    if (!stage?.sceneRoot || !obb) return;

    // Remove stale helper if exists
    if (this._cropBbox.helper) {
      stage.sceneRoot.remove(this._cropBbox.helper);
      this._disposeObject(this._cropBbox.helper);
      this._cropBbox.helper = null;
    }

    const scale = Math.max(0.01, Number(cropScale) || 1.0);
    const [ex, ey, ez] = obb.extent;

    // EdgesGeometry for clean wireframe box
    const boxGeom = new THREE.BoxGeometry(ex * scale, ey * scale, ez * scale);
    const edges = new THREE.EdgesGeometry(boxGeom);
    boxGeom.dispose();  // no longer needed after EdgesGeometry is built
    const mat = new THREE.LineBasicMaterial({
      color: 0x00ccff, transparent: true, opacity: 0.7,
    });
    const helper = new THREE.LineSegments(edges, mat);

    // Apply OBB position (adjusted for stage centering)
    helper.position.set(...obb.adjustedCenter);

    // 3x3 rotation matrix -> THREE.Matrix4 -> quaternion
    const R = obb.rotation;
    const m4 = new THREE.Matrix4().set(
      R[0][0], R[0][1], R[0][2], 0,
      R[1][0], R[1][1], R[1][2], 0,
      R[2][0], R[2][1], R[2][2], 0,
      0,       0,       0,       1,
    );
    helper.quaternion.setFromRotationMatrix(m4);

    stage.sceneRoot.add(helper);
    this._cropBbox.helper = helper;
  }

  updateCropBbox(cropScale) {
    if (!this._cropBbox.obbData) return;

    const stage = this._stages[6];
    if (!stage?.sceneRoot) return;

    // Remove old helper
    if (this._cropBbox.helper) {
      stage.sceneRoot.remove(this._cropBbox.helper);
      this._disposeObject(this._cropBbox.helper);
      this._cropBbox.helper = null;
    }

    this._createCropBboxHelper(cropScale);
  }

  clearCropBbox() {
    const stage = this._stages[6];
    if (this._cropBbox.helper) {
      if (stage?.sceneRoot) {
        stage.sceneRoot.remove(this._cropBbox.helper);
      }
      this._disposeObject(this._cropBbox.helper);
      this._cropBbox.helper = null;
    }
    this._cropBbox.obbData = null;
  }

  _applyStageCenterOffset(stage, point3) {
    if (!Array.isArray(point3) || point3.length < 3) return null;
    const x = Number(point3[0]);
    const y = Number(point3[1]);
    const z = Number(point3[2]);
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return null;

    const offset = stage?.centerOffset;
    if (!offset) return [x, y, z];
    return [x - offset.x, y - offset.y, z - offset.z];
  }

  async beginMeshRepairSelection(payload = {}) {
    const loops = Array.isArray(payload?.loops) ? payload.loops : [];
    if (loops.length === 0) {
      throw new Error('No mesh-repair loop candidates');
    }

    await this.initSceneForStage(7);
    await this._ensureSceneFlipForStage(7);
    this.activateStage(7);

    const sourceFile = String(payload?.source_mesh_relpath || '').trim()
      || String(payload?.source_mesh_path || '').trim().split('/').pop()
      || 'object_mesh_wrapped.ply';
    const loaded = await this.loadStageResult(7, {
      file: sourceFile,
      renderMode: 'mesh',
      stripVertexColors: true,
      enableShadows: true,
    });
    if (!loaded) {
      throw new Error(`Failed to load source mesh for repair selection: ${sourceFile}`);
    }

    this._clearMeshRepairOverlay();
    const stage = this._stages[7];
    if (!stage?.sceneRoot) {
      throw new Error('Stage 7 scene is not available');
    }

    const group = new THREE.Group();
    stage.sceneRoot.add(group);

    let valid = 0;
    for (const loop of loops) {
      const loopId = Number(loop?.loop_id);
      const normalY = Number(loop?.normal_y);
      const points = Array.isArray(loop?.points) ? loop.points : [];
      if (!Number.isFinite(loopId) || points.length < 3) continue;

      const positions = [];
      for (let i = 0; i < points.length; i++) {
        const a = points[i];
        const b = points[(i + 1) % points.length];
        const p0 = this._applyStageCenterOffset(stage, a);
        const p1 = this._applyStageCenterOffset(stage, b);
        if (!p0 || !p1) continue;
        positions.push(p0[0], p0[1], p0[2]);
        positions.push(p1[0], p1[1], p1[2]);
      }
      if (positions.length < 6) continue;

      const geom = new THREE.BufferGeometry();
      geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
      const mat = new THREE.LineBasicMaterial({
        color: this._meshRepair.colors.candidate,
        transparent: true,
        opacity: 0.9,
      });
      const line = new THREE.LineSegments(geom, mat);
      line.userData.loopId = loopId;
      group.add(line);
      this._meshRepair.loopObjects.set(loopId, line);
      this._meshRepair.loopMeta.set(loopId, {
        normalY: Number.isFinite(normalY) ? normalY : null,
      });
      valid += 1;
    }

    if (valid === 0) {
      stage.sceneRoot.remove(group);
      this._clearMeshRepairOverlay();
      throw new Error('No valid mesh-repair loops to render');
    }

    this._meshRepair.group = group;
    this._meshRepair.active = true;
    this._meshRepair.confirmed = false;
    this._meshRepair.selected.clear();
    this._applyMeshRepairVisibility();
    this._updateMeshRepairColors();
    this._setMeshRepairClickEnabled(true);
    this.onMeshRepairSelectionChanged?.(this.getMeshRepairSelectedLoopIds());
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
   * Load Pi3X results: point cloud + camera poses.
   * @param {CameraOverlay} cameraOverlay - camera overlay instance
   * @param {string} [plyFile='object_full.ply'] - PLY filename to load
   */
  async loadPi3xResults(cameraOverlay, plyFile = 'object_full.ply') {
    await this.initSceneForStage(2);
    this.activateStage(2);

    const empty = document.getElementById('stage-2-empty');
    const toolbar = document.getElementById('pi3x-toolbar');
    if (toolbar) toolbar.style.display = 'none';

    const stage = this._stages[2];
    if (!stage) return;
    stage.centerOffset = null;
    cameraOverlay.remove();
    const cacheToken = this._nextPreviewRevision();

    // Load point cloud
    const loaded = await this._loadPLYIntoStage(2, plyFile, { cacheToken });
    if (!loaded) {
      console.warn(`Pi3X point cloud not ready: ${plyFile}`);
      if (empty) empty.classList.remove('hidden');
      const pointCountEl = document.getElementById('pi3x-point-count');
      if (pointCountEl) pointCountEl.textContent = '';
    } else {
      if (empty) empty.classList.add('hidden');
      if (toolbar) toolbar.style.display = 'flex';
      if (stage.currentObject?.geometry) {
        const count = stage.currentObject.geometry.attributes.position?.count || 0;
        const el = document.getElementById('pi3x-point-count');
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
        console.info('Pi3X scene flip X(pi):', sceneFlipX);
        console.info('Camera overlay forwardSign:', forwardSign, forwardSignSource);

        cameraOverlay.create(THREE, stage.sceneRoot, poseArray, { forwardSign });
        cameraLoaded = true;
        stage.container?.classList.add('visible');

        // Apply same centering offset as point cloud
        if (stage.centerOffset) {
          cameraOverlay.applyOffset(stage.centerOffset);
        }

        // Display camera count
        const countEl = document.getElementById('pi3x-camera-count');
        if (countEl) countEl.textContent = `${poseArray.length} cameras`;

        // Bind toggle
        const toggle = document.getElementById('pi3x-cameras-toggle');
        if (toggle) {
          toggle.onchange = () => cameraOverlay.setVisible(toggle.checked);
        }
        if (empty) empty.classList.add('hidden');
        if (toolbar) toolbar.style.display = 'flex';
      } else {
        cameraOverlay.remove();
      }
    } catch (e) {
      cameraOverlay.remove();
      console.error('Failed to load camera poses:', e);
    }

    if (!cameraLoaded) {
      const countEl = document.getElementById('pi3x-camera-count');
      if (countEl) countEl.textContent = '';
    }
    if (!loaded && !cameraLoaded) {
      if (empty) empty.classList.remove('hidden');
      if (toolbar) toolbar.style.display = 'none';
      stage.container?.classList.remove('visible');
    }
  }

  _poseJsonToArray(data) {
    const poses = Array.isArray(data?.poses) ? data.poses : [];
    const frameIndices = Array.isArray(data?.frame_indices) ? data.frame_indices : [];
    return poses.map((mat4x4, i) => {
      // Transpose: row-major (Python) → column-major (three.js Matrix4.fromArray)
      const flat = new Array(16);
      for (let r = 0; r < 4; r++)
        for (let c = 0; c < 4; c++)
          flat[c * 4 + r] = mat4x4[r][c];
      return { matrix: flat, frame_index: frameIndices[i] ?? i };
    });
  }

  _applySceneFlipToLoadedStages(sceneFlipX) {
    const rotationX = sceneFlipX ? Math.PI : 0;
    for (const stageNum of [2, 4, 5, 6, 7, 8]) {
      const stage = this._stages[stageNum];
      if (stage?.sceneRoot) stage.sceneRoot.rotation.x = rotationX;
    }
  }

  async _ensureSceneFlipForStage(stageNum, cacheToken = null) {
    if (stageNum < 4 || stageNum > 8) return;
    if (this._sceneFlipX == null) {
      this._sceneFlipX = await this._resolveSceneFlipFromCameraPoses(cacheToken);
    }
    const stage = this._stages[stageNum];
    if (stage?.sceneRoot) {
      stage.sceneRoot.rotation.x = this._sceneFlipX ? Math.PI : 0;
    }
  }

  async _resolveSceneFlipFromCameraPoses(cacheToken = null) {
    try {
      const res = await fetch(this._buildPreviewFileUrl('camera_poses.json', cacheToken));
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const data = await res.json();
      const rawPoseArray = this._poseJsonToArray(data);
      const normalized = this._normalizePoseConvention(rawPoseArray);
      return this._shouldApplySceneFlipX(normalized.poseArray, data.alignment);
    } catch (e) {
      console.warn('Failed to infer scene flip from camera poses:', e);
      return false;
    }
  }

  _forwardSignFromAlignment(alignment) {
    if (!alignment || typeof alignment !== 'object') return null;
    const axis = String(alignment.inferred_forward_axis || '').toLowerCase();
    if (axis.includes('-z') || axis.includes('-col 2')) return -1;
    if (axis.includes('+z') || axis.includes('+col 2') || axis.includes('col 2')) return 1;
    return null;
  }

  _normalizePoseConvention(poseArray) {
    if (!poseArray || poseArray.length === 0) {
      return {
        poseArray: [],
        convention: 'c2w',
        forwardSign: 1,
        confidence: 0,
        used: 0,
      };
    }

    const c2wEval = this._evaluateForwardFromOrbitCenter(poseArray);
    const w2cAsC2w = this._invertPoseArray(poseArray);
    const w2cEval = this._evaluateForwardFromOrbitCenter(w2cAsC2w);

    // Prefer c2w unless w2c interpretation is meaningfully more consistent.
    const chooseW2C = (w2cEval.confidence - c2wEval.confidence) > 0.05;
    if (chooseW2C) {
      return {
        poseArray: w2cAsC2w,
        convention: 'w2c->c2w',
        forwardSign: w2cEval.forwardSign,
        confidence: w2cEval.confidence,
        used: w2cEval.used,
      };
    }

    return {
      poseArray,
      convention: 'c2w',
      forwardSign: c2wEval.forwardSign,
      confidence: c2wEval.confidence,
      used: c2wEval.used,
    };
  }

  _shouldApplySceneFlipX(poseArray, alignment) {
    const meanDownY = this._meanCameraDownY(poseArray);
    if (meanDownY != null && Math.abs(meanDownY) >= 0.15) {
      // If camera-down is +Y, data is likely in OpenCV Y-down world => flip.
      return meanDownY > 0;
    }
    if (alignment && alignment.applied === true) {
      // Aligned outputs are expected to already use +Y as up.
      return false;
    }
    return true;
  }

  _meanCameraDownY(poseArray) {
    if (!poseArray || poseArray.length === 0) return null;
    const mat = new THREE.Matrix4();
    const basisX = new THREE.Vector3();
    const basisY = new THREE.Vector3();
    const basisZ = new THREE.Vector3();
    let sumY = 0;
    let used = 0;

    for (const pose of poseArray) {
      mat.fromArray(pose.matrix);
      mat.extractBasis(basisX, basisY, basisZ);
      basisY.normalize(); // camera +Y (down)
      sumY += basisY.y;
      used += 1;
    }

    if (used === 0) return null;
    return sumY / used;
  }

  _invertPoseArray(poseArray) {
    const mat = new THREE.Matrix4();
    const inv = new THREE.Matrix4();
    return poseArray.map((pose) => {
      mat.fromArray(pose.matrix);
      inv.copy(mat).invert();
      return { matrix: inv.toArray(), frame_index: pose.frame_index };
    });
  }

  _evaluateForwardFromOrbitCenter(poseArray) {
    const center = this._computePoseCentroid(poseArray);
    if (!center) {
      return { forwardSign: 1, confidence: 0, used: 0 };
    }
    return this._scoreForwardTowardTarget(poseArray, center);
  }

  _computePoseCentroid(poseArray) {
    if (!poseArray || poseArray.length === 0) return null;
    const mat = new THREE.Matrix4();
    const position = new THREE.Vector3();
    const center = new THREE.Vector3();
    let used = 0;

    for (const pose of poseArray) {
      mat.fromArray(pose.matrix);
      position.setFromMatrixPosition(mat);
      center.add(position);
      used += 1;
    }

    if (used === 0) return null;
    return center.divideScalar(used);
  }

  _scoreForwardTowardTarget(poseArray, targetCenter) {
    if (!poseArray || poseArray.length === 0) {
      return { forwardSign: 1, confidence: 0, used: 0 };
    }

    const mat = new THREE.Matrix4();
    const position = new THREE.Vector3();
    const basisX = new THREE.Vector3();
    const basisY = new THREE.Vector3();
    const basisZ = new THREE.Vector3();
    const toTarget = new THREE.Vector3();
    const center = targetCenter ? targetCenter.clone() : new THREE.Vector3(0, 0, 0);

    let score = 0;
    let used = 0;
    for (const pose of poseArray) {
      mat.fromArray(pose.matrix);
      position.setFromMatrixPosition(mat);
      toTarget.copy(center).sub(position);
      const len = toTarget.length();
      if (len < 1e-6) continue;
      toTarget.divideScalar(len);

      mat.extractBasis(basisX, basisY, basisZ);
      basisZ.normalize();
      score += basisZ.dot(toTarget);
      used += 1;
    }

    if (used === 0) {
      return { forwardSign: 1, confidence: 0, used: 0 };
    }

    const avg = score / used;
    return {
      forwardSign: avg < 0 ? -1 : 1,
      confidence: Math.abs(avg),
      used,
    };
  }

  _inferForwardSignFromTarget(poseArray, targetCenter) {
    return this._scoreForwardTowardTarget(poseArray, targetCenter).forwardSign;
  }

  _nextPreviewRevision() {
    this._previewAssetRevision += 1;
    return this._previewAssetRevision;
  }

  _buildPreviewFileUrl(relativePath, cacheToken = null) {
    const base = `/api/preview/file/${relativePath}`;
    if (cacheToken == null || cacheToken === '') return base;
    return `${base}?rev=${encodeURIComponent(String(cacheToken))}`;
  }

  _classicalPhaseDescriptor(step) {
    const key = String(step || '').toLowerCase();
    const map = {
      preprocess: { file: 'object_mesh_input.ply', renderMode: 'points' },
      main: { file: 'object_mesh_raw.ply', renderMode: 'mesh' },
      postprocess: { file: 'object_mesh_postprocessed.ply', renderMode: 'mesh' },
      downsample: { file: 'object_mesh.ply', renderMode: 'mesh' },
    };
    return map[key] || null;
  }

  async loadClassicalPhase(step, opts = {}) {
    const descriptor = this._classicalPhaseDescriptor(step);
    if (!descriptor) return false;
    const isMeshPhase = descriptor.renderMode === 'mesh';
    return this.loadStageResult(5, {
      file: descriptor.file,
      renderMode: descriptor.renderMode,
      cacheToken: opts.cacheToken,
      stripVertexColors: opts.stripVertexColors ?? isMeshPhase,
      enableShadows: opts.enableShadows ?? isMeshPhase,
    });
  }

  /**
   * Auto-load the appropriate result file for a stage.
   */
  async loadStageResult(stageNum, opts = {}) {
    const fileMap = {
      4: 'object_denoised.ply',
      5: 'object_mesh.ply',
      6: 'object_mesh_wrapped.ply',
      7: 'object_mesh_repaired.ply',
      8: 'textured_mesh.obj',
    };

    const overrideFile = String(opts.file || '').trim();
    const defaultFile = fileMap[stageNum];
    const preferPreview = !overrideFile && stageNum === 5 && opts.preferPreview === true;
    const file = overrideFile || (preferPreview ? 'object_mesh_preview.ply' : defaultFile);
    if (!file) return false;
    const fileName = file.split('/').pop()?.toLowerCase() || '';
    const isFirstMeshPreview = FIRST_MESH_PREVIEW_FILES.has(fileName);

    const cacheToken = opts.cacheToken ?? this._nextPreviewRevision();
    await this.initSceneForStage(stageNum);
    await this._ensureSceneFlipForStage(stageNum, cacheToken);
    this.activateStage(stageNum);

    // Toggle empty placeholder based on load result.
    const empty = document.getElementById(`stage-${stageNum}-empty`);

    const ext = file.split('.').pop().toLowerCase();
    const renderMode = String(opts.renderMode || (isFirstMeshPreview ? 'mesh' : '')).toLowerCase();
    const stripVertexColors = opts.stripVertexColors ?? (isFirstMeshPreview && stageNum !== 7);
    const enableShadows = opts.enableShadows ?? isFirstMeshPreview;
    let loaded = await this._loadPreviewAssetForStage(stageNum, file, {
      ext,
      cacheToken,
      renderMode,
      stripVertexColors: stripVertexColors === true,
      enableShadows: enableShadows === true,
    });
    if (!loaded && preferPreview && defaultFile && file !== defaultFile) {
      loaded = await this._loadPreviewAssetForStage(stageNum, defaultFile, {
        cacheToken,
        renderMode,
        stripVertexColors: stripVertexColors === true,
        enableShadows: enableShadows === true,
      });
    }
    if (loaded && stageNum >= 4) {
      this.showGroundPlane(stageNum).catch(() => {});
    }
    if (loaded && stageNum === 7) {
      this._showSectionLoop().catch(() => {});
    }
    if (empty) empty.classList.toggle('hidden', loaded);
    return loaded;
  }

  async showGroundPlane(stageNum) {
    const stage = this._stages[stageNum];
    if (!stage) return;

    if (!this._groundPlane.data) {
      try {
        const resp = await fetch(`/api/preview/file/ground_plane.json?_t=${Date.now()}`);
        if (!resp.ok) return;
        this._groundPlane.data = await resp.json();
      } catch { return; }
    }

    this._createGroundPlaneMesh(stageNum);
  }

  _createGroundPlaneMesh(stageNum) {
    const stage = this._stages[stageNum];
    const data = this._groundPlane.data;
    if (!stage || !data || !THREE) return;

    // Remove existing mesh for this stage if any
    this._removeGroundPlane(stageNum);

    const extent = data.extent || [1, 1, 1];
    const planeSize = Math.max(...extent) * 1.5;

    const geometry = new THREE.PlaneGeometry(planeSize, planeSize);
    const material = new THREE.MeshStandardMaterial({
      color: 0x2ecc71,
      transparent: true,
      opacity: 0.25,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const mesh = new THREE.Mesh(geometry, material);

    // Orient plane: rotate from default Z-up to match the ground normal
    const normal = data.normal || [0, 0, 1];
    const from = new THREE.Vector3(0, 0, 1);
    const to = new THREE.Vector3(normal[0], normal[1], normal[2]).normalize();
    const quat = new THREE.Quaternion().setFromUnitVectors(from, to);
    mesh.quaternion.copy(quat);

    // Position: project the model center (origin in display space) onto the
    // ground plane so the plane always appears directly under the model,
    // regardless of how far the ground centroid is from the object.
    const offset = stage?.centerOffset;
    if (offset) {
      const nx = normal[0], ny = normal[1], nz = normal[2];
      const d = Number(data.d) || 0;
      // Plane in display coords: n·p + (n·offset + d) = 0
      const dAdj = nx * offset.x + ny * offset.y + nz * offset.z + d;
      // Closest point on plane to origin: -dAdj * n  (n is unit length)
      mesh.position.set(-dAdj * nx, -dAdj * ny, -dAdj * nz);
    } else {
      // No centering offset — fall back to raw ground center
      const c = data.center || [0, 0, 0];
      mesh.position.set(c[0], c[1], c[2]);
    }

    mesh.renderOrder = -1;
    mesh.visible = this._groundPlane.visible;
    stage.sceneRoot.add(mesh);
    this._groundPlane.meshByStage[stageNum] = mesh;
  }

  _removeGroundPlane(stageNum) {
    const mesh = this._groundPlane.meshByStage[stageNum];
    if (!mesh) return;
    mesh.parent?.remove(mesh);
    mesh.geometry?.dispose();
    mesh.material?.dispose();
    delete this._groundPlane.meshByStage[stageNum];
  }

  toggleGroundPlane(visible) {
    this._groundPlane.visible = visible;
    for (const mesh of Object.values(this._groundPlane.meshByStage)) {
      if (mesh) mesh.visible = visible;
    }
  }

  _clearSectionLoop() {
    const grp = this._sectionLoop.group;
    if (!grp) return;
    grp.traverse((node) => {
      if (node?.geometry?.dispose) node.geometry.dispose();
      if (node?.material?.dispose) node.material.dispose();
    });
    grp.parent?.remove(grp);
    this._sectionLoop.group = null;
  }

  async _showSectionLoop() {
    this._clearSectionLoop();
    const stage = this._stages[7];
    if (!stage) return;

    let data;
    try {
      const resp = await fetch(
        `/api/preview/file/contact_hole_repair/section_loop.json?_t=${Date.now()}`
      );
      if (!resp.ok) return;
      data = await resp.json();
    } catch { return; }

    const samples = data.samples;
    if (!samples || !samples.length) return;

    const selectedShift = data.selected_shift;
    const offset = stage.centerOffset;
    const group = new THREE.Group();

    for (const sample of samples) {
      const loops = sample.loops;
      if (!loops || !loops.length) continue;

      const isSelected = Math.abs(sample.shift - selectedShift) < 1e-6;
      // Below selected = removed region (cyan); at selected = yellow;
      // above selected = kept region (dim green)
      let color, opacity;
      if (isSelected) {
        color = 0xffee00; opacity = 1.0;
      } else if (sample.shift < selectedShift) {
        color = 0x00ddff; opacity = 0.6;
      } else {
        color = 0x44cc44; opacity = 0.3;
      }

      for (let li = 0; li < loops.length; li++) {
        const pts = loops[li].points;
        if (!pts || pts.length < 2) continue;

        const positions = new Float32Array(pts.length * 3);
        for (let i = 0; i < pts.length; i++) {
          positions[i * 3]     = pts[i][0] - (offset?.x ?? 0);
          positions[i * 3 + 1] = pts[i][1] - (offset?.y ?? 0);
          positions[i * 3 + 2] = pts[i][2] - (offset?.z ?? 0);
        }
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));

        // Secondary loops within same sample are dimmer
        const loopOpacity = li === 0 ? opacity : opacity * 0.4;
        const mat = new THREE.LineBasicMaterial({
          color,
          linewidth: 2,
          depthTest: false,
          transparent: true,
          opacity: loopOpacity,
        });
        const line = new THREE.LineLoop(geo, mat);
        line.renderOrder = 999;
        group.add(line);
      }
    }

    stage.sceneRoot.add(group);
    this._sectionLoop.group = group;
  }

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
   * Load a PLY file into a specific stage's scene.
   */
  async _loadPLYIntoStage(stageNum, relativePath, opts = {}) {
    const stage = this._stages[stageNum];
    if (!stage) return false;
    const gen = ++stage._loadGeneration;
    if (stageNum === 6) {
      this.clearCropBbox();
    }
    if (stageNum === 7) {
      this._clearMeshRepairOverlay();
      this._clearSectionLoop();
    }

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
      const center = new THREE.Vector3();
      geometry.boundingBox.getCenter(center);
      geometry.translate(-center.x, -center.y, -center.z);
      stage.centerOffset = center;
      // Recompute bounds after centering so the camera fit uses the updated box.
      geometry.computeBoundingBox();

      let obj;
      const mode = String(opts.renderMode || '').toLowerCase();
      const forcePoints = mode === 'points';
      const forceMesh = mode === 'mesh';
      const renderAsMesh = forceMesh || (!forcePoints && (stageNum === 5 || (geometry.index && geometry.index.count > 0)));
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

        // Lightweight edge overlay to reveal silhouette when no texture is present.
        let overlay = null;
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
      this._fitCamera(stage, geometry.boundingBox);
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

    try {
      const mtlLoader = new MTLLoader();
      mtlLoader.setPath('/api/preview/file/');
      materials = await new Promise((resolve, reject) => {
        mtlLoader.load(mtlPathWithRevision, resolve, undefined, reject);
      });
      if (stage._loadGeneration !== gen) return false;
      materials.preload();
    } catch (e) {
      if (stage._loadGeneration !== gen) return false;
      // No MTL file — okay
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
      object.position.sub(center);

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
   * Configure optional shadow-focused lighting profile for mesh previews.
   */
  _setMeshShadowProfile(stage, box, enabled) {
    if (!stage) return;
    const p = SCENE_THEMES[this._currentTheme];
    if (stage.ambientLight) {
      stage.ambientLight.intensity = enabled ? p.shadowAmbientInt : p.ambientInt;
    }
    if (stage.keyLight) {
      stage.keyLight.intensity = enabled ? p.shadowDirInt : p.dirInt;
    }

    const floor = stage.shadowFloor;
    if (!floor) return;
    if (!enabled || !box) {
      floor.visible = false;
      return;
    }

    const size = box.getSize(new THREE.Vector3());
    const extent = Math.max(1.2, Math.max(size.x, size.y, size.z) * 1.6);
    floor.scale.set(extent, extent, 1);
    floor.position.y = box.min.y - Math.max(size.y * 0.02, 0.002);
    floor.visible = true;

    if (stage.keyLight?.shadow?.camera) {
      const shadowExtent = Math.max(2.0, extent);
      stage.keyLight.shadow.camera.left = -shadowExtent;
      stage.keyLight.shadow.camera.right = shadowExtent;
      stage.keyLight.shadow.camera.top = shadowExtent;
      stage.keyLight.shadow.camera.bottom = -shadowExtent;
      stage.keyLight.shadow.camera.far = Math.max(20, shadowExtent * 4);
      stage.keyLight.shadow.needsUpdate = true;
    }
  }

  /**
   * Fit camera to bounding box for a specific stage.
   */
  _fitCamera(stage, box) {
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z, 1e-3);
    const dist = maxDim * 1.5;
    stage.camera.position.set(dist * 0.5, dist * 0.5, dist);
    stage.controls.target.set(0, 0, 0);
    stage.controls.update();
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
        headerEl.style.display = '';
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
