/**
 * 3D preview manager — per-stage scene support for PLY/OBJ.
 *
 * Uses a single shared renderer that moves between stage containers.
 * Stage 2 has a dedicated scene for camera overlay support.
 */

let THREE;
let OrbitControls;
let PLYLoader;
let OBJLoader;
let MTLLoader;

const FIRST_MESH_PREVIEW_FILES = new Set([
  'object_mesh_raw.ply',
  'object_mesh_poisson_raw.ply',
  'object_mesh_wrapped.ply',
  'object_mesh_repaired.ply',
]);

export class PreviewPanel {
  constructor() {
    this._galleryGrid = document.getElementById('gallery-grid');

    // Per-stage scene state: { scene, camera, controls, container, currentObject, initialized }
    this._stages = {};
    this._renderer = null;
    this._threeLoaded = false;
    this._activeStage = null;
    this._animating = false;
    this._sceneFlipX = null;
    this._previewAssetRevision = 0;
    this.onMeshRepairSelectionChanged = null;
    this._meshRepair = {
      active: false,
      confirmed: false,
      group: null,
      loopObjects: new Map(),
      selected: new Set(),
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

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a14);

    const camera = new THREE.PerspectiveCamera(60, w / h, 0.01, 100);
    camera.position.set(0, 0, 2);

    const controls = new OrbitControls(camera, this._renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.1;

    // Lights
    const ambient = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambient);
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
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
    const grid = new THREE.GridHelper(4, 20, 0x333355, 0x222244);
    scene.add(grid);

    // Stage 2 starts with OpenCV->OpenGL flip. Stages 4-8 use inferred flip when available.
    const sceneRoot = new THREE.Group();
    sceneRoot.rotation.x = this._defaultSceneFlipX(stageNum) ? Math.PI : 0;
    scene.add(sceneRoot);

    // Hidden by default. Enabled only when mesh-shadow profile is active.
    const shadowFloor = new THREE.Mesh(
      new THREE.PlaneGeometry(1, 1),
      new THREE.ShadowMaterial({ color: 0x000000, opacity: 0.28 }),
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
    if (stageNum === 7) {
      this._clearMeshRepairOverlay();
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
    const objects = Array.from(this._meshRepair.loopObjects.values());
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
    this._updateMeshRepairColors();
    this.onMeshRepairSelectionChanged?.(this.getMeshRepairSelectedLoopIds());
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
    this._meshRepair.selected = new Set();
    this._meshRepair.active = false;
    this._meshRepair.confirmed = false;
    this.onMeshRepairSelectionChanged?.([]);
  }

  clearMeshRepairSelection() {
    if (!this._meshRepair.active) return;
    this._meshRepair.confirmed = false;
    this._meshRepair.selected.clear();
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
    this._updateMeshRepairColors();
    this._setMeshRepairClickEnabled(true);
    this.onMeshRepairSelectionChanged?.(this.getMeshRepairSelectedLoopIds());
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

    // Load point cloud
    const loaded = await this._loadPLYIntoStage(2, plyFile);
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
      const res = await fetch('/api/preview/file/camera_poses.json');
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

  async _ensureSceneFlipForStage(stageNum) {
    if (stageNum < 4 || stageNum > 8) return;
    if (this._sceneFlipX == null) {
      this._sceneFlipX = await this._resolveSceneFlipFromCameraPoses();
    }
    const stage = this._stages[stageNum];
    if (stage?.sceneRoot) {
      stage.sceneRoot.rotation.x = this._sceneFlipX ? Math.PI : 0;
    }
  }

  async _resolveSceneFlipFromCameraPoses() {
    try {
      const res = await fetch('/api/preview/file/camera_poses.json');
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
    const file = overrideFile || fileMap[stageNum];
    if (!file) return false;
    const fileName = file.split('/').pop()?.toLowerCase() || '';
    const isFirstMeshPreview = FIRST_MESH_PREVIEW_FILES.has(fileName);

    await this.initSceneForStage(stageNum);
    await this._ensureSceneFlipForStage(stageNum);
    this.activateStage(stageNum);

    // Toggle empty placeholder based on load result.
    const empty = document.getElementById(`stage-${stageNum}-empty`);

    const ext = file.split('.').pop().toLowerCase();
    const cacheToken = opts.cacheToken ?? this._nextPreviewRevision();
    const renderMode = String(opts.renderMode || (isFirstMeshPreview ? 'mesh' : '')).toLowerCase();
    const stripVertexColors = opts.stripVertexColors ?? isFirstMeshPreview;
    const enableShadows = opts.enableShadows ?? isFirstMeshPreview;
    let loaded = false;
    if (ext === 'ply') {
      loaded = await this._loadPLYIntoStage(stageNum, file, {
        cacheToken,
        renderMode,
        stripVertexColors: stripVertexColors === true,
        enableShadows: enableShadows === true,
      });
    } else if (ext === 'obj') {
      loaded = await this._loadOBJIntoStage(stageNum, file, { cacheToken });
    } else {
      loaded = false;
    }
    if (empty) empty.classList.toggle('hidden', loaded);
    return loaded;
  }

  /**
   * Load a PLY file into a specific stage's scene.
   */
  async _loadPLYIntoStage(stageNum, relativePath, opts = {}) {
    const stage = this._stages[stageNum];
    if (!stage) return false;
    if (stageNum === 7) {
      this._clearMeshRepairOverlay();
    }

    // Remove previous object
    if (stage.currentObject) {
      this._disposeObject(stage.currentObject);
      stage.sceneRoot.remove(stage.currentObject);
      stage.currentObject = null;
    }

    const url = this._buildPreviewFileUrl(relativePath, opts.cacheToken);
    const loader = new PLYLoader();

    try {
      const geometry = await new Promise((resolve, reject) => {
        loader.load(url, resolve, undefined, reject);
      });

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
          color: hasColor ? undefined : 0xb3b3b3,
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
            color: 0x101820,
            transparent: true,
            opacity: 0.35,
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
          materialParams.color = 0x4a9eff;
        }
        const material = new THREE.PointsMaterial(materialParams);
        obj = new THREE.Points(geometry, material);
      }

      stage.currentObject = obj;
      stage.sceneRoot.add(obj);
      // clearFromStage() can hide initialized containers; show it again on successful load.
      stage.container?.classList.add('visible');
      this._fitCamera(stage, geometry.boundingBox);
      return true;
    } catch (e) {
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

    if (stage.currentObject) {
      this._disposeObject(stage.currentObject);
      stage.sceneRoot.remove(stage.currentObject);
      stage.currentObject = null;
    }

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
      materials.preload();
    } catch (e) {
      // No MTL file — okay
    }

    try {
      const objLoader = new OBJLoader();
      if (materials) objLoader.setMaterials(materials);

      const object = await new Promise((resolve, reject) => {
        objLoader.load(url, resolve, undefined, reject);
      });

      const box = new THREE.Box3().setFromObject(object);
      const center = box.getCenter(new THREE.Vector3());
      object.position.sub(center);

      stage.currentObject = object;
      stage.sceneRoot.add(object);
      // clearFromStage() can hide initialized containers; show it again on successful load.
      stage.container?.classList.add('visible');
      this._fitCamera(stage, box);
      return true;
    } catch (e) {
      console.error(`Failed to load OBJ (stage ${stageNum}):`, e);
      return false;
    }
  }

  /**
   * Configure optional shadow-focused lighting profile for mesh previews.
   */
  _setMeshShadowProfile(stage, box, enabled) {
    if (!stage) return;
    if (stage.ambientLight) {
      stage.ambientLight.intensity = enabled ? 0.36 : 0.6;
    }
    if (stage.keyLight) {
      stage.keyLight.intensity = enabled ? 1.05 : 0.8;
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
