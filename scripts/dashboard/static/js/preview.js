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

export class PreviewPanel {
  constructor() {
    this._galleryGrid = document.getElementById('gallery-grid');

    // Per-stage scene state: { scene, camera, controls, container, currentObject, initialized }
    this._stages = {};
    this._renderer = null;
    this._threeLoaded = false;
    this._activeStage = null;
    this._animating = false;
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
    scene.add(dirLight);

    // Grid
    const grid = new THREE.GridHelper(4, 20, 0x333355, 0x222244);
    scene.add(grid);

    // Stage-specific axis conversion is configured when data is loaded.
    const sceneRoot = new THREE.Group();
    sceneRoot.rotation.x = Math.PI;
    scene.add(sceneRoot);

    // Show container
    container.classList.add('visible');

    this._stages[stageNum] = {
      scene, sceneRoot, camera, controls, container,
      currentObject: null,
      initialized: true,
    };

    // Resize observer
    const ro = new ResizeObserver(() => {
      const nw = container.clientWidth;
      const nh = container.clientHeight;
      if (nw > 0 && nh > 0) {
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

    // Resize
    const w = stage.container.clientWidth;
    const h = stage.container.clientHeight;
    if (w > 0 && h > 0) {
      this._renderer.setSize(w, h);
      stage.camera.aspect = w / h;
      stage.camera.updateProjectionMatrix();
    }
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

  /**
   * Load Pi3X results: point cloud + camera poses.
   * @param {CameraOverlay} cameraOverlay - camera overlay instance
   * @param {string} [plyFile='object_full.ply'] - PLY filename to load
   */
  async loadPi3xResults(cameraOverlay, plyFile = 'object_full.ply') {
    await this.initSceneForStage(2);
    this.activateStage(2);

    // Hide empty placeholder
    const empty = document.getElementById('stage-2-empty');
    if (empty) empty.classList.add('hidden');

    // Show toolbar
    const toolbar = document.getElementById('pi3x-toolbar');
    if (toolbar) toolbar.style.display = 'flex';

    const stage = this._stages[2];
    if (!stage) return;

    // Load point cloud
    await this._loadPLYIntoStage(2, plyFile);

    // Update point count
    if (stage.currentObject?.geometry) {
      const count = stage.currentObject.geometry.attributes.position?.count || 0;
      const el = document.getElementById('pi3x-point-count');
      if (el) el.textContent = `${count.toLocaleString()} points`;
    }

    // Load camera poses
    try {
      const res = await fetch('/api/preview/file/camera_poses.json');
      if (res.ok) {
        const data = await res.json();

        // Convert {poses: [[4x4], ...], frame_indices: [...]} → [{matrix: flat16, frame_index}]
        const rawPoseArray = (data.poses || []).map((mat4x4, i) => {
          // Transpose: row-major (Python) → column-major (three.js Matrix4.fromArray)
          const flat = new Array(16);
          for (let r = 0; r < 4; r++)
            for (let c = 0; c < 4; c++)
              flat[c * 4 + r] = mat4x4[r][c];
          return { matrix: flat, frame_index: (data.frame_indices || [])[i] ?? i };
        });

        const normalized = this._normalizePoseConvention(rawPoseArray);
        const poseArray = normalized.poseArray;
        const sceneFlipX = this._shouldApplySceneFlipX(poseArray, data.alignment);
        stage.sceneRoot.rotation.x = sceneFlipX ? Math.PI : 0;

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
      }
    } catch (e) {
      console.error('Failed to load camera poses:', e);
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

  /**
   * Auto-load the appropriate result file for a stage.
   */
  async loadStageResult(stageNum) {
    const fileMap = {
      4: 'object_denoised.ply',
      5: 'object_mesh.ply',
      6: 'textured_mesh.obj',
    };

    const file = fileMap[stageNum];
    if (!file) return;

    await this.initSceneForStage(stageNum);
    this.activateStage(stageNum);

    // Hide empty placeholder
    const empty = document.getElementById(`stage-${stageNum}-empty`);
    if (empty) empty.classList.add('hidden');

    const ext = file.split('.').pop().toLowerCase();
    if (ext === 'ply') {
      await this._loadPLYIntoStage(stageNum, file);
    } else if (ext === 'obj') {
      await this._loadOBJIntoStage(stageNum, file);
    }
  }

  /**
   * Load a PLY file into a specific stage's scene.
   */
  async _loadPLYIntoStage(stageNum, relativePath) {
    const stage = this._stages[stageNum];
    if (!stage) return;

    // Remove previous object
    if (stage.currentObject) {
      stage.sceneRoot.remove(stage.currentObject);
      stage.currentObject = null;
    }

    const url = `/api/preview/file/${relativePath}`;
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

      let obj;
      if (geometry.index && geometry.index.count > 0) {
        // Mesh
        const meshMat = new THREE.MeshStandardMaterial({
          vertexColors: geometry.hasAttribute('color'),
          side: THREE.DoubleSide,
        });
        obj = new THREE.Mesh(geometry, meshMat);
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
      this._fitCamera(stage, geometry.boundingBox);
    } catch (e) {
      console.error(`Failed to load PLY (stage ${stageNum}):`, e);
    }
  }

  /**
   * Load an OBJ file into a specific stage's scene.
   */
  async _loadOBJIntoStage(stageNum, relativePath) {
    const stage = this._stages[stageNum];
    if (!stage) return;

    if (stage.currentObject) {
      stage.sceneRoot.remove(stage.currentObject);
      stage.currentObject = null;
    }

    const url = `/api/preview/file/${relativePath}`;
    const mtlPath = relativePath.replace(/\.obj$/i, '.mtl');
    let materials = null;

    try {
      const mtlLoader = new MTLLoader();
      mtlLoader.setPath('/api/preview/file/');
      materials = await new Promise((resolve, reject) => {
        mtlLoader.load(mtlPath, resolve, undefined, reject);
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
      this._fitCamera(stage, box);
    } catch (e) {
      console.error(`Failed to load OBJ (stage ${stageNum}):`, e);
    }
  }

  /**
   * Fit camera to bounding box for a specific stage.
   */
  _fitCamera(stage, box) {
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
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
      const res = await fetch('/api/preview/outputs');
      const data = await res.json();
      this._galleryGrid.innerHTML = '';

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
        img.src = `/api/preview/file/${f.path}`;
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
