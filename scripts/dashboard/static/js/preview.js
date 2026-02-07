/**
 * 3D preview (PLY point clouds, OBJ meshes) via three.js + frame gallery.
 */

let THREE;
let OrbitControls;
let PLYLoader;
let OBJLoader;
let MTLLoader;

export class PreviewPanel {
  constructor() {
    this._container = document.getElementById('three-container');
    this._fileSelect = document.getElementById('preview-file-select');
    this._loadBtn = document.getElementById('preview-load');
    this._galleryGrid = document.getElementById('gallery-grid');

    this._scene = null;
    this._camera = null;
    this._renderer = null;
    this._controls = null;
    this._currentObject = null;
    this._initialized = false;

    this._bindEvents();
  }

  async init3D() {
    if (this._initialized) return;
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

      this._setup3DScene();
      this._initialized = true;
    } catch (e) {
      console.error('Failed to load three.js:', e);
      this._container.innerHTML = '<p style="padding:20px;color:#888">3D preview unavailable (three.js load failed)</p>';
    }
  }

  _setup3DScene() {
    const w = this._container.clientWidth || 640;
    const h = this._container.clientHeight || 480;

    this._scene = new THREE.Scene();
    this._scene.background = new THREE.Color(0x0a0a14);

    this._camera = new THREE.PerspectiveCamera(60, w / h, 0.01, 100);
    this._camera.position.set(0, 0, 2);

    this._renderer = new THREE.WebGLRenderer({ antialias: true });
    this._renderer.setSize(w, h);
    this._renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this._container.appendChild(this._renderer.domElement);

    this._controls = new OrbitControls(this._camera, this._renderer.domElement);
    this._controls.enableDamping = true;
    this._controls.dampingFactor = 0.1;

    // Lights
    const ambient = new THREE.AmbientLight(0xffffff, 0.6);
    this._scene.add(ambient);
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(2, 3, 4);
    this._scene.add(dirLight);

    // Grid
    const grid = new THREE.GridHelper(4, 20, 0x333355, 0x222244);
    this._scene.add(grid);

    // Resize observer
    const ro = new ResizeObserver(() => {
      const nw = this._container.clientWidth;
      const nh = this._container.clientHeight;
      if (nw > 0 && nh > 0) {
        this._camera.aspect = nw / nh;
        this._camera.updateProjectionMatrix();
        this._renderer.setSize(nw, nh);
      }
    });
    ro.observe(this._container);

    // Animate
    const animate = () => {
      requestAnimationFrame(animate);
      this._controls.update();
      this._renderer.render(this._scene, this._camera);
    };
    animate();
  }

  async refreshFileList() {
    try {
      const res = await fetch('/api/preview/outputs');
      const data = await res.json();
      this._fileSelect.innerHTML = '<option value="">Select file to preview...</option>';

      for (const f of data.files) {
        if (['.ply', '.obj'].includes(f.ext)) {
          const opt = document.createElement('option');
          opt.value = f.path;
          opt.textContent = `${f.name} (${f.size_mb} MB)`;
          this._fileSelect.appendChild(opt);
        }
      }
    } catch (e) {
      console.error('Failed to load output files:', e);
    }
  }

  async loadFile(relativePath) {
    if (!this._initialized) await this.init3D();
    if (!relativePath) return;

    // Remove previous object
    if (this._currentObject) {
      this._scene.remove(this._currentObject);
      this._currentObject = null;
    }

    const url = `/api/preview/file/${relativePath}`;
    const ext = relativePath.split('.').pop().toLowerCase();

    try {
      if (ext === 'ply') {
        await this._loadPLY(url);
      } else if (ext === 'obj') {
        await this._loadOBJ(url, relativePath);
      }
    } catch (e) {
      console.error('Failed to load 3D file:', e);
    }
  }

  async _loadPLY(url) {
    const loader = new PLYLoader();
    const geometry = await new Promise((resolve, reject) => {
      loader.load(url, resolve, undefined, reject);
    });

    geometry.computeBoundingBox();
    const center = new THREE.Vector3();
    geometry.boundingBox.getCenter(center);
    geometry.translate(-center.x, -center.y, -center.z);

    let material;
    if (geometry.hasAttribute('color')) {
      material = new THREE.PointsMaterial({
        size: 0.005,
        vertexColors: true,
        sizeAttenuation: true,
      });
    } else {
      material = new THREE.PointsMaterial({
        size: 0.005,
        color: 0x4a9eff,
        sizeAttenuation: true,
      });
    }

    // Detect if mesh (has faces) or point cloud
    if (geometry.index && geometry.index.count > 0) {
      const meshMat = new THREE.MeshStandardMaterial({
        vertexColors: geometry.hasAttribute('color'),
        side: THREE.DoubleSide,
      });
      this._currentObject = new THREE.Mesh(geometry, meshMat);
    } else {
      this._currentObject = new THREE.Points(geometry, material);
    }

    this._scene.add(this._currentObject);
    this._fitCamera(geometry.boundingBox);
  }

  async _loadOBJ(url, relativePath) {
    // Try loading MTL if available
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

    const objLoader = new OBJLoader();
    if (materials) objLoader.setMaterials(materials);

    const object = await new Promise((resolve, reject) => {
      objLoader.load(url, resolve, undefined, reject);
    });

    // Center the object
    const box = new THREE.Box3().setFromObject(object);
    const center = box.getCenter(new THREE.Vector3());
    object.position.sub(center);

    this._currentObject = object;
    this._scene.add(this._currentObject);
    this._fitCamera(box);
  }

  _fitCamera(box) {
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    const dist = maxDim * 1.5;
    this._camera.position.set(dist * 0.5, dist * 0.5, dist);
    this._controls.target.set(0, 0, 0);
    this._controls.update();
  }

  async loadGallery() {
    try {
      const res = await fetch('/api/preview/outputs');
      const data = await res.json();
      this._galleryGrid.innerHTML = '';

      const images = data.files.filter(f => ['.png', '.jpg'].includes(f.ext));
      // Sort: frames first, then masks, then others
      images.sort((a, b) => a.path.localeCompare(b.path));

      for (const f of images) {
        const img = document.createElement('img');
        img.className = 'gallery-thumb';
        img.src = `/api/preview/file/${f.path}`;
        img.alt = f.name;
        img.title = f.path;
        img.loading = 'lazy';
        this._galleryGrid.appendChild(img);
      }

      if (images.length === 0) {
        this._galleryGrid.innerHTML = '<p style="padding:20px;color:#555">No images yet.</p>';
      }
    } catch (e) {
      console.error('Failed to load gallery:', e);
    }
  }

  _bindEvents() {
    this._loadBtn.addEventListener('click', () => {
      const val = this._fileSelect.value;
      if (val) this.loadFile(val);
    });
  }
}
