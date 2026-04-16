/**
 * Scene utility helpers: disposal, camera fitting, shadow profiles,
 * ground plane rendering, and stage center offset.
 *
 * All exported functions are mixin methods — they use `this` to refer to
 * the PreviewPanel instance.
 * THREE and SCENE_THEMES must be set via initThree() before use.
 */

import { SCENE_THEMES } from './constants.js';

let THREE;

export function initThree(threeModule) {
  THREE = threeModule;
}

// ── Disposal & cleanup ────────────────────────────────────────────

const _TEX_PROPS = ['map', 'normalMap', 'bumpMap', 'specularMap', 'emissiveMap', 'alphaMap'];

function _disposeMaterial(mat) {
  if (!mat) return;
  for (const prop of _TEX_PROPS) {
    if (mat[prop]?.dispose) mat[prop].dispose();
  }
  if (typeof mat.dispose === 'function') mat.dispose();
}

export function _disposeObject(object3d) {
  if (!object3d) return;
  object3d.traverse((node) => {
    if (node.geometry && typeof node.geometry.dispose === 'function') {
      node.geometry.dispose();
    }
    if (!node.material) return;
    if (Array.isArray(node.material)) {
      for (const material of node.material) _disposeMaterial(material);
      return;
    }
    _disposeMaterial(node.material);
  });
}

export function _cleanupCurrentObject(stage) {
  if (stage.currentObject) {
    this._disposeObject(stage.currentObject);
    stage.sceneRoot.remove(stage.currentObject);
    stage.currentObject = null;
  }
  this._cleanupOverlayObject(stage);
}

export function _cleanupOverlayObject(stage) {
  if (stage.overlayObject) {
    this._disposeObject(stage.overlayObject);
    stage.sceneRoot.remove(stage.overlayObject);
    stage.overlayObject = null;
  }
}

// ── Camera & shadow ───────────────────────────────────────────────

export function _fitCamera(stage, box) {
  const size = box.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z, 1e-3);
  const dist = maxDim * 1.5;
  stage.camera.position.set(dist * 0.5, dist * 0.5, dist);
  stage.controls.target.set(0, 0, 0);
  stage.controls.update();
}

/**
 * Frame the camera so the always-visible "stage" (the flat XZ grid around
 * origin) fills the view with a small horizontal margin. This ignores the
 * full point-cloud bbox (which may include floaters) and locks the zoom to
 * the stage footprint instead.
 */
export function _fitCameraToStage(stage, gridHalfExtent = 2.0, marginFactor = 1.1) {
  const camera = stage.camera;
  const fovRad = (camera.fov * Math.PI) / 180;
  const aspect = Math.max(camera.aspect || 1, 1e-3);
  const tanV = Math.tan(fovRad / 2);
  const tanH = tanV * aspect;

  const dir = new THREE.Vector3(0.5, 0.5, 1).normalize();

  // Provisional placement so lookAt() gives us usable world axes.
  const provisionalD = 10;
  camera.position.set(dir.x * provisionalD, dir.y * provisionalD, dir.z * provisionalD);
  camera.lookAt(0, 0, 0);
  camera.updateMatrixWorld(true);

  const right = new THREE.Vector3(1, 0, 0).transformDirection(camera.matrixWorld);
  const up = new THREE.Vector3(0, 1, 0).transformDirection(camera.matrixWorld);

  const h = gridHalfExtent;
  const corners = [
    new THREE.Vector3( h, 0,  h),
    new THREE.Vector3( h, 0, -h),
    new THREE.Vector3(-h, 0,  h),
    new THREE.Vector3(-h, 0, -h),
  ];
  let maxRight = 0;
  let maxUp = 0;
  for (const p of corners) {
    const r = Math.abs(p.dot(right));
    const u = Math.abs(p.dot(up));
    if (r > maxRight) maxRight = r;
    if (u > maxUp) maxUp = u;
  }

  const Dh = (maxRight * marginFactor) / tanH;
  const Dv = (maxUp * marginFactor) / tanV;
  const D = Math.max(Dh, Dv, 1e-3);

  camera.position.set(dir.x * D, dir.y * D, dir.z * D);
  stage.controls.target.set(0, 0, 0);
  stage.controls.update();
}

export function _setMeshShadowProfile(stage, box, enabled) {
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

// ── Stage center offset ───────────────────────────────────────────

export function _applyStageCenterOffset(stage, point3) {
  if (!Array.isArray(point3) || point3.length < 3) return null;
  const x = Number(point3[0]);
  const y = Number(point3[1]);
  const z = Number(point3[2]);
  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return null;

  const offset = stage?.centerOffset;
  if (!offset) return [x, y, z];
  return [x - offset.x, y - offset.y, z - offset.z];
}

// ── Ground plane ──────────────────────────────────────────────────

export async function showGroundPlane(stageNum) {
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

export function _createGroundPlaneMesh(stageNum) {
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

export function _removeGroundPlane(stageNum) {
  const mesh = this._groundPlane.meshByStage[stageNum];
  if (!mesh) return;
  mesh.parent?.remove(mesh);
  mesh.geometry?.dispose();
  mesh.material?.dispose();
  delete this._groundPlane.meshByStage[stageNum];
}

export function toggleGroundPlane(visible) {
  this._groundPlane.visible = visible;
  for (const mesh of Object.values(this._groundPlane.meshByStage)) {
    if (mesh) mesh.visible = visible;
  }
}
