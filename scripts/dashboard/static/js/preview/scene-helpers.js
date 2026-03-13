/**
 * Scene utility helpers: disposal, camera fitting, shadow profiles,
 * crop bounding box, ground plane, and section loop rendering.
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

export function _disposeObject(object3d) {
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

export function _cleanupCurrentObject(stage) {
  if (stage.currentObject) {
    this._disposeObject(stage.currentObject);
    stage.sceneRoot.remove(stage.currentObject);
    stage.currentObject = null;
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

// ── Classical phase descriptor ────────────────────────────────────

export function _classicalPhaseDescriptor(step) {
  const key = String(step || '').toLowerCase();
  const map = {
    preprocess: { file: 'object_mesh_input.ply', renderMode: 'points' },
    main: { file: 'object_mesh_raw.ply', renderMode: 'mesh' },
    postprocess: { file: 'object_mesh_postprocessed.ply', renderMode: 'mesh' },
    downsample: { file: 'object_mesh.ply', renderMode: 'mesh' },
  };
  return map[key] || null;
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

// ── Crop bounding box ─────────────────────────────────────────────

export async function showCropBbox(cropScale, opts = {}) {
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

export function _createCropBboxHelper(cropScale) {
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

export function updateCropBbox(cropScale) {
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

export function clearCropBbox() {
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

// ── Section loop ──────────────────────────────────────────────────

export function _clearSectionLoop() {
  const grp = this._sectionLoop.group;
  if (!grp) return;
  grp.traverse((node) => {
    if (node?.geometry?.dispose) node.geometry.dispose();
    if (node?.material?.dispose) node.material.dispose();
  });
  grp.parent?.remove(grp);
  this._sectionLoop.group = null;
}

export async function _showSectionLoop() {
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
