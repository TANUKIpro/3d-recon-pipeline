/**
 * Mesh repair overlay: interactive loop selection and visibility filtering.
 *
 * All exported functions are mixin methods — they use `this` to refer to
 * the PreviewPanel instance.
 * THREE must be set via initThree() before use.
 */

import { clampMeshRepairThreshold } from '../utils.js';

let THREE;

export function initThree(threeModule) {
  THREE = threeModule;
}

export function _setMeshRepairClickEnabled(enabled) {
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

export function _handleMeshRepairClick(event) {
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

export function _isMeshRepairLoopVisible(loopId) {
  if (this._meshRepair.selected.has(loopId)) return true;
  const meta = this._meshRepair.loopMeta.get(loopId);
  const normalY = Number(meta?.normalY);
  if (!Number.isFinite(normalY)) return true;
  return normalY <= this._meshRepair.threshold;
}

export function _applyMeshRepairVisibility() {
  let visible = 0;
  for (const [loopId, obj] of this._meshRepair.loopObjects.entries()) {
    if (!obj) continue;
    const isVisible = this._isMeshRepairLoopVisible(loopId);
    obj.visible = isVisible;
    if (isVisible) visible += 1;
  }
  return visible;
}

export function _updateMeshRepairColors() {
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

export function _clearMeshRepairOverlay() {
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

export function clearMeshRepairSelection() {
  if (!this._meshRepair.active) return;
  this._meshRepair.confirmed = false;
  this._meshRepair.selected.clear();
  this._applyMeshRepairVisibility();
  this._updateMeshRepairColors();
  this.onMeshRepairSelectionChanged?.(this.getMeshRepairSelectedLoopIds());
}

export function setMeshRepairConfirmed() {
  if (!this._meshRepair.active) return;
  this._meshRepair.confirmed = true;
  this._updateMeshRepairColors();
  this._setMeshRepairClickEnabled(false);
  this.onMeshRepairSelectionChanged?.(this.getMeshRepairSelectedLoopIds());
}

export function getMeshRepairSelectedLoopIds() {
  return Array.from(this._meshRepair.selected).sort((a, b) => a - b);
}

export function getMeshRepairVisibleLoopCount() {
  let visible = 0;
  for (const obj of this._meshRepair.loopObjects.values()) {
    if (obj?.visible !== false) visible += 1;
  }
  return visible;
}

export function getMeshRepairTotalLoopCount() {
  return this._meshRepair.loopObjects.size;
}

export function getMeshRepairThreshold() {
  return this._meshRepair.threshold;
}

export function setMeshRepairThreshold(value) {
  this._meshRepair.threshold = clampMeshRepairThreshold(value);
  if (!this._meshRepair.active) return;
  this._applyMeshRepairVisibility();
  this._updateMeshRepairColors();
  this.onMeshRepairSelectionChanged?.(this.getMeshRepairSelectedLoopIds());
}

export async function beginMeshRepairSelection(payload = {}) {
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
