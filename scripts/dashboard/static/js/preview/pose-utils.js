/**
 * Camera pose utilities: convention detection, scene flip inference.
 *
 * All exported functions are mixin methods — they use `this` to refer to
 * the PreviewPanel instance and to call sibling methods.
 * THREE must be set via initThree() before use.
 */

let THREE;

export function initThree(threeModule) {
  THREE = threeModule;
}

export function _poseJsonToArray(data) {
  // Two on-disk shapes are accepted (mirrors scripts/texture/io_utils.py:_load_poses):
  //   1) {poses: [[[..]x4]x4, ...], frame_indices: [...]}  — legacy Pi3 / gs2mesh
  //   2) [{frame_name, transform_matrix: [[..]x4]x4}, ...] — COLMAP stage output
  let matrices = [];
  let frameIndices = [];
  if (Array.isArray(data?.poses)) {
    matrices = data.poses;
    frameIndices = Array.isArray(data?.frame_indices) ? data.frame_indices : [];
  } else if (Array.isArray(data)) {
    for (let i = 0; i < data.length; i++) {
      const item = data[i];
      if (!item || !Array.isArray(item.transform_matrix)) continue;
      matrices.push(item.transform_matrix);
      const explicit = item.frame_index;
      if (Number.isFinite(explicit)) {
        frameIndices.push(Number(explicit));
      } else {
        const stem = String(item.frame_name || '').replace(/\.[^./]+$/, '');
        const m = stem.match(/(\d+)$/);
        frameIndices.push(m ? parseInt(m[1], 10) : i);
      }
    }
  }

  return matrices.map((mat4x4, i) => {
    // Transpose: row-major (Python) → column-major (three.js Matrix4.fromArray)
    const flat = new Array(16);
    for (let r = 0; r < 4; r++)
      for (let c = 0; c < 4; c++)
        flat[c * 4 + r] = mat4x4[r][c];
    return { matrix: flat, frame_index: frameIndices[i] ?? i };
  });
}

export function _applySceneFlipToLoadedStages(sceneFlipX) {
  const rotationX = sceneFlipX ? Math.PI : 0;
  for (const stageNum of [2, 4, 5, 6]) {
    const stage = this._stages[stageNum];
    if (stage?.sceneRoot) stage.sceneRoot.rotation.x = rotationX;
  }
}

export async function _ensureSceneFlipForStage(stageNum, cacheToken = null) {
  if (stageNum < 4 || stageNum > 6) return;
  if (this._sceneFlipX == null) {
    this._sceneFlipX = await this._resolveSceneFlipFromCameraPoses(cacheToken);
  }
  const stage = this._stages[stageNum];
  if (stage?.sceneRoot) {
    stage.sceneRoot.rotation.x = this._sceneFlipX ? Math.PI : 0;
  }
}

export async function _resolveSceneFlipFromCameraPoses(cacheToken = null) {
  try {
    const res = await fetch(this._buildPreviewFileUrl('p2_colmap/camera_poses.json', cacheToken));
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

export function _forwardSignFromAlignment(alignment) {
  if (!alignment || typeof alignment !== 'object') return null;
  const axis = String(alignment.inferred_forward_axis || '').toLowerCase();
  if (axis.includes('-z') || axis.includes('-col 2')) return -1;
  if (axis.includes('+z') || axis.includes('+col 2') || axis.includes('col 2')) return 1;
  return null;
}

export function _normalizePoseConvention(poseArray) {
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

export function _shouldApplySceneFlipX(poseArray, alignment) {
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

export function _meanCameraDownY(poseArray) {
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

export function _invertPoseArray(poseArray) {
  const mat = new THREE.Matrix4();
  const inv = new THREE.Matrix4();
  return poseArray.map((pose) => {
    mat.fromArray(pose.matrix);
    inv.copy(mat).invert();
    return { matrix: inv.toArray(), frame_index: pose.frame_index };
  });
}

export function _evaluateForwardFromOrbitCenter(poseArray) {
  const center = this._computePoseCentroid(poseArray);
  if (!center) {
    return { forwardSign: 1, confidence: 0, used: 0 };
  }
  return this._scoreForwardTowardTarget(poseArray, center);
}

export function _computePoseCentroid(poseArray) {
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

export function _scoreForwardTowardTarget(poseArray, targetCenter) {
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

export function _inferForwardSignFromTarget(poseArray, targetCenter) {
  return this._scoreForwardTowardTarget(poseArray, targetCenter).forwardSign;
}
