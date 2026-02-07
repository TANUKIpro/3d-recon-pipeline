/**
 * Camera frustum overlay for Pi3X 3D scene.
 *
 * Renders wireframe frustum pyramids at camera positions,
 * colored by index (blue→red gradient via HSL).
 */

export class CameraOverlay {
  constructor() {
    this._group = null;
    this._parent = null;
    this._THREE = null;
  }

  /**
   * Create camera frustum markers from pose matrices.
   * @param {object} THREE - three.js module
   * @param {THREE.Object3D} parent - target parent (scene or group)
   * @param {Array} poses - array of {matrix: number[16], frame_index: number}
   */
  create(THREE, parent, poses) {
    this._THREE = THREE;
    this._parent = parent;
    this.remove();

    this._group = new THREE.Group();
    this._group.name = 'camera-frustums';

    const count = poses.length;

    for (let i = 0; i < count; i++) {
      const pose = poses[i];
      const mat4 = new THREE.Matrix4();

      // Pi3X outputs camera-to-world (c2w) — use directly
      mat4.fromArray(pose.matrix);
      const camToWorld = mat4;

      // Extract camera position
      const position = new THREE.Vector3();
      const quaternion = new THREE.Quaternion();
      const scale = new THREE.Vector3();
      camToWorld.decompose(position, quaternion, scale);

      // Create frustum wireframe
      const frustum = this._createFrustum(THREE, i, count);
      frustum.position.copy(position);
      frustum.quaternion.copy(quaternion);

      this._group.add(frustum);
    }

    parent.add(this._group);
  }

  /**
   * Remove all frustum objects from parent.
   */
  remove() {
    if (this._group && this._parent) {
      this._parent.remove(this._group);
      this._group.traverse((obj) => {
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) obj.material.dispose();
      });
      this._group = null;
    }
  }

  /**
   * Shift the camera group by the same centering offset applied to the point cloud.
   * @param {THREE.Vector3} center - the bounding-box center used to translate the geometry
   */
  applyOffset(center) {
    if (this._group) {
      this._group.position.set(-center.x, -center.y, -center.z);
    }
  }

  /**
   * Toggle visibility of the camera frustum group.
   */
  setVisible(visible) {
    if (this._group) {
      this._group.visible = visible;
    }
  }

  /**
   * Create a single wireframe frustum pyramid.
   * Frustum: apex at origin, base extends in -Z direction (camera looks down -Z).
   */
  _createFrustum(THREE, index, total) {
    const size = 0.03;
    const aspect = 1.5;
    const depth = size * 2;
    const halfW = size * aspect * 0.5;
    const halfH = size * 0.5;

    // 5 vertices: apex + 4 base corners
    const vertices = new Float32Array([
      // Apex (camera position / local origin)
      0, 0, 0,
      // Base corners (in -Z direction = camera forward)
      -halfW, -halfH, -depth,
       halfW, -halfH, -depth,
       halfW,  halfH, -depth,
      -halfW,  halfH, -depth,
    ]);

    // 8 line segments: 4 from apex to corners, 4 around base
    const indices = [
      0, 1, 0, 2, 0, 3, 0, 4, // apex → corners
      1, 2, 2, 3, 3, 4, 4, 1, // base rectangle
    ];

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    geometry.setIndex(indices);

    // Color: blue→red gradient via HSL
    const hue = total > 1 ? (240 - (index / (total - 1)) * 240) / 360 : 0.66;
    const color = new THREE.Color().setHSL(hue, 0.9, 0.6);

    const material = new THREE.LineBasicMaterial({ color, linewidth: 1 });
    return new THREE.LineSegments(geometry, material);
  }
}
