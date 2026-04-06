/**
 * Configuration preset constants for the gs2mesh pipeline.
 */

export const NEW_OBJECT_VALUE = '__new__';
export const GS2MESH_PRESET_DEFAULT = 'default';
export const GS2MESH_PRESET_HIGH = 'high';
export const GS2MESH_PRESET_CUSTOM = 'custom';
export const GS2MESH_PUBLIC_PRESETS = {
  [GS2MESH_PRESET_DEFAULT]: {
    gs2mesh_gs_iterations: 5000,
    gs2mesh_runtime_profile: 'auto',
    gs2mesh_stereo_model: 'DLNR',
    gs2mesh_tsdf_voxel_size: 0.005,
    gs2mesh_tsdf_depth_trunc: 0.04,
    gs2mesh_use_masks: true,
  },
  [GS2MESH_PRESET_HIGH]: {
    gs2mesh_gs_iterations: 15000,
    gs2mesh_runtime_profile: 'auto',
    gs2mesh_stereo_model: 'DLNR',
    gs2mesh_tsdf_voxel_size: 0.005,
    gs2mesh_tsdf_depth_trunc: 0.03,
    gs2mesh_use_masks: true,
  },
};

export const STAGE_LABELS = {
  1: 'Extract Frames',
  2: 'COLMAP',
  3: 'SAM2',
  4: 'gs2mesh',
  5: 'Texture Bake',
  6: 'Post Cleanup',
};
