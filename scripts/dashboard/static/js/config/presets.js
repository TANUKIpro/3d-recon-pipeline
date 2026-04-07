/**
 * Configuration preset constants for the MILo pipeline.
 */

export const NEW_OBJECT_VALUE = '__new__';
export const MILO_PRESET_DEFAULT = 'default';
export const MILO_PRESET_HIGH = 'high';
export const MILO_PRESET_FAST = 'fast';
export const MILO_PRESET_CUSTOM = 'custom';
export const MILO_PUBLIC_PRESETS = {
  [MILO_PRESET_DEFAULT]: {
    milo_iterations: 18000,
    milo_scene_type: 'indoor',
    milo_mesh_config: 'default',
    milo_extraction_method: 'sdf',
    milo_use_masks: true,
  },
  [MILO_PRESET_HIGH]: {
    milo_iterations: 30000,
    milo_scene_type: 'indoor',
    milo_mesh_config: 'highres',
    milo_extraction_method: 'sdf',
    milo_use_masks: true,
  },
  [MILO_PRESET_FAST]: {
    milo_iterations: 7000,
    milo_scene_type: 'indoor',
    milo_mesh_config: 'lowres',
    milo_extraction_method: 'sdf',
    milo_use_masks: true,
  },
};

export const STAGE_LABELS = {
  1: 'Extract Frames',
  2: 'COLMAP',
  3: 'SAM2',
  4: 'MILo',
  5: 'Texture Bake',
  6: 'Post Cleanup',
};
