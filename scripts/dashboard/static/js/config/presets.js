/**
 * Configuration preset constants for the clip2mesh pipeline.
 */

export const NEW_OBJECT_VALUE = '__new__';
export const GWRAPPING_PRESET_DEFAULT = 'default';
export const GWRAPPING_PRESET_HIGH = 'high';
export const GWRAPPING_PRESET_FAST = 'fast';
export const GWRAPPING_PRESET_CUSTOM = 'custom';
export const GWRAPPING_PUBLIC_PRESETS = {
  [GWRAPPING_PRESET_DEFAULT]: {
    gwrapping_iterations: 30000,
    gwrapping_rasterizer: 'ours',
    gwrapping_resolution: 2,
    gwrapping_use_masks: true,
    gwrapping_extraction_method: 'pivot',
  },
  [GWRAPPING_PRESET_HIGH]: {
    gwrapping_iterations: 40000,
    gwrapping_rasterizer: 'ours',
    gwrapping_resolution: 1,
    gwrapping_use_masks: true,
    gwrapping_extraction_method: 'pivot',
  },
  [GWRAPPING_PRESET_FAST]: {
    gwrapping_iterations: 15000,
    gwrapping_rasterizer: 'radegs',
    gwrapping_resolution: 4,
    gwrapping_use_masks: true,
    gwrapping_extraction_method: 'pivot',
  },
};

export const STAGE_LABELS = {
  1: 'Extract Frames',
  2: 'COLMAP',
  3: 'SAM2',
  4: 'GWrapping',
  5: 'Texture Bake',
  6: 'Post Cleanup',
};
