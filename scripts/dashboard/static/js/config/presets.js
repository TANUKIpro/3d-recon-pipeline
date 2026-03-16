/**
 * Configuration preset constants for the gs2mesh pipeline.
 */

export const NEW_OBJECT_VALUE = '__new__';

export const STAGE_LABELS = {
  1: 'Extract Frames',
  2: 'COLMAP',
  3: 'SAM2',
  4: 'gs2mesh',
  5: 'Texture Bake',
  6: 'Post Cleanup',
};
