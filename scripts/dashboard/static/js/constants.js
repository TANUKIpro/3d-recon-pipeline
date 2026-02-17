/**
 * Shared constants used across dashboard modules.
 */

export const STAGE_COUNT = 8;
export const TRANSITION_STAGE_MAX = 7;

export const MESH_METHOD_DEFAULT = 'poisson';
export const MESH_METHOD_SET = new Set(['diffcd', 'poisson']);

export const MESH_REPAIR_THRESHOLD_MIN = -1.0;
export const MESH_REPAIR_THRESHOLD_MAX = 0.0;
export const MESH_REPAIR_THRESHOLD_DEFAULT = -0.25;

export const DEFAULT_TAUBIN_NU = -0.53;
export const CLASSICAL_PREVIEW_TITLE = 'Classical Mesh result';

export const STORAGE_KEYS = {
  theme: 'clip2mesh:theme',
  lang: 'clip2mesh:lang',
  autoScroll: 'clip2mesh:log.autoScroll',
  maxLines: 'clip2mesh:log.maxLines',
  autoAccept: 'clip2mesh:autoAccept',
};
