/**
 * Configuration preset constants for denoise, classical, meshwrap, and mesh repair.
 */

export const NEW_OBJECT_VALUE = '__new__';

export const STAGE_LABELS = {
  1: 'Extract Frames',
  2: 'Pi3X',
  3: 'SAM2',
  4: 'Denoise',
  5: 'Mesh Reconstruction',
  6: 'Mesh Wrap',
  7: 'Mesh Repair',
  8: 'Texture Bake',
};

export const DENOISE_CUSTOM_PRESET = 'custom';

export const DENOISE_ALGO_LABELS = {
  dbscan_sor: 'DBSCAN + SOR',
  dbscan_only: 'DBSCAN',
  sor_only: 'SOR',
  radius_only: 'Radius Outlier Removal',
  dbscan_radius: 'DBSCAN + Radius Outlier Removal',
};

export const DENOISE_ALGO_STEPS = {
  dbscan_sor: { dbscan: true, sor: true, radius: false },
  dbscan_only: { dbscan: true, sor: false, radius: false },
  sor_only: { dbscan: false, sor: true, radius: false },
  radius_only: { dbscan: false, sor: false, radius: true },
  dbscan_radius: { dbscan: true, sor: false, radius: true },
};

export const DENOISE_PRESETS = {
  balanced: {
    denoise_algorithm: 'dbscan_sor',
    denoise_dbscan_eps: 0.0,
    denoise_dbscan_eps_ratio: 0.02,
    denoise_dbscan_min_samples: 10,
    denoise_dbscan_max_points: 500000,
    denoise_sor_neighbors: 20,
    denoise_sor_std_ratio: 2.0,
    denoise_radius_neighbors: 8,
    denoise_radius_radius_ratio: 0.015,
  },
  detail_preserving: {
    denoise_algorithm: 'sor_only',
    denoise_dbscan_eps: 0.0,
    denoise_dbscan_eps_ratio: 0.02,
    denoise_dbscan_min_samples: 10,
    denoise_dbscan_max_points: 500000,
    denoise_sor_neighbors: 16,
    denoise_sor_std_ratio: 2.6,
    denoise_radius_neighbors: 8,
    denoise_radius_radius_ratio: 0.015,
  },
  isolate_subject: {
    denoise_algorithm: 'dbscan_only',
    denoise_dbscan_eps: 0.0,
    denoise_dbscan_eps_ratio: 0.018,
    denoise_dbscan_min_samples: 8,
    denoise_dbscan_max_points: 500000,
    denoise_sor_neighbors: 20,
    denoise_sor_std_ratio: 2.0,
    denoise_radius_neighbors: 8,
    denoise_radius_radius_ratio: 0.015,
  },
  sparse_noise: {
    denoise_algorithm: 'radius_only',
    denoise_dbscan_eps: 0.0,
    denoise_dbscan_eps_ratio: 0.02,
    denoise_dbscan_min_samples: 10,
    denoise_dbscan_max_points: 500000,
    denoise_sor_neighbors: 20,
    denoise_sor_std_ratio: 2.0,
    denoise_radius_neighbors: 6,
    denoise_radius_radius_ratio: 0.012,
  },
  aggressive_cleanup: {
    denoise_algorithm: 'dbscan_radius',
    denoise_dbscan_eps: 0.0,
    denoise_dbscan_eps_ratio: 0.024,
    denoise_dbscan_min_samples: 14,
    denoise_dbscan_max_points: 500000,
    denoise_sor_neighbors: 20,
    denoise_sor_std_ratio: 2.0,
    denoise_radius_neighbors: 10,
    denoise_radius_radius_ratio: 0.02,
  },
};

export const MESHWRAP_DEFAULTS = {
  meshwrap_method: 'alpha_wrap',
  meshwrap_poisson_depth: 8,
  meshwrap_poisson_scale: 1.18,
  meshwrap_iterations: 1,
  meshwrap_sample_points: 400000,
  meshwrap_density_trim_q: 0.003,
  meshwrap_crop_scale: 1.08,
  meshwrap_target_face_ratio: 2.20,
  meshwrap_normal_radius_ratio: 0.02,
  meshwrap_smooth_iterations: 2,
  meshwrap_quality_threshold: 0.02,
  meshwrap_alpha_ratio: 0.02,
  meshwrap_offset_ratio: 0.3,
};

export const MESH_REPAIR_DEFAULTS = {
  mesh_repair_max_diameter_ratio: 0.46,
  mesh_repair_y_band_ratio: 0.06,
  mesh_repair_smooth_iters: 3,
};

export const CLASSICAL_DEFAULTS = {
  classical_preprocess_enabled: false,
  classical_poisson_depth: 11,
  classical_density_trim_q: 0.001,
  classical_auto_smooth: false,
  classical_smooth_iterations: 0,
  classical_downsample_enabled: false,
  classical_downsample_target_faces: 500000,
};

export const CLASSICAL_PRESETS = {
  lightweight: {
    classical_preprocess_enabled: true,
    classical_poisson_depth: 9,
    classical_density_trim_q: 0.005,
    classical_auto_smooth: false,
    classical_smooth_iterations: 2,
    classical_downsample_enabled: true,
    classical_downsample_target_faces: 120000,
  },
  trust_point_cloud: {
    classical_preprocess_enabled: false,
    classical_poisson_depth: 11,
    classical_density_trim_q: 0.001,
    classical_auto_smooth: false,
    classical_smooth_iterations: 0,
    classical_downsample_enabled: false,
    classical_downsample_target_faces: 500000,
  },
};
