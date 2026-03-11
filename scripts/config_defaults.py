"""Centralised default values for the clip2mesh.

Every tunable constant lives here as the single source of truth.
Stage scripts and dashboard code import from this module instead
of defining their own copies.

RULES
- stdlib only — no numpy, torch, cv2, etc.
- Two sections: USER-CONFIGURABLE (UI-facing) and INTERNAL (non-UI).
- Constants are grouped by pipeline stage.
"""

from __future__ import annotations

# ============================================================
# USER-CONFIGURABLE DEFAULTS (UI から操作可能な値)
# ============================================================

# --- Stage 1: ExtractFrames --------------------------------
EXTRACT_FRAME_INTERVAL = 10
EXTRACT_MAX_FRAMES = 50

# --- Stage 2: Pi3X ----------------------------------------
PI3X_PIXEL_LIMIT = 255_000
PI3X_FRAME_TARGET = 50
PI3X_CONFIDENCE_THRESHOLD = 0.2
PI3X_EDGE_RTOL = 0.03

# --- Stage 3: SAM2 ----------------------------------------
SAM2_DEFAULT_MODEL = "large"
SAM2_MODEL_CONFIGS: dict[str, dict[str, str]] = {
    "tiny": {
        "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "hf_model_id": "facebook/sam2.1-hiera-tiny",
    },
    "small": {
        "config": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "hf_model_id": "facebook/sam2.1-hiera-small",
    },
    "base": {
        "config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "hf_model_id": "facebook/sam2.1-hiera-base-plus",
    },
    "large": {
        "config": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "hf_model_id": "facebook/sam2.1-hiera-large",
    },
}

# --- Stage 4: Denoise -------------------------------------
DENOISE_DEFAULT_PRESET = "balanced"
DENOISE_DEFAULT_ALGORITHM = "dbscan_sor"
DENOISE_ALGORITHMS: set[str] = {
    "dbscan_sor",
    "dbscan_only",
    "sor_only",
    "radius_only",
    "dbscan_radius",
}
DENOISE_ALGORITHM_STEPS: dict[str, tuple[str, ...]] = {
    "dbscan_sor": ("dbscan", "sor"),
    "dbscan_only": ("dbscan",),
    "sor_only": ("sor",),
    "radius_only": ("radius",),
    "dbscan_radius": ("dbscan", "radius"),
}
DENOISE_PRESET_DEFAULTS: dict[str, dict[str, float | int | str]] = {
    "balanced": {
        "algorithm": "dbscan_sor",
        "dbscan_eps": 0.0,
        "dbscan_eps_ratio": 0.02,
        "dbscan_min_samples": 10,
        "dbscan_max_points": 500000,
        "sor_neighbors": 20,
        "sor_std_ratio": 2.0,
        "radius_neighbors": 8,
        "radius_ratio": 0.015,
    },
    "detail_preserving": {
        "algorithm": "sor_only",
        "dbscan_eps": 0.0,
        "dbscan_eps_ratio": 0.02,
        "dbscan_min_samples": 10,
        "dbscan_max_points": 500000,
        "sor_neighbors": 16,
        "sor_std_ratio": 2.6,
        "radius_neighbors": 8,
        "radius_ratio": 0.015,
    },
    "isolate_subject": {
        "algorithm": "dbscan_only",
        "dbscan_eps": 0.0,
        "dbscan_eps_ratio": 0.018,
        "dbscan_min_samples": 8,
        "dbscan_max_points": 500000,
        "sor_neighbors": 20,
        "sor_std_ratio": 2.0,
        "radius_neighbors": 8,
        "radius_ratio": 0.015,
    },
    "sparse_noise": {
        "algorithm": "radius_only",
        "dbscan_eps": 0.0,
        "dbscan_eps_ratio": 0.02,
        "dbscan_min_samples": 10,
        "dbscan_max_points": 500000,
        "sor_neighbors": 20,
        "sor_std_ratio": 2.0,
        "radius_neighbors": 6,
        "radius_ratio": 0.012,
    },
    "aggressive_cleanup": {
        "algorithm": "dbscan_radius",
        "dbscan_eps": 0.0,
        "dbscan_eps_ratio": 0.024,
        "dbscan_min_samples": 14,
        "dbscan_max_points": 500000,
        "sor_neighbors": 20,
        "sor_std_ratio": 2.0,
        "radius_neighbors": 10,
        "radius_ratio": 0.02,
    },
}

# --- Stage 5a: ClassicalMesh ------------------------------
CLASSICAL_DEFAULT_PRESET = "trust_point_cloud"
CLASSICAL_PREPROCESS_ENABLED = False
CLASSICAL_PREPROCESS_VOXEL_RATIO = 0.003
CLASSICAL_PREPROCESS_MAX_POINTS = 700_000
CLASSICAL_PREPROCESS_SOR_NEIGHBORS = 20
CLASSICAL_PREPROCESS_SOR_STD_RATIO = 3.2
CLASSICAL_POISSON_DEPTH = 11
CLASSICAL_DENSITY_TRIM_Q = 0.001
CLASSICAL_AUTO_SMOOTH = False
CLASSICAL_SMOOTH_METHOD = "laplacian"
CLASSICAL_SMOOTH_ITERATIONS = 0
CLASSICAL_SMOOTH_LAMBDA = 0.5
CLASSICAL_SMOOTH_TAUBIN_NU = -0.53
CLASSICAL_DOWNSAMPLE_ENABLED = False
CLASSICAL_DOWNSAMPLE_TARGET_FACES = 500_000
CLASSICAL_DOWNSAMPLE_TRIGGER_FACES = 170_000
CLASSICAL_PRESET_DEFAULTS: dict[str, dict[str, object]] = {
    "lightweight": {
        "classical_preprocess_enabled": True,
        "classical_poisson_depth": 9,
        "classical_density_trim_q": 0.005,
        "classical_auto_smooth": False,
        "classical_smooth_iterations": 2,
        "classical_downsample_enabled": True,
        "classical_downsample_target_faces": 120_000,
    },
    "trust_point_cloud": {
        "classical_preprocess_enabled": False,
        "classical_poisson_depth": 11,
        "classical_density_trim_q": 0.001,
        "classical_auto_smooth": False,
        "classical_smooth_iterations": 0,
        "classical_downsample_enabled": False,
        "classical_downsample_target_faces": 500_000,
    },
}

# --- Stage 5b: DiffCD ------------------------------------
DIFFCD_BATCH_SIZE = 5000
DIFFCD_N_BATCHES = 30000
DIFFCD_RESOLUTION = 512
DIFFCD_SMOOTH_METHOD = "laplacian"
DIFFCD_SMOOTH_ITERATIONS = 2
DIFFCD_SMOOTH_LAMBDA = 0.5
DIFFCD_SMOOTH_TAUBIN_NU = -0.53

# --- Stage 6: MeshWrap -----------------------------------
MESHWRAP_ITERATIONS = 1
MESHWRAP_SAMPLE_POINTS = 400_000
MESHWRAP_POISSON_DEPTH = 8
MESHWRAP_POISSON_SCALE = 1.18
MESHWRAP_DENSITY_TRIM_Q = 0.003
MESHWRAP_TARGET_FACE_RATIO = 2.20
MESHWRAP_CROP_SCALE = 1.08
MESHWRAP_NORMAL_RADIUS_RATIO = 0.02
MESHWRAP_SMOOTH_ITERATIONS = 2
MESHWRAP_QUALITY_THRESHOLD = 0.02
MESHWRAP_ALPHA_RATIO = 0.02
MESHWRAP_OFFSET_RATIO = 0.3

# --- Stage 7: MeshRepair ---------------------------------
REPAIR_ENABLED = True
REPAIR_MAX_DIAMETER_RATIO = 0.08
REPAIR_Y_BAND_RATIO = 0.06
REPAIR_SMOOTH_ITERS = 3

# --- Stage 8: TextureBake --------------------------------
TEXTURE_SIZE = 0
TEXTURE_VIEW_ASSIGN_MODE = "legacy"
TEXTURE_VIEW_ASSIGN_MODES: set[str] = {"legacy", "region_gc"}
TEXTURE_QUALITY_BOOST = False
TEXTURE_OVERSAMPLE = 2
TEXTURE_MIN_COS = 0.2
TEXTURE_ANGLE_EXP = 4.0
TEXTURE_DIST_POW = 1.0
TEXTURE_SHARPEN = 0.15
TEXTURE_BLEND_TOPK = 3
TEXTURE_BLEND_HARD_RATIO = 2.0

# --- Mesh Method ------------------------------------------
MESH_METHODS: set[str] = {"poisson", "diffcd"}
MESH_DEFAULT_METHOD = "poisson"


# ============================================================
# INTERNAL CONSTANTS (UI から操作不可の値)
# ============================================================

# --- Stage 1: ExtractFrames --------------------------------
_EXTRACT_FPS_FALLBACK = 30

# --- Stage 2: Pi3X ----------------------------------------
_PI3X_OOM_FRAME_REDUCTION_FACTOR = 0.80
_PI3X_OOM_REDUCTION_FACTOR = 0.70
_PI3X_MIN_INFER_FRAMES = 12
_PI3X_MIN_PIXEL_LIMIT = 50_000
_PI3X_MAX_OOM_RETRIES = 7
_PI3X_USE_CHUNK_FALLBACK = True
_PI3X_CHUNK_SCALE_MIN = 0.85
_PI3X_CHUNK_SCALE_MAX = 1.15
_PI3X_ALIGN_MIN_FRAMES = 4
_PI3X_ALIGN_MIN_LINE_RATIO = 1e-3
_PI3X_ALIGN_MAX_PLANAR_RATIO = 0.35
_PI3X_ALIGN_MIN_RADIUS = 1e-4
_PI3X_ALIGN_ORIENTATION_SCORE_EPS = 1e-4
_PI3X_TARGET_PLANE_NORMAL = (0.0, 1.0, 0.0)

# --- Stage 3: SAM2 ----------------------------------------
# _NORM_UPPER requires numpy (nextafter) — stays in stage_sam2_ui.py
_SAM2_MASK_OVERLAY_ALPHA = 0.45
_SAM2_MASK_COLOR = (30, 144, 255)
_SAM2_CIRCLE_RADIUS = 8
_SAM2_CIRCLE_OUTLINE_WIDTH = 2

# --- Stage 5a: ClassicalMesh ------------------------------
_CLASSICAL_NORMAL_RADIUS_RATIO = 0.02
_CLASSICAL_NORMAL_MAX_NN = 32
_CLASSICAL_NORMAL_ORIENT_K = 24
_CLASSICAL_POISSON_SCALE = 1.08
_CLASSICAL_POISSON_LINEAR_FIT = False
_CLASSICAL_CROP_SCALE = 1.06
_CLASSICAL_POST_MIN_COMPONENT_TRIANGLES = 150
_CLASSICAL_POST_MIN_COMPONENT_RATIO = 0.005
_CLASSICAL_PREVIEW_TARGET_FACES = 120_000

# --- Stage 5b: DiffCD ------------------------------------
_DIFFCD_MIN_BATCH_SIZE = 500
_DIFFCD_BATCH_STEP = 100
_DIFFCD_MIN_N_BATCHES = 1_000
_DIFFCD_AUTO_MIN_N_BATCHES = 10_000
_DIFFCD_SMOOTH_METHODS: set[str] = {"laplacian", "taubin"}
_DIFFCD_OOM_MARKERS: tuple[str, ...] = (
    "out of memory",
    "resource exhausted",
    "cuda_error_out_of_memory",
    "cudnn_status_alloc_failed",
    "std::bad_alloc",
)

# --- Stage 6: MeshWrap -----------------------------------
_MESHWRAP_ENABLED = True
_MESHWRAP_METHOD = "alpha_wrap"
_MESHWRAP_METHODS: set[str] = {"poisson_iterative", "alpha_wrap"}
_MESHWRAP_NORMAL_MAX_NN = 32
_MESHWRAP_NORMAL_ORIENT_K = 24
_MESHWRAP_POISSON_LINEAR_FIT = False
_MESHWRAP_KEEP_LARGEST_COMPONENT = True
_MESHWRAP_MIN_FACES = 25_000
_MESHWRAP_MAX_FACES = 200_000
_MESHWRAP_PRESERVE_INPUT_ON_FAILURE = True
_MESHWRAP_SMOOTH_LAMBDA = 0.5
_MESHWRAP_SMOOTH_NU = -0.53
_MESHWRAP_QUALITY_SAMPLE_POINTS = 50_000

# --- Stage 7: MeshRepair ---------------------------------
_REPAIR_SMOOTH_LAMBDA = 0.18
_REPAIR_MIN_DOWNWARD_NORMAL_Y = 0.25
_REPAIR_MIN_LOOP_VERTICES = 4

# --- Stage 8: TextureBake --------------------------------
_TEXTURE_CACHE_SAFETY_MB = 1024.0
_TEXTURE_FRAME_BUDGET_RATIO = 0.7
_TEXTURE_MASK_BUDGET_RATIO = 0.3
_TEXTURE_MEM_FALLBACK_MB = 4096.0

# --- VRAM Management -------------------------------------
_VRAM_PI3X_TARGET_UTILIZATION = 0.95
_VRAM_PI3X_ESTIMATED_MODEL_MB = 5500
_VRAM_PI3X_RUNTIME_OVERHEAD_MB = 1200
_VRAM_PI3X_FRAME_PIXELS_PER_MB = 800
_VRAM_GATE_MIN_FREE_MB = 12_000

# --- Infrastructure --------------------------------------
_LOG_QUEUE_MAXSIZE = 4096
_OUTPUT_DIR_DEFAULT = "/data/output"
