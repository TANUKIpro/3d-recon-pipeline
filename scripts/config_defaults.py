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

# --- Stage 2: COLMAP SfM -----------------------------------
COLMAP_MATCHER = "exhaustive"
COLMAP_MAX_FEATURES = 32768
COLMAP_IMAGE_SIZE = 2048
COLMAP_USE_GPU = False
COLMAP_DSP_SIFT = True
COLMAP_FIRST_OCTAVE = -1

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

# --- Stage 4: gs2mesh Reconstruction -----------------------
GS2MESH_PRESET = "default"
GS2MESH_PRESET_CUSTOM = "custom"
GS2MESH_GS_ITERATIONS = 5000
GS2MESH_RUNTIME_PROFILE = "auto"
GS2MESH_RUNTIME_PROFILES: set[str] = {"auto", "compat"}
GS2MESH_STEREO_MODEL = "DLNR_Middlebury"
GS2MESH_TSDF_VOXEL_SIZE = 0.005
GS2MESH_TSDF_DEPTH_TRUNC = 0.04
GS2MESH_USE_MASKS = True
GS2MESH_TSDF_SCALE = 1.0
GS2MESH_TSDF_MIN_DEPTH_BASELINES = 4
GS2MESH_TSDF_MAX_DEPTH_BASELINES = 20
GS2MESH_TSDF_DILATE = 1
GS2MESH_TSDF_CLEANING_THRESHOLD = 100_000
GS2MESH_TSDF_USE_OCCLUSION_MASK = True
GS2MESH_TSDF_INVERT_MASK = False
GS2MESH_TSDF_ERODE_MASK = True
GS2MESH_TSDF_EROSION_KERNEL_SIZE = 10
GS2MESH_TSDF_CLOSING_KERNEL_SIZE = 10
GS2MESH_TSDF_BLOCK_COUNT = 100_000
# TSDF device — overridden per VRAM tier at runtime via
# scripts/vram_tier.py (CPU:0 on low-VRAM tiers so the same parameters
# run reliably without VRAM-reduction fallbacks).
GS2MESH_TSDF_DEVICE = "CUDA:0"
GS2MESH_PRESETS: dict[str, dict[str, object]] = {
    "default": {
        "gs_iterations": GS2MESH_GS_ITERATIONS,
        "runtime_profile": GS2MESH_RUNTIME_PROFILE,
        "stereo_model": GS2MESH_STEREO_MODEL,
        "tsdf_voxel_size": GS2MESH_TSDF_VOXEL_SIZE,
        "tsdf_depth_trunc": GS2MESH_TSDF_DEPTH_TRUNC,
        "use_masks": GS2MESH_USE_MASKS,
        "tsdf_scale": GS2MESH_TSDF_SCALE,
        "tsdf_min_depth_baselines": GS2MESH_TSDF_MIN_DEPTH_BASELINES,
        "tsdf_max_depth_baselines": GS2MESH_TSDF_MAX_DEPTH_BASELINES,
        "tsdf_dilate": GS2MESH_TSDF_DILATE,
        "tsdf_cleaning_threshold": GS2MESH_TSDF_CLEANING_THRESHOLD,
        "tsdf_use_occlusion_mask": GS2MESH_TSDF_USE_OCCLUSION_MASK,
        "tsdf_invert_mask": GS2MESH_TSDF_INVERT_MASK,
        "tsdf_erode_mask": GS2MESH_TSDF_ERODE_MASK,
        "tsdf_erosion_kernel_size": GS2MESH_TSDF_EROSION_KERNEL_SIZE,
        "tsdf_closing_kernel_size": GS2MESH_TSDF_CLOSING_KERNEL_SIZE,
        "block_count": GS2MESH_TSDF_BLOCK_COUNT,
        "tsdf_device": GS2MESH_TSDF_DEVICE,
    },
    "high": {
        "gs_iterations": 15000,
        "runtime_profile": "auto",
        "stereo_model": "DLNR_Middlebury",
        "tsdf_voxel_size": 0.005,
        "tsdf_depth_trunc": 0.03,
        "use_masks": True,
        "tsdf_scale": 1.0,
        "tsdf_min_depth_baselines": 4,
        "tsdf_max_depth_baselines": 20,
        "tsdf_dilate": 1,
        "tsdf_cleaning_threshold": 50_000,
        "tsdf_use_occlusion_mask": True,
        "tsdf_invert_mask": False,
        "tsdf_erode_mask": True,
        "tsdf_erosion_kernel_size": 8,
        "tsdf_closing_kernel_size": 8,
        "block_count": 100_000,
        "tsdf_device": GS2MESH_TSDF_DEVICE,
    },
}
GS2MESH_PRESET_CHOICES: set[str] = set(GS2MESH_PRESETS) | {
    GS2MESH_PRESET_CUSTOM
}

# Mesh decimation (post-MC, pre-PLY-write in scripts/gpu_tsdf.py).
# Targets are resolved per VRAM tier × preset in scripts/vram_tier.py.
# Set MESH_DECIMATION=0 (env) to fully disable and reproduce legacy output.
MESH_DECIMATION_ENABLED = True
MESH_TARGET_FACES_DEFAULT = 200_000
MESH_TARGET_FACES_HIGH = 500_000
# CPU-TSDF tiers run on lower-spec hardware; cap mesh complexity tighter.
MESH_TARGET_FACES_DEFAULT_CPU = 80_000
MESH_TARGET_FACES_HIGH_CPU = 200_000
# Silhouette IoU rollback gate. Compares pre/post-decimation rasterised
# silhouettes from 8 evenly-sampled poses. If mean IoU < threshold the
# decimation is rolled back to the original mesh.
MESH_DECIMATION_MIN_IOU = 0.985
MESH_DECIMATION_IOU_VIEWS = 8

# Taubin smoothing applied to the cleaned TSDF mesh before decimation. The
# raw TSDF surface carries ~0.5 mm vertex displacement noise; on cylindrical
# objects this maps to 1.5–3 px shifts in projected pixel space at typical
# capture distances, which the texture bake amplifies into per-region speckle
# (each region_gc-locked patch sees fine details from a slightly different
# image location). A few iterations of curvature-flow smoothing remove the
# sub-mm noise while preserving silhouettes (Taubin's two-pass λ/μ filter
# does not shrink the mesh, unlike pure Laplacian). Set MESH_SMOOTH_ITERS=0
# (env) to disable. Same silhouette-IoU rollback as decimation.
MESH_SMOOTH_ENABLED = True
MESH_SMOOTH_ITERS = 5
MESH_SMOOTH_LAMBDA = 0.5
MESH_SMOOTH_MU = -0.53
MESH_SMOOTH_MIN_IOU = 0.985

# --- Stage 5: TextureBake --------------------------------
TEXTURE_SIZE = 0
# Cap auto-resolved texture size (sqrt(W·H)). Manual TEXTURE_SIZE>0
# bypasses this cap. Set TEXTURE_MAX_SIZE=0 to disable the cap entirely.
TEXTURE_MAX_SIZE = 2048
TEXTURE_VIEW_ASSIGN_MODE = "region_gc"
TEXTURE_VIEW_ASSIGN_MODES: set[str] = {"legacy", "region_gc"}
TEXTURE_QUALITY_BOOST = False
TEXTURE_OVERSAMPLE = 2
TEXTURE_MIN_COS = 0.2
TEXTURE_ANGLE_EXP = 4.0
TEXTURE_DIST_POW = 1.0
TEXTURE_SHARPEN = 0.15
TEXTURE_BLEND_TOPK = 3
TEXTURE_BLEND_HARD_RATIO = 2.0
TEXTURE_COLOR_HARDENING = True
TEXTURE_COLOR_HARDENING_THRESHOLD = 0.18

# --- Stage 6: PostTextureContactCleanup -------------------
POST_TEXTURE_CLEANUP_ENABLED = True
CLEANUP_LOWER_HALF_THRESHOLD = 0.2

# --- Shared Repair Defaults -------------------------------
# Contact-hole repair helpers are reused by post-texture cleanup.
# Keep these defaults available for backward compatibility even though the
# original standalone repair stage is no longer part of the main pipeline.
REPAIR_ENABLED = True
REPAIR_MAX_DIAMETER_RATIO = 0.46
REPAIR_Y_BAND_RATIO = 0.06
REPAIR_SMOOTH_ITERS = 3

# --- Infrastructure (user-facing) -------------------------
OUTPUT_DIR_DEFAULT = "/data/output"


# ============================================================
# INTERNAL CONSTANTS (UI から操作不可の値)
# ============================================================

# --- Stage 1: ExtractFrames --------------------------------
_EXTRACT_FPS_FALLBACK = 30

# --- Stage 2: COLMAP SfM -----------------------------------
_COLMAP_MATCHERS: set[str] = {"sequential", "exhaustive"}

# --- Stage 3: SAM2 ----------------------------------------
# _NORM_UPPER requires numpy (nextafter) — stays in stage_sam2_ui.py
_SAM2_MASK_OVERLAY_ALPHA = 0.45
_SAM2_MASK_COLOR = (30, 144, 255)
_SAM2_GROUND_MASK_COLOR = (255, 165, 0)
_SAM2_GROUND_MASK_OVERLAY_ALPHA = 0.35
_SAM2_CIRCLE_RADIUS = 8
_SAM2_CIRCLE_OUTLINE_WIDTH = 2
GROUND_PLANE_ENABLED = True

# --- Stage 5: TextureBake --------------------------------
_TEXTURE_CACHE_SAFETY_MB = 1024.0
_TEXTURE_FRAME_BUDGET_RATIO = 0.7
_TEXTURE_MASK_BUDGET_RATIO = 0.3
_TEXTURE_MEM_FALLBACK_MB = 4096.0
_TEXTURE_SEAM_PAD_ITERS = 8  # dilation iterations for the final UV-seam pad

# xatlas UV atlas face budget — auto-scaling parameters.
# _TEXTURE_UV_MAX_FACES acts as a safety net when MESH_DECIMATION is disabled
# (decimation already happens upstream in scripts/gpu_tsdf.py).
_TEXTURE_UV_MIN_FACES = 50_000       # 下限: これ以下には簡略化しない
_TEXTURE_UV_MAX_FACES = 300_000      # 0=無制限。Stage 4減面が無効化された場合の保険
_TEXTURE_UV_BYTES_PER_FACE = 10_000  # xatlasの1面あたり推定メモリ (~10 KB)
_TEXTURE_UV_RAM_RESERVE_MB = 2048.0  # xatlas以外のパイプライン用に確保するRAM (MB)

# Parallel UV atlas generation — spatial partition parameters
_TEXTURE_UV_PARALLEL_MIN_TOTAL_FACES = 10_000  # これ以下は並列化しない
_TEXTURE_UV_PARALLEL_MAX_WORKERS = 8           # 最大ワーカー数

# --- VRAM Management -------------------------------------
_VRAM_GATE_MIN_FREE_MB = 12_000
_VRAM_GATE_STRICT = True

# --- Shared Repair Internals ------------------------------
_REPAIR_SMOOTH_LAMBDA = 0.18
_REPAIR_MIN_DOWNWARD_NORMAL_Y = 0.25
_REPAIR_MIN_LOOP_VERTICES = 4
_REPAIR_GROUND_SECTION_MIN_AREA_RATIO = 1e-3
_REPAIR_GROUND_SECTION_START_QUANTILE = 0.75
_REPAIR_SKIRT_MIN_AREA_RATIO = 0.05  # 底面cap面積が最大capの5%未満なら細い柱扱い
_REPAIR_SKIRT_SLENDERNESS_MAX = 4.0  # skirt_height / sqrt(cap_area) がこれを超えたら細い柱扱い

# --- Infrastructure --------------------------------------
_LOG_QUEUE_MAXSIZE = 4096
_OUTPUT_DIR_DEFAULT = OUTPUT_DIR_DEFAULT  # backward-compat alias
