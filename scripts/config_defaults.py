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

# --- Stage 4 pre-process: COLMAP Sparse Filter -------------
COLMAP_FILTER_ENABLED = True
COLMAP_FILTER_MIN_INSIDE_VIEWS = 2
COLMAP_FILTER_MIN_INSIDE_RATIO = 0.6
COLMAP_FILTER_MASK_DILATE = 1
COLMAP_FILTER_MIN_POINTS = 200

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

# --- Stage 4: MILo Reconstruction -------------------------
MILO_PRESET = "default"
MILO_PRESET_CUSTOM = "custom"
MILO_ITERATIONS = 12000
MILO_SCENE_TYPE = "indoor"
MILO_SCENE_TYPES: set[str] = {"indoor", "outdoor"}
MILO_MESH_CONFIG = "default"
MILO_MESH_CONFIGS: set[str] = {"default", "highres", "veryhighres", "lowres", "verylowres"}
MILO_EXTRACTION_METHOD = "sdf"
MILO_EXTRACTION_METHODS: set[str] = {"sdf", "integration", "regular_tsdf"}
MILO_RASTERIZER = "radegs"
MILO_RASTERIZERS: set[str] = {"radegs", "gof"}
MILO_USE_MASKS = True
MILO_DENSE_GAUSSIANS = False
MILO_DATA_DEVICE = "cuda"
MILO_MESH_RES = 1024
MILO_PRESETS: dict[str, dict[str, object]] = {
    "default": {
        "iterations": MILO_ITERATIONS,
        "scene_type": MILO_SCENE_TYPE,
        "mesh_config": MILO_MESH_CONFIG,
        "extraction_method": MILO_EXTRACTION_METHOD,
        "rasterizer": MILO_RASTERIZER,
        "use_masks": MILO_USE_MASKS,
        "dense_gaussians": MILO_DENSE_GAUSSIANS,
        "data_device": MILO_DATA_DEVICE,
        "mesh_res": MILO_MESH_RES,
    },
    "high": {
        "iterations": 30000,
        "scene_type": "indoor",
        "mesh_config": "highres",
        "extraction_method": "sdf",
        "rasterizer": "radegs",
        "use_masks": True,
        "dense_gaussians": True,
        "data_device": "cuda",
        "mesh_res": 1024,
    },
    "fast": {
        "iterations": 7000,
        "scene_type": "indoor",
        "mesh_config": "lowres",
        "extraction_method": "regular_tsdf",
        "rasterizer": "radegs",
        "use_masks": True,
        "dense_gaussians": False,
        "data_device": "cuda",
        "mesh_res": 512,
    },
}
MILO_PRESET_CHOICES: set[str] = set(MILO_PRESETS) | {MILO_PRESET_CUSTOM}

# --- Stage 5: TextureBake --------------------------------
TEXTURE_MODE = "multi_view"
TEXTURE_MODES: set[str] = {"multi_view", "vertex_color"}
TEXTURE_SIZE = 0
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

# --- Stage 6: PostTextureContactCleanup -------------------
POST_TEXTURE_CLEANUP_ENABLED = True
CLEANUP_LOWER_HALF_THRESHOLD = 0.2
UNTEXTURED_FILL_ENABLED = True

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

# xatlas UV atlas face budget — auto-scaling parameters
_TEXTURE_UV_MIN_FACES = 50_000       # 下限: これ以下には簡略化しない
_TEXTURE_UV_MAX_FACES = 0            # 0=無制限 (並列UV生成により時間制約を撤廃)
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

# --- Stage 6: Untextured Face Fill -----------------------
_UNTEXTURED_FILL_EMPTY_THRESHOLD = 10    # sum(RGB) < この値で「空テクセル」判定
_UNTEXTURED_FILL_EMPTY_FRACTION = 0.90   # 面のテクセルの90%以上が空なら「無テクスチャ面」
_UNTEXTURED_FILL_INPAINT_RADIUS = 5      # cv2.inpaint の補間半径

# --- Infrastructure --------------------------------------
_LOG_QUEUE_MAXSIZE = 4096
_OUTPUT_DIR_DEFAULT = OUTPUT_DIR_DEFAULT  # backward-compat alias
