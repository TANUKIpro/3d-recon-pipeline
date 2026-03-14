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
COLMAP_MATCHER = "sequential"
COLMAP_MAX_FEATURES = 8192
COLMAP_IMAGE_SIZE = 1024

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
GS2MESH_GS_ITERATIONS = 30000
GS2MESH_STEREO_MODEL = "DLNR_Middlebury"
GS2MESH_TSDF_VOXEL_SIZE = 0.005
GS2MESH_TSDF_DEPTH_TRUNC = 0.04
GS2MESH_USE_MASKS = True

# --- Stage 5: TextureBake --------------------------------
TEXTURE_SIZE = 0
TEXTURE_VIEW_ASSIGN_MODE = "region_gc"
TEXTURE_VIEW_ASSIGN_MODES: set[str] = {"legacy", "region_gc"}
TEXTURE_QUALITY_BOOST = True
TEXTURE_OVERSAMPLE = 2
TEXTURE_MIN_COS = 0.2
TEXTURE_ANGLE_EXP = 4.0
TEXTURE_DIST_POW = 1.0
TEXTURE_SHARPEN = 0.15
TEXTURE_BLEND_TOPK = 3
TEXTURE_BLEND_HARD_RATIO = 2.0

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

# --- VRAM Management -------------------------------------
_VRAM_GATE_MIN_FREE_MB = 12_000

# --- Infrastructure --------------------------------------
_LOG_QUEUE_MAXSIZE = 4096
_OUTPUT_DIR_DEFAULT = OUTPUT_DIR_DEFAULT  # backward-compat alias
