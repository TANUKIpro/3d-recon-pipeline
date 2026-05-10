"""LiTo (Apple ml-lito) backend for Stage 4 reconstruction.

This package provides an alternative Stage 4 implementation that delegates
3D reconstruction to Apple's LiTo (ICLR 2026) image-to-3D model. The model
weights are licensed for research purposes only.

Modules (added across phases):
- config         : LiTo-specific tunables
- frame_selector : compound-score frame selection + quality gates
- mask_compositor: SAM2 mask + RGB → 518×518 RGBA letterbox
- lito_runner    : subprocess bridge to /opt/ml-lito/.venv
- gaussian_to_mesh: 3D Gaussians → mesh via opacity-weighted multi-view TSDF
- tsdf_core      : pure-function TSDF fusion (shared with gs2mesh backend)
- colmap_align   : Sim(3) alignment (umeyama → ICP) into COLMAP world frame
"""
