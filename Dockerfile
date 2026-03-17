# clip2mesh: Docker-based 3D reconstruction pipeline
# COLMAP + 3D Gaussian Splatting + gs2mesh + SAM2

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV QT_QPA_PLATFORM=offscreen

# --- Layer 1: System dependencies + COLMAP ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-distutils \
    git wget curl ffmpeg \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 \
    libegl1-mesa libgbm1 \
    colmap \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Layer 2: PyTorch (CUDA 12.1 wheels, matches base image) ---
RUN pip install --no-cache-dir \
    "torch>=2.5.0" torchvision \
    --index-url https://download.pytorch.org/whl/cu121

# --- Layer 3: Python dependencies ---
RUN pip install --no-cache-dir \
    "numpy>=1.24.0,<2.0" \
    einops timm \
    "fastapi>=0.104.0" \
    "uvicorn[standard]>=0.24.0" \
    python-multipart \
    "opencv-python-headless>=4.8.0" \
    "plyfile>=1.0.0" \
    "scipy>=1.10.0" \
    "scikit-learn>=1.3.0" \
    "open3d>=0.17.0" \
    "trimesh>=3.20.0" \
    xatlas \
    "pillow>=10.0.0" \
    "tqdm>=4.65.0" \
    hydra-core iopath \
    wandb \
    && pip install --no-cache-dir scikit-image

# CUDA arch list needed at build time (no GPU in Docker build context)
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"

# --- Layer 3b: nvdiffrast (GPU rasterization for texture baking) ---
# Requires CUDA toolkit + PyTorch; not on PyPI, install from GitHub.
RUN pip install --no-cache-dir --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git

# --- Layer 4: SAM2 (editable install with CUDA extension for post-processing) ---
RUN git clone https://github.com/facebookresearch/sam2.git /opt/sam2 \
    && cd /opt/sam2 \
    && SAM2_BUILD_CUDA=1 pip install --no-build-isolation -e .

# --- Layer 5a: gaussian-splatting CUDA extensions ---

RUN git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git /opt/gaussian-splatting \
    && cd /opt/gaussian-splatting/submodules/diff-gaussian-rasterization \
    && git checkout 3dgs_accel \
    && pip install --no-cache-dir --no-build-isolation \
        /opt/gaussian-splatting/submodules/diff-gaussian-rasterization \
        /opt/gaussian-splatting/submodules/simple-knn \
    && pip install --no-cache-dir --no-build-isolation \
        /opt/gaussian-splatting/submodules/fused-ssim 2>/dev/null || true \
    && cd /opt/gaussian-splatting \
    && python3 -c "from diff_gaussian_rasterization import SparseGaussianAdam"

# Build a compat rasterizer overlay so stage 4 can fall back without changing
# reconstruction inputs or quality settings. Reuse the main checkout and only
# replace diff_gaussian_rasterization; the global simple_knn/fused-ssim wheels
# from the accel build remain available on the default site-packages path.
RUN mkdir -p /opt/gs-compat-site \
    && git clone --recursive https://github.com/graphdeco-inria/diff-gaussian-rasterization.git /opt/diff-gaussian-rasterization-compat \
    && pip install --no-cache-dir --no-build-isolation \
        --target /opt/gs-compat-site \
        /opt/diff-gaussian-rasterization-compat
RUN PYTHONPATH=/opt/gs-compat-site python3 -c "import diff_gaussian_rasterization"

# --- Layer 5b: gs2mesh + DLNR stereo weights ---
RUN git clone https://github.com/yanivw12/gs2mesh.git /opt/gs2mesh \
    && rm -rf /opt/gs2mesh/third_party/gaussian-splatting \
    && ln -s /opt/gaussian-splatting /opt/gs2mesh/third_party/gaussian-splatting \
    && cd /opt/gs2mesh \
    && pip install --no-cache-dir -r requirements.txt 2>/dev/null; true

# Explicitly install gs2mesh/DLNR deps that fail silently from requirements.txt.
# opt_einsum: DLNR core/update.py (confirmed missing).
# matplotlib, plotly, tensorboard, imageio: currently installed via the error-
# swallowing requirements.txt line — pin them here for robustness.
RUN pip install --no-cache-dir opt_einsum matplotlib plotly tensorboard imageio

# k3d is a Jupyter 3D widget imported by gs2mesh's visualize.py at module level.
# It's never used in our headless pipeline; stub it to avoid pulling Jupyter deps.
RUN mkdir -p /usr/local/lib/python3.11/dist-packages/k3d \
    && touch /usr/local/lib/python3.11/dist-packages/k3d/__init__.py

# GroundingDINO (auto-masking) has uninstalled deps (supervision, addict, yapf).
# We skip masking (--skip_masking), but masker_utils.py imports it at module level.
# Wrap in try/except so it doesn't crash at import time.
RUN python3 -c "\
p='/opt/gs2mesh/gs2mesh_utils/masker_utils.py'; \
t=open(p).read().replace( \
  'import third_party.GroundingDINO.groundingdino.util.inference as GD', \
  'try:\n    import third_party.GroundingDINO.groundingdino.util.inference as GD\nexcept (ImportError, ModuleNotFoundError):\n    GD = None'); \
open(p,'w').write(t)"

# gs2mesh hardcodes gaussian-splatting renderer integration.
# Newer gaussian-splatting expects `depths` / `train_test_exp` fields and a
# different Camera constructor signature.
RUN python3 - <<'PY'
from pathlib import Path

p = Path('/opt/gs2mesh/gs2mesh_utils/renderer_utils.py')
t = p.read_text()
if "from PIL import Image" not in t:
    t = t.replace("import copy\n", "import copy\nfrom PIL import Image\n", 1)
img = "                         images='images', \n"
ev = "                         eval=False, \n"
dbg = "                         debug=False, \n"
if "depths=''" not in t:
    t = t.replace(img, img + "                         depths='', \n", 1)
if "train_test_exp=False" not in t:
    t = t.replace(ev, ev + "                         train_test_exp=False, \n", 1)
if "antialiasing=False" not in t:
    t = t.replace(dbg, dbg + "                         antialiasing=False, \n", 1)
old_camera = '                view = cameras.Camera(0, R, T, FoVx, FoVy, torch.rand(3,h,w), None, "abcd", 0)\n'
new_camera = """                dummy_image = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))
                view = cameras.Camera(
                    (w, h),
                    0,
                    R,
                    T,
                    FoVx,
                    FoVy,
                    None,
                    dummy_image,
                    None,
                    f"{camera_name}",
                    camera_number,
                    data_device=self.device,
                )
"""
if old_camera in t:
    t = t.replace(old_camera, new_camera, 1)
p.write_text(t)
PY

RUN mkdir -p /opt/gs2mesh/third_party/DLNR/pretrained \
    && wget -q -O /opt/gs2mesh/third_party/DLNR/pretrained/DLNR_Middlebury.pth \
        https://github.com/David-Zhao-1997/High-frequency-Stereo-Matching-Network/releases/download/v1.0.0/DLNR_Middlebury.pth

# --- Layer 6: Application code ---
COPY scripts/ /app/scripts/

# --- Layer 7: Test dependencies & test files ---
COPY tests/ /app/tests/
RUN pip install --no-cache-dir pytest pytest-asyncio

# Add repos to Python path
ENV PYTHONPATH="/app:/opt/gs2mesh:/opt/gaussian-splatting:/opt/sam2:${PYTHONPATH}"

EXPOSE 7860

ENTRYPOINT ["python3", "-m", "uvicorn"]
CMD ["scripts.dashboard.app:app", "--host", "0.0.0.0", "--port", "7860"]
