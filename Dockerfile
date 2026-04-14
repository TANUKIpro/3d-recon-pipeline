# clip2mesh: Docker-based 3D reconstruction pipeline
# COLMAP + MILo (Mesh-In-the-Loop 3DGS) + SAM2

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

# --- Layer 5: MILo (Mesh-In-the-Loop 3DGS) ---

# tetra_triangulation requires cmake, GMP, CGAL for Delaunay triangulation
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake libgmp-dev libcgal-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git config --global url."https://github.com/".insteadOf "git@github.com:" \
    && git clone --recursive https://github.com/Anttwo/MILo.git /opt/MILo

# Build MILo's gaussian rasterizers and core CUDA extensions
RUN cd /opt/MILo \
    && pip install --no-cache-dir --no-build-isolation \
        submodules/diff-gaussian-rasterization \
        submodules/diff-gaussian-rasterization_gof \
        submodules/diff-gaussian-rasterization_ms \
        submodules/simple-knn \
        submodules/fused-ssim

# Build tetra_triangulation (Delaunay mesh extraction)
# NOTE: setup.py has ext_modules commented out, so pip install only copies the
# Python package.  We cmake/make first to build the C++ SO, then pip-install
# the Python package, then manually copy the SO into the installed location.
RUN cd /opt/MILo/submodules/tetra_triangulation \
    && cmake -DCMAKE_CXX_FLAGS="-I${CUDA_HOME}/include" . && make \
    && pip install --no-cache-dir --no-build-isolation . \
    && cp tetranerf/utils/extension/tetranerf_cpp_extension.cpython-*.so \
       /usr/local/lib/python3.11/dist-packages/tetranerf/utils/extension/

# nvdiffrast is already installed from Layer 3b

# MILo additional Python deps (plyfile/tqdm/scikit-image already in Layer 3)
RUN pip install --no-cache-dir trimesh==4.6.8

# Verify MILo rasterizer build
RUN python3 -c "from diff_gaussian_rasterization import GaussianRasterizationSettings; print('MILo radegs rasterizer OK')"

# --- Layer 6: Application code ---
COPY scripts/ /app/scripts/

# --- Layer 7: Test dependencies & test files ---
COPY tests/ /app/tests/
RUN pip install --no-cache-dir pytest pytest-asyncio

# Allow git to read /app/.git even though the bind-mounted dir is owned by the host UID
RUN git config --system --add safe.directory /app

# Add repos to Python path
ENV PYTHONPATH="/app:/opt/MILo:/opt/sam2:${PYTHONPATH}"

EXPOSE 7860

ENTRYPOINT ["python3", "-m", "uvicorn"]
CMD ["scripts.dashboard.app:app", "--host", "0.0.0.0", "--port", "7860"]
