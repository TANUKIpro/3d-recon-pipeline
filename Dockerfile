# clip2mesh: Docker-based 3D reconstruction pipeline
# COLMAP + GaussianWrapping (3DGS surface reconstruction) + SAM2

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
    "torch>=2.5.0" torchvision torchaudio \
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
    wandb dacite \
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

# --- Layer 5: GaussianWrapping (3DGS surface reconstruction) ---

# tetra_triangulation requires cmake, GMP, CGAL for Delaunay triangulation
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake libgmp-dev libcgal-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone with all submodules (SSH→HTTPS redirect for Depth-Anything-V2, nvdiffrast)
RUN git config --global url."https://github.com/".insteadOf "git@github.com:" \
    && git clone --recursive https://github.com/diego1401/GaussianWrapping.git /opt/GaussianWrapping

# GaussianWrapping Python requirements (open3d, trimesh, dacite, etc.)
RUN cd /opt/GaussianWrapping \
    && pip install --no-cache-dir -r requirements.txt 2>/dev/null; true

# Build 4 rasterizers (Mini-Splatting2, RaDe-GS, Ours, SOF)
RUN pip install --no-cache-dir --no-build-isolation \
    /opt/GaussianWrapping/submodules/diff-gaussian-rasterization_ms \
    /opt/GaussianWrapping/submodules/diff-gaussian-rasterization \
    /opt/GaussianWrapping/submodules/diff-gaussian-rasterization_ours \
    /opt/GaussianWrapping/submodules/diff-gaussian-rasterization_sof

# Patch diff_gaussian_rasterization_sof for Python 3.11 dataclass compatibility.
# Mutable defaults (SortQueueSizes(), SortSettings(), CullingSettings()) are
# forbidden in Python >=3.10 dataclasses; replace with field(default_factory=...).
RUN python3 -c "\
p='/usr/local/lib/python3.11/dist-packages/diff_gaussian_rasterization_sof/__init__.py'; \
t=open(p).read(); \
t=t.replace('from dataclasses import dataclass, asdict', \
            'from dataclasses import dataclass, asdict, field'); \
t=t.replace('queue_sizes : SortQueueSizes = SortQueueSizes()', \
            'queue_sizes : SortQueueSizes = field(default_factory=SortQueueSizes)'); \
t=t.replace('sort_settings : SortSettings = SortSettings()', \
            'sort_settings : SortSettings = field(default_factory=SortSettings)'); \
t=t.replace('culling_settings : CullingSettings = CullingSettings()', \
            'culling_settings : CullingSettings = field(default_factory=CullingSettings)'); \
t=t.replace('meshing_settings : MeshingSettings = MeshingSettings()', \
            'meshing_settings : MeshingSettings = field(default_factory=MeshingSettings)'); \
open(p,'w').write(t); \
print('Patched diff_gaussian_rasterization_sof for Python 3.11')"

# Build simple-knn, fused-ssim
RUN pip install --no-cache-dir --no-build-isolation \
    /opt/GaussianWrapping/submodules/simple-knn \
    /opt/GaussianWrapping/submodules/fused-ssim

# Build tetra_triangulation (Delaunay mesh extraction)
# cmake+make builds the C++ SO, pip install copies the Python package,
# then manually copy the SO into the installed location.
RUN cd /opt/GaussianWrapping/submodules/tetra_triangulation \
    && cmake -DCMAKE_CXX_FLAGS="-I${CUDA_HOME}/include" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 . \
    && make \
    && pip install --no-cache-dir --no-build-isolation . \
    && cp tetranerf/utils/extension/tetranerf_cpp_extension.cpython-*.so \
       /usr/local/lib/python3.11/dist-packages/tetranerf/utils/extension/ 2>/dev/null; true

# Build warp-patch-ncc (inside Geometry-Grounded-GS submodule)
RUN pip install --no-cache-dir --no-build-isolation \
    /opt/GaussianWrapping/submodules/Geometry-Grounded-Gaussian-Splatting/submodules/warp-patch-ncc

# torch_geometric + PyG sparse/scatter wheels
RUN pip install --no-cache-dir torch_geometric \
    && pip install --no-cache-dir pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
       -f https://data.pyg.org/whl/torch-2.5.0+cu121.html 2>/dev/null; true

# Verify core rasterizer build
RUN python3 -c "from diff_gaussian_rasterization import GaussianRasterizationSettings; print('RaDe-GS OK')"

# --- Layer 6: Application code ---
COPY scripts/ /app/scripts/

# --- Layer 7: Test dependencies & test files ---
COPY tests/ /app/tests/
RUN pip install --no-cache-dir pytest pytest-asyncio

# Allow git to read /app/.git even though the bind-mounted dir is owned by the host UID
RUN git config --system --add safe.directory /app

# Add repos to Python path
ENV PYTHONPATH="/app:/opt/GaussianWrapping:/opt/GaussianWrapping/submodules/Depth-Anything-V2:/opt/sam2:${PYTHONPATH}"

EXPOSE 7860

ENTRYPOINT ["python3", "-m", "uvicorn"]
CMD ["scripts.dashboard.app:app", "--host", "0.0.0.0", "--port", "7860"]
