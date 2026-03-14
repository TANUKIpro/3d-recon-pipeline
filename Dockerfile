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

# --- Layer 4: SAM2 (editable install with CUDA extension for post-processing) ---
RUN git clone https://github.com/facebookresearch/sam2.git /opt/sam2 \
    && cd /opt/sam2 \
    && SAM2_BUILD_CUDA=1 pip install --no-build-isolation -e .

# --- Layer 5a: gaussian-splatting CUDA extensions ---
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"

RUN git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git /opt/gaussian-splatting \
    && pip install --no-cache-dir --no-build-isolation \
        /opt/gaussian-splatting/submodules/diff-gaussian-rasterization \
        /opt/gaussian-splatting/submodules/simple-knn \
    && pip install --no-cache-dir --no-build-isolation \
        /opt/gaussian-splatting/submodules/fused-ssim 2>/dev/null || true

# --- Layer 5b: gs2mesh + DLNR stereo weights ---
RUN git clone https://github.com/yanivw12/gs2mesh.git /opt/gs2mesh \
    && rm -rf /opt/gs2mesh/third_party/gaussian-splatting \
    && ln -s /opt/gaussian-splatting /opt/gs2mesh/third_party/gaussian-splatting \
    && cd /opt/gs2mesh \
    && pip install --no-cache-dir -r requirements.txt 2>/dev/null; true

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
