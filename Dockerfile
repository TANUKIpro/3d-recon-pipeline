# im2pc-pipeline: Docker-based 3D reconstruction pipeline
# PyTorch (SAM2/Pi3X) + JAX (DiffCD) in a single container

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# --- Layer 1: System dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-distutils \
    git wget curl ffmpeg \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Layer 2: PyTorch (CUDA 12.1 wheels, matches base image) ---
RUN pip install --no-cache-dir \
    "torch>=2.5.0" torchvision \
    --index-url https://download.pytorch.org/whl/cu121

# --- Layer 3: Python dependencies (includes JAX via optax/flax) ---
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
    tyro optax flax \
    hydra-core iopath

# --- Layer 3b: Extra deps needed by DiffCD (wandb, scikit-image) ---
# Must come after numpy is installed (scikit-image build needs it)
RUN pip install --no-cache-dir wandb scikit-image

# --- Layer 4: SAM2 (editable install with CUDA extension for post-processing) ---
RUN git clone https://github.com/facebookresearch/sam2.git /opt/sam2 \
    && cd /opt/sam2 \
    && SAM2_BUILD_CUDA=1 pip install --no-build-isolation -e .

# --- Layer 5: Pi3 (sys.path usage, pinned for API compat with stage offloading) ---
ARG PI3_COMMIT=08d7288aaf4b0c08c8498bea7bafedc4672bb006
RUN git clone https://github.com/yyfz/Pi3.git /opt/pi3 \
    && cd /opt/pi3 \
    && git checkout ${PI3_COMMIT} \
    && pip install --no-cache-dir -r requirements.txt 2>/dev/null || true

# --- Layer 6: DiffCD (clone + patches) ---
RUN git clone https://github.com/Linusnie/diffcd.git /opt/diffcd \
    && cd /opt/diffcd \
    && sed -i 's/alpha: float = 100$/alpha: float = 100.0/' diffcd/methods.py \
    && find . -name "*.py" -exec sed -i 's/jax\.tree_map/jax.tree.map/g' {} + \
    && pip install --no-cache-dir -r requirements.txt 2>/dev/null || true

# --- Layer 7: Fix JAX CUDA (must be AFTER all pip installs that touch torch/jax) ---
# SAM2 pins nvidia-cudnn-cu12==9.1.0.70 (via torch), but jaxlib needs >=9.8.0.
# Upgrade CuDNN + install matching JAX CUDA plugin. torch works fine with higher CuDNN.
RUN JAX_V=$(python3 -c "import jax; print(jax.__version__)") \
    && pip install --no-cache-dir --no-deps \
    "jax-cuda12-plugin==${JAX_V}" "jax-cuda12-pjrt==${JAX_V}" \
    "nvidia-cudnn-cu12>=9.8.0"

# --- Layer 8: Application code ---
COPY scripts/ /app/scripts/

# Add repos to Python path
ENV PYTHONPATH="/opt/pi3:/opt/sam2:${PYTHONPATH}"

EXPOSE 7860

ENTRYPOINT ["python3", "-m", "uvicorn"]
CMD ["scripts.dashboard.app:app", "--host", "0.0.0.0", "--port", "7860"]
