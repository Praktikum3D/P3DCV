# Runtime environment for P3DCV (E-RayZer fine-tuning) on the CAMP cluster.
# The image only contains the environment. Code, data and checkpoints are
# mounted from /mnt at run time, so code changes never require a rebuild.
#
# E-RayZer pins Python 3.10, torch 2.3.0 (CUDA 12.1) and xformers 0.0.26.post1.
FROM docker.io/pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# gsplat fork used by E-RayZer (adds gradients w.r.t. intrinsics), pinned to
# the commit of branch v1.4.0-working. The CUDA extension is compiled here.
# No GPU is visible during the build, so the target architectures are listed
# explicitly: TITAN X (5.2), TITAN Xp (6.1), RTX 20xx / TITAN RTX (7.5),
# RTX 3090 / A5000 (8.6), RTX 4090 (8.9), H200 (9.0).
ENV TORCH_CUDA_ARCH_LIST="5.2 6.1 7.5 8.6 8.9 9.0"
RUN MAX_JOBS=4 pip install --no-build-isolation \
        "git+https://github.com/QitaoZhao/gsplat.git@daad91d9e667cf49bab815d872aa65a5cda0a77e"
