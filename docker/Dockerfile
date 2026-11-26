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
# explicitly: RTX 20xx / TITAN RTX / RTX 5000 (7.5), RTX 3090 / A5000 (8.6),
# RTX 4090 (8.9), H200 (9.0). gsplat does not compile for anything older
# (it needs cooperative_groups::labeled_partition), so the image cannot
# render on the TITAN X / TITAN Xp nodes dortmund, jena and ulm.
# GSPLAT_REPO and GSPLAT_REF can point to another gsplat, for example the 3DGEER
# extension https://github.com/Praktikum3D/3dgeer.git at the head of branch
# gsplat-geer (f15cee225206944e38acdd0ab4a7ab4e4aae78a1). That branch lists glm
# as a submodule without committing it, so glm is cloned at the commit gsplat
# 1.5.3 pins when it is missing.
ARG GSPLAT_REPO=https://github.com/QitaoZhao/gsplat.git
ARG GSPLAT_REF=daad91d9e667cf49bab815d872aa65a5cda0a77e
ENV TORCH_CUDA_ARCH_LIST="7.5 8.6 8.9 9.0"
RUN git clone "$GSPLAT_REPO" /tmp/gsplat && cd /tmp/gsplat && git checkout "$GSPLAT_REF" \
    && git submodule update --init --recursive \
    && GLM=gsplat/cuda/csrc/third_party/glm \
    && if [ ! -f "$GLM/glm/glm.hpp" ]; then rm -rf "$GLM" && git clone https://github.com/g-truc/glm.git "$GLM" \
        && git -C "$GLM" checkout 33b4a621a697a305bc3a7610d290677b96beb181; fi \
    && MAX_JOBS=4 pip install --no-build-isolation . && cd / && rm -rf /tmp/gsplat
