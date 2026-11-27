#!/bin/bash
#SBATCH --job-name=test-gsplat-kb
#SBATCH --partition=12g
# gsplat needs an RTX 20xx or newer. ccc is the CUDA compute capability of the
# node times ten; ccc:75 keeps the job off the TITAN X / TITAN Xp nodes.
#SBATCH --gres=gpu:1,ccc:75
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/home/%u/runs/%x-%j.out
#SBATCH --error=/mnt/home/%u/runs/%x-%j.err

# Builds the vendored gsplat fork with the Kannala-Brandt camera
# (third_party/gsplat) and runs tests/gsplat_kb on one GPU.
#
# Inside one container start (its root file system is thrown away afterwards):
#   1. saves reference outputs of the pinhole / fisheye paths with the
#      UNMODIFIED gsplat that is installed in the image,
#   2. builds the vendored fork as a wheel for this node's GPU architecture,
#      or reuses the cached wheel if the sources did not change, and installs it,
#   3. runs pytest tests/gsplat_kb and prints a short summary.
#
# The wheel cache lives in $WHEEL_CACHE (default
# /mnt/home/$USER/.cache/p3dcv-gsplat-kb). Its key is a hash of the gsplat
# sources plus the GPU architecture and the torch version, so a rerun after a
# test-only change skips the ~10 minute CUDA build. Logs and the per-test
# report go to $OUT_DIR (default /mnt/home/$USER/runs/test-gsplat-kb-<jobid>).
#
# The container image itself is not modified.
#
# Example:
#   sbatch slurm/test_gsplat_kb.sh
set -euo pipefail
source "${REPO_DIR:-/mnt/home/$USER/code/P3DCV}/slurm/common.sh"

if [[ "${1:-}" != "--inside" ]]; then
    OUT_DIR="${OUT_DIR:-/mnt/home/$USER/runs/test-gsplat-kb-${SLURM_JOB_ID:-manual}}"
    WHEEL_CACHE="${WHEEL_CACHE:-/mnt/home/$USER/.cache/p3dcv-gsplat-kb}"
    mkdir -p "$OUT_DIR" "$WHEEL_CACHE"
    in_container bash -c "env REPO_DIR='$REPO_DIR' OUT_DIR='$OUT_DIR' WHEEL_CACHE='$WHEEL_CACHE' \
        bash '$REPO_DIR/slurm/test_gsplat_kb.sh' --inside"
    exit $?
fi

# ---------------------------------------------------------------- in container
cd "$REPO_DIR"
export GSPLAT_KB_REPORT="$OUT_DIR/report.jsonl"
export GSPLAT_KB_REGRESSION_REF="$OUT_DIR/regression_reference.pt"
rm -f "$GSPLAT_KB_REPORT"
nvidia-smi -L

echo "== 1/3 reference outputs with the gsplat of the image"
python -c "import gsplat; print('installed gsplat:', gsplat.__file__)"
python tests/gsplat_kb/regression_reference.py --save "$GSPLAT_KB_REGRESSION_REF"

echo "== 2/3 build / install third_party/gsplat"
ARCH="$(python -c "import torch; print('%d.%d' % torch.cuda.get_device_capability())")"
TORCH_VERSION="$(python -c "import torch; print(torch.__version__)")"
SRC_HASH="$(cd third_party/gsplat && find setup.py gsplat -type f \
    \( -name '*.py' -o -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.cpp' \
       -o -name '*.hpp' -o -name '*.inl' \) -not -path '*/__pycache__/*' -print0 \
    | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-16)"
WHEEL_DIR="$WHEEL_CACHE/${SRC_HASH}-sm${ARCH/./}-torch${TORCH_VERSION}"
if ls "$WHEEL_DIR"/gsplat-*.whl >/dev/null 2>&1; then
    echo "reusing cached wheel $WHEEL_DIR"
else
    echo "building wheel for sm_${ARCH/./} into $WHEEL_DIR"
    rm -rf "$WHEEL_DIR.tmp"
    mkdir -p "$WHEEL_DIR.tmp"
    TORCH_CUDA_ARCH_LIST="$ARCH" MAX_JOBS=4 \
        pip wheel --no-build-isolation --no-deps -w "$WHEEL_DIR.tmp" third_party/gsplat
    mv "$WHEEL_DIR.tmp" "$WHEEL_DIR"
fi
pip install --no-deps --force-reinstall "$WHEEL_DIR"/gsplat-*.whl
python -c "
import gsplat, gsplat.cuda._wrapper as w
from gsplat.cuda._backend import _C
assert hasattr(_C.CameraModelType, 'KANNALA_BRANDT'), 'the installed gsplat is not the modified fork'
print('testing gsplat:', gsplat.__file__)
"

echo "== 3/3 pytest tests/gsplat_kb"
set +e
python -m pytest tests/gsplat_kb -v -rs -p no:cacheprovider 2>&1 | tee "$OUT_DIR/pytest.log"
STATUS=${PIPESTATUS[0]}
set -e

echo
echo "== summary (full report: $GSPLAT_KB_REPORT)"
python - "$GSPLAT_KB_REPORT" <<'EOF'
import json, sys
try:
    lines = [json.loads(l) for l in open(sys.argv[1])]
except FileNotFoundError:
    lines = []
for r in lines:
    name = r.pop("test")
    if name.startswith("finite_differences"):
        rel = {k[4:]: round(v, 4) for k, v in r.items() if k.startswith("rel_")}
        print(f"{name}: relative error analytic vs FD {rel}")
    elif name == "recovery":
        h = r["history"]
        print(f"{name}: start {h[0]} -> end {h[-1]} (truth {r['truth']})")
    else:
        short = {k: (round(v, 7) if isinstance(v, float) else v) for k, v in r.items()}
        print(f"{name}: {short}")
EOF
tail -n 1 "$OUT_DIR/pytest.log"
exit "$STATUS"
