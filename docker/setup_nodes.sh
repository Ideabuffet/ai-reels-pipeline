#!/bin/bash
# Installs the pinned custom nodes + Python packages into a ComfyUI tree and applies the compat fixes.
# Used by the Dockerfile (on /opt/comfyui-baked, copied to /workspace on first boot).
#
# usage: setup_nodes.sh [comfyui_dir]
set -euo pipefail
C=${1:-/opt/comfyui-baked}
N=$C/custom_nodes
HERE=$(cd "$(dirname "$0")" && pwd)
: "${WANVIDEO_SHA:=088128b22424}"   # kijai/ComfyUI-WanVideoWrapper, the commit used in the 17.10 test
: "${VHS_SHA:=4d907bee61e9}"        # Kosinkadink/ComfyUI-VideoHelperSuite
: "${KJNODES_SHA:=}"                # empty = keep the KJNodes baked into runpod/comfyui (RTX VSR worked with it)

clone() {  # url dir sha
  [ -d "$N/$2/.git" ] || git clone -q "$1" "$N/$2"
  git -C "$N/$2" fetch -q origin
  git -C "$N/$2" checkout -q "$3"
  echo "$2 @ $(git -C "$N/$2" rev-parse --short=12 HEAD)"
}
clone https://github.com/kijai/ComfyUI-WanVideoWrapper ComfyUI-WanVideoWrapper "$WANVIDEO_SHA"
clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite ComfyUI-VideoHelperSuite "$VHS_SHA"
if [ -n "$KJNODES_SHA" ]; then
  clone https://github.com/kijai/ComfyUI-KJNodes ComfyUI-KJNodes "$KJNODES_SHA"
else
  echo "ComfyUI-KJNodes @ $(git -C "$N/ComfyUI-KJNodes" rev-parse --short=12 HEAD 2>/dev/null || echo baked) (from base image)"
fi

# Keep the image's CUDA-specific torch stack: node requirements must not swap it for another build.
python3.12 - > /tmp/torch-constraints.txt <<'EOF'
import importlib.metadata as m
for p in ("torch", "torchvision", "torchaudio"):
    try:
        print(f"{p}=={m.version(p)}")
    except m.PackageNotFoundError:
        pass
EOF
cat /tmp/torch-constraints.txt

# huggingface_hub < 2.0: with 2.x ComfyUI does not start (found in the 17.10 test)
python3.12 -m pip install --no-cache-dir -c /tmp/torch-constraints.txt \
  -r "$N/ComfyUI-WanVideoWrapper/requirements.txt" \
  -r "$N/ComfyUI-VideoHelperSuite/requirements.txt" \
  "sageattention==1.0.6" nvidia-vfx "huggingface_hub>=1.5.0,<2.0"

python3.12 "$HERE/patch_wav2vec.py" "$N/ComfyUI-WanVideoWrapper/multitalk/wav2vec2.py"

python3.12 - <<'EOF'
import importlib.metadata as m
for p in ("torch", "transformers", "huggingface_hub", "sageattention", "gguf", "nvidia-vfx"):
    print(f"  {p} {m.version(p)}")  # raises if a package is missing
assert int(m.version("huggingface_hub").split(".")[0]) < 2, "huggingface_hub must stay < 2.0"
EOF
