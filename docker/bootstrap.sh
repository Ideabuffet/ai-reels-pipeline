#!/bin/bash
# No-registry mode: turns the STOCK runpod/comfyui image into our image at pod start — the same steps
# as the Dockerfile, run at boot (+3–5 min per pod). Used as the pod entrypoint by `run_reel.py --bootstrap`:
#   bash -c 'curl -fsSL https://raw.githubusercontent.com/<repo>/<ref>/docker/bootstrap.sh | bash -s <repo> <ref>'
set -euo pipefail
REPO=${1:-Ideabuffet/ai-reels-pipeline}
REF=${2:-main}
mkdir -p /tmp/pipeline
exec > >(tee -a /tmp/pipeline/bootstrap.log) 2>&1
echo "bootstrap $REPO@$REF $(date -u +%T)"
apt-get update -qq && apt-get install -y -qq --no-install-recommends aria2 > /dev/null
git clone -q "https://github.com/$REPO" /opt/pipeline-src
git -C /opt/pipeline-src checkout -q "$REF"
D=/opt/pipeline-src/docker
mkdir -p /opt/pipeline
cp "$D"/setup_nodes.sh "$D"/patch_wav2vec.py "$D"/boot.sh "$D"/fetch_models.sh "$D"/watchdog.sh "$D"/models.txt /opt/pipeline/
chmod +x /opt/pipeline/*.sh
# models download can start right away, in parallel with the node install
if [ ! -L /opt/comfyui-baked/models ]; then
  mv /opt/comfyui-baked/models /opt/pipeline/models-skel
  ln -s /models /opt/comfyui-baked/models
fi
cp -r "$D/pipeline_status" /opt/comfyui-baked/custom_nodes/pipeline_status
PIPELINE_PREPARE_ONLY=1 /opt/pipeline/boot.sh
bash /opt/pipeline/setup_nodes.sh /opt/comfyui-baked
echo "bootstrap done $(date -u +%T)"
exec /opt/pipeline/boot.sh
