#!/bin/bash
# Image entry point: point ComfyUI's models dir at the right storage, start the model download and the
# self-stop watchdog in the background, then hand over to the stock runpod/comfyui /start.sh
# (SSH, ComfyUI on :8188). ComfyUI starts right away; the orchestrator waits for /pipeline/status.
#
# Models live in $MODELS_ROOT:
#   Network Volume mounted at /runpod-volume -> /runpod-volume/models (downloaded once, kept)
#   otherwise                                -> /workspace/models (pod disk, downloaded on every pod)
mkdir -p /tmp/pipeline
if [ -z "${MODELS_ROOT:-}" ]; then
  if [ -d /runpod-volume ]; then MODELS_ROOT=/runpod-volume/models; else MODELS_ROOT=/workspace/models; fi
fi
export MODELS_ROOT
mkdir -p "$MODELS_ROOT"
cp -rn /opt/pipeline/models-skel/. "$MODELS_ROOT/" 2>/dev/null || true
ln -sfn "$MODELS_ROOT" /models   # /opt/comfyui-baked/models -> /models (set in the Dockerfile)
echo "$MODELS_ROOT" > /tmp/pipeline/models_root

if [ "${PIPELINE_FETCH_MODELS:-1}" != 0 ]; then
  setsid nohup /opt/pipeline/fetch_models.sh > /tmp/pipeline/fetch.log 2>&1 < /dev/null &
fi
setsid nohup /opt/pipeline/watchdog.sh > /tmp/pipeline/watchdog.log 2>&1 < /dev/null &

exec /start.sh "$@"
