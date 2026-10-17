# AI-persona reels on RunPod: ComfyUI + Kijai WanVideoWrapper (Wan 2.2 I2V, InfiniteTalk) + RTX VSR.
# Models are NOT baked in (62 GB): boot.sh downloads them with aria2 or finds them on a Network Volume.
#
#   docker build -t ghcr.io/<owner>/ai-reels-comfy:<tag> pipeline/docker
ARG BASE_IMAGE=runpod/comfyui:1.3.3-comfyuiv0.30.0-cuda12.8
FROM ${BASE_IMAGE}

ARG WANVIDEO_SHA=088128b22424
ARG VHS_SHA=4d907bee61e9
ARG KJNODES_SHA=

RUN apt-get update && apt-get install -y --no-install-recommends aria2 \
    && rm -rf /var/lib/apt/lists/*

COPY setup_nodes.sh patch_wav2vec.py /opt/pipeline/
RUN WANVIDEO_SHA=$WANVIDEO_SHA VHS_SHA=$VHS_SHA KJNODES_SHA=$KJNODES_SHA \
    bash /opt/pipeline/setup_nodes.sh /opt/comfyui-baked

# ComfyUI/models -> /models -> $MODELS_ROOT (chosen at boot); the stock subfolders become a skeleton.
# start.sh copies /opt/comfyui-baked to /workspace with `cp -r`, which keeps the symlink.
RUN mv /opt/comfyui-baked/models /opt/pipeline/models-skel \
    && ln -s /models /opt/comfyui-baked/models

COPY pipeline_status /opt/comfyui-baked/custom_nodes/pipeline_status
COPY boot.sh fetch_models.sh watchdog.sh models.txt /opt/pipeline/
RUN chmod +x /opt/pipeline/*.sh

ENTRYPOINT ["/opt/pipeline/boot.sh"]
