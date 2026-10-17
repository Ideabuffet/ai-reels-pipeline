"""ComfyUI route for the reel orchestrator: GET /pipeline/status.

Reports model download progress (by stat-ing the files from models.txt) and pod uptime, and doubles
as the heartbeat for watchdog.sh, so the orchestrator needs nothing but the ComfyUI HTTP proxy.
"""
import os
import time

from aiohttp import web
from server import PromptServer

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

MODELS_TXT = "/opt/pipeline/models.txt"
P = "/tmp/pipeline"


def _read(name, default=""):
    try:
        return open(os.path.join(P, name)).read().strip()
    except OSError:
        return default


@PromptServer.instance.routes.get("/pipeline/status")
async def pipeline_status(request):
    os.makedirs(P, exist_ok=True)
    with open(os.path.join(P, "heartbeat"), "a"):
        os.utime(os.path.join(P, "heartbeat"))
    root = _read("models_root", "/models")
    total = done = 0
    missing = []
    for line in open(MODELS_TXT):
        parts = line.split()
        if len(parts) != 3 or line.startswith("#"):
            continue
        _, rel, size = parts
        size = int(size)
        total += size
        f = os.path.join(root, rel)
        st = os.stat(f) if os.path.exists(f) else None
        have = st.st_size if st else 0
        partial = os.path.exists(f + ".aria2")
        # aria2 with several connections extends the file to full size at once (sparse), so while the
        # control file exists count the blocks actually written, not the apparent size
        done += min(st.st_blocks * 512 if partial else have, size) if st else 0
        if have != size or partial:
            missing.append(os.path.basename(rel))
    boot = _read("boot.start")
    uptime = time.time() - float(boot) if boot else None  # /proc/uptime is the host's, not the pod's
    return web.json_response({
        "models_root": root, "fetch_state": _read("fetch.state", "not started"),
        "bytes_total": total, "bytes_done": done, "missing": missing, "ready": not missing,
        "uptime_s": round(uptime) if uptime else None, "time": time.time(),
    })
