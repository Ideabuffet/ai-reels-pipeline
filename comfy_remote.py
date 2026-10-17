"""ComfyUI over the RunPod HTTP proxy (https://<pod>-8188.proxy.runpod.net): upload, queue, wait, download.

No SSH needed. Every status poll also hits /pipeline/status, which is the watchdog heartbeat.
"""
import os
import time
import uuid

import requests


class ComfyError(RuntimeError):
    pass


class Comfy:
    def __init__(self, base):
        self.base = base.rstrip("/")
        self.client_id = str(uuid.uuid4())
        self.s = requests.Session()

    def get(self, path, **kw):
        r = self.s.get(self.base + path, timeout=kw.pop("timeout", 60), **kw)
        r.raise_for_status()
        return r

    def post(self, path, **kw):
        r = self.s.post(self.base + path, timeout=kw.pop("timeout", 60), **kw)
        if r.status_code >= 400:
            raise ComfyError(f"POST {path}: HTTP {r.status_code} {r.text[:1500]}")
        return r

    def status(self):
        """Pipeline status (models, uptime) or None while ComfyUI / the proxy is not up yet."""
        try:
            r = self.s.get(self.base + "/pipeline/status", timeout=20)
            return r.json() if r.status_code == 200 else None
        except (requests.RequestException, ValueError):
            return None

    def system_stats(self):
        return self.get("/system_stats").json()

    def upload(self, path, name):
        for attempt in range(3):
            try:
                with open(path, "rb") as f:
                    r = self.post("/upload/image", files={"image": (name, f)},
                                  data={"overwrite": "true", "type": "input"}, timeout=300)
                return r.json()["name"]
            except (requests.RequestException, ComfyError):
                if attempt == 2:
                    raise
                time.sleep(5)

    def submit(self, graph):
        return self.post("/prompt", json={"prompt": graph, "client_id": self.client_id}).json()["prompt_id"]

    def free(self):
        """Unload models and drop caches between phases instead of restarting ComfyUI."""
        self.post("/free", json={"unload_models": True, "free_memory": True})

    def history(self, pid):
        return self.get(f"/history/{pid}").json().get(pid)

    def wait(self, pids, on_done=None, poll=10, timeout=4 * 3600):
        """Wait for queued prompts; returns {pid: result}. on_done(pid, result) is called as each finishes."""
        left, out, t0 = list(pids), {}, time.time()
        errors = 0
        while left:
            if time.time() - t0 > timeout:
                raise ComfyError(f"timeout waiting for {len(left)} prompts")
            time.sleep(poll)
            self.status()  # heartbeat
            for pid in list(left):
                try:
                    h = self.history(pid)
                    errors = 0
                except requests.RequestException:
                    errors += 1
                    if errors > 30:  # ~5 min without the proxy
                        raise
                    break
                if not h or h.get("status", {}).get("completed") is None:
                    continue
                res = parse_result(h)
                out[pid] = res
                left.remove(pid)
                if on_done:
                    on_done(pid, res)
        return out

    def download(self, fileinfo, dest):
        params = {"filename": fileinfo["filename"], "subfolder": fileinfo.get("subfolder", ""),
                  "type": fileinfo.get("type", "output")}
        tmp = dest + ".part"
        for attempt in range(3):
            try:
                with self.s.get(self.base + "/view", params=params, stream=True, timeout=300) as r:
                    r.raise_for_status()
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(1 << 20):
                            f.write(chunk)
                os.replace(tmp, dest)
                return dest
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(5)


def parse_result(h):
    st = h.get("status", {})
    msgs = {m[0]: m[1] for m in st.get("messages", [])}
    t_start = msgs.get("execution_start", {}).get("timestamp")
    t_end = (msgs.get("execution_success") or msgs.get("execution_error") or {}).get("timestamp")
    err = msgs.get("execution_error")
    files = [g for o in h.get("outputs", {}).values() for g in o.get("gifs", [])]
    return {"ok": st.get("status_str") == "success", "files": files,
            "exec_s": round((t_end - t_start) / 1000, 1) if t_start and t_end else None,
            "error": (f"{err.get('node_type')}: {err.get('exception_message')}" if err else None)}
