"""Thin RunPod client: REST v1 for pods / network volumes, GraphQL for prices, stock and balance.

The key is read from RUNPOD_API_KEY and only ever goes into the Authorization header.
"""
import os
import time

import requests

REST = "https://rest.runpod.io/v1"
GQL = "https://api.runpod.io/graphql"


class RunPodError(RuntimeError):
    pass


def _headers():
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise RunPodError("RUNPOD_API_KEY is not set (put it in ~/.zshenv)")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _rest(method, path, body=None, ok=(200, 201, 204)):
    for attempt in range(4):
        try:
            r = requests.request(method, REST + path, headers=_headers(), json=body, timeout=60)
        except requests.RequestException as e:
            if attempt == 3:
                raise RunPodError(f"{method} {path}: {e}") from e
            time.sleep(3 * (attempt + 1))
            continue
        if r.status_code in ok:
            return r.json() if r.content else {}
        if r.status_code >= 500 and attempt < 3:
            time.sleep(3 * (attempt + 1))
            continue
        raise RunPodError(f"{method} {path}: HTTP {r.status_code} {r.text[:400]}")


def gql(query):
    r = requests.post(GQL, headers=_headers(), json={"query": query}, timeout=60)
    r.raise_for_status()
    d = r.json()
    if d.get("errors"):
        raise RunPodError(str(d["errors"])[:400])
    return d["data"]


def balance():
    return gql("query { myself { clientBalance currentSpendPerHr } }")["myself"]


def gpu_quote(gpu_id, cloud):
    """Price ($/h) and stock for one GPU type in COMMUNITY or SECURE cloud."""
    secure = "true" if cloud == "SECURE" else "false"
    d = gql(f'query {{ gpuTypes(input:{{id:"{gpu_id}"}}) {{ id displayName memoryInGb communityPrice securePrice '
            f'lowestPrice(input:{{gpuCount:1, secureCloud:{secure}}}) {{ stockStatus uninterruptablePrice }} }} }}')
    g = (d.get("gpuTypes") or [None])[0]
    if not g:
        return None
    lp = g.get("lowestPrice") or {}
    price = lp.get("uninterruptablePrice") or (g["securePrice"] if cloud == "SECURE" else g["communityPrice"])
    return {"gpu": gpu_id, "name": g["displayName"], "cloud": cloud, "vram_gb": g["memoryInGb"],
            "price": price, "stock": lp.get("stockStatus")}


def create_pod(body):
    return _rest("POST", "/pods", body)


def get_pod(pod_id):
    return _rest("GET", f"/pods/{pod_id}")


def list_pods():
    return _rest("GET", "/pods")


def stop_pod(pod_id):
    return _rest("POST", f"/pods/{pod_id}/stop")


def terminate_pod(pod_id):
    return _rest("DELETE", f"/pods/{pod_id}")


def create_network_volume(name, size_gb, data_center):
    return _rest("POST", "/networkvolumes", {"name": name, "size": size_gb, "dataCenterId": data_center})


def get_network_volume(volume_id):
    return _rest("GET", f"/networkvolumes/{volume_id}")


def pod_billing(pod_id, start_iso):
    """Billed amount for a pod since start_iso (RunPod aggregates with some delay)."""
    try:
        recs = requests.get(f"{REST}/billing/pods", headers=_headers(), timeout=60,
                            params={"podId": pod_id, "startTime": start_iso, "bucketSize": "hour"}).json()
        return round(sum(float(r.get("amount") or 0) for r in recs), 4)
    except Exception:
        return None
