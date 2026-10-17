#!/usr/bin/env python3
"""Reel spec -> finished vertical mp4, on your own GPU in RunPod. One command, run from the Mac.

  python3 run_reel.py reels/nina_shchi.json            # quote, ask «да», generate, stop the pod, montage
  python3 run_reel.py reels/nina_shchi.json --quote    # only show GPU prices / stock
  python3 run_reel.py reels/nina_shchi.json --tts      # voice the script with ElevenLabs first
  python3 run_reel.py cleanup                          # terminate leftover reel-* pods

Steps: [TTS] -> create pod -> boot -> models (aria2 / Network Volume) -> upload -> Wan 2.2 clips (batch)
-> free memory -> InfiniteTalk parts (batch) -> download -> terminate pod (always, also on errors / Ctrl-C)
-> montage. Clips are upscaled with RTX VSR inside each graph. Prints time and cost per step.
"""
import argparse, contextlib, datetime as dt, json, os, signal, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import comfy_remote, graphs, montage, reel_spec, runpod_api as rp, tts  # noqa: E402

DEFAULT_IMAGE = os.environ.get("REEL_IMAGE", "ghcr.io/ideabuffet/ai-reels-comfy:latest")
BASE_IMAGE = "runpod/comfyui:1.3.3-comfyuiv0.30.0-cuda12.8"
REPO = "Ideabuffet/ai-reels-pipeline"
# Cheapest first. Speed on these jobs: A40 ≈ 3090 (17.9 vs 18.4 s/step), 4090 ~1.6-2x faster.
DEFAULT_GPUS = ["NVIDIA GeForce RTX 3090:COMMUNITY", "NVIDIA GeForce RTX 4090:COMMUNITY",
                "NVIDIA RTX A6000:COMMUNITY", "NVIDIA A40:SECURE", "NVIDIA RTX A6000:SECURE",
                "NVIDIA GeForce RTX 4090:SECURE", "NVIDIA L40S:SECURE"]
DISK_USD_PER_GB_H = 0.10 / 730   # pod volume / container disk while running, $0.10 per GB-month
POD_PREFIX = "reel-"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Steps:
    """Wall time per step; pod-billed steps are priced at the pod rate."""

    def __init__(self):
        self.rows, self.rate = [], 0.0

    @contextlib.contextmanager
    def step(self, name, billed=True):
        t0 = time.time()
        log(f"▶ {name}")
        try:
            yield
        finally:
            self.add(name, time.time() - t0, billed)

    def add(self, name, seconds, billed=True, usd=None):
        self.rows.append({"step": name, "s": round(seconds, 1), "billed": billed,
                          "usd": usd if usd is not None else (seconds / 3600 * self.rate if billed else 0.0)})

    def table(self, pod_total_s=None):
        lines = [f"{'шаг':<44}{'время':>10}{'$':>9}", "-" * 63]
        for r in self.rows:
            lines.append(f"{r['step'][:43]:<44}{fmt_s(r['s']):>10}{r['usd']:>9.3f}")
        billed = sum(r["s"] for r in self.rows if r["billed"])
        if pod_total_s and pod_total_s > billed + 1:
            other = pod_total_s - billed
            lines.append(f"{'прочее время пода (между шагами)':<44}{fmt_s(other):>10}{other / 3600 * self.rate:>9.3f}")
        total_usd = sum(r["usd"] for r in self.rows) + (max(0, (pod_total_s or 0) - billed) / 3600 * self.rate)
        total_s = sum(r["s"] for r in self.rows) + max(0, (pod_total_s or 0) - billed)
        lines += ["-" * 63, f"{'ИТОГО':<44}{fmt_s(total_s):>10}{total_usd:>9.3f}"]
        return "\n".join(lines), total_usd


def fmt_s(s):
    s = int(round(s))
    return f"{s // 60}:{s % 60:02d}" if s < 3600 else f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def parse_gpus(items):
    out = []
    for it in items:
        name, _, cloud = it.rpartition(":")
        if cloud not in ("COMMUNITY", "SECURE"):
            name, cloud = it, "COMMUNITY"
        out.append((name, cloud))
    return out


def quote(gpus, max_price, network_volume):
    rows = []
    for name, cloud in gpus:
        if network_volume and cloud != "SECURE":
            continue  # Network Volumes attach to Secure Cloud pods only
        try:
            q = rp.gpu_quote(name, cloud)
        except Exception as e:
            log(f"quote {name} {cloud}: {e}")
            continue
        if q:
            rows.append(q)
    for q in rows:
        ok = q["stock"] and q["price"] and q["price"] <= max_price
        q["ok"] = bool(ok)
        print(f"  {'✓' if ok else '·'} {q['name']:<14} {q['cloud']:<9} {q['vram_gb']:>3} GB  "
              f"${q['price'] or 0:.2f}/ч  наличие: {q['stock'] or 'нет'}")
    return [q for q in rows if q["ok"]]


def confirm(text):
    print(text)
    try:
        ans = input("Создать под? [да/нет] ").strip().lower()
    except EOFError:
        ans = ""
    return ans in ("да", "д", "yes", "y")


def pod_body(a, gpu, cloud, name):
    body = {
        "name": name, "imageName": a.image, "gpuTypeIds": [gpu], "gpuCount": 1, "cloudType": cloud,
        "containerDiskInGb": 20, "volumeInGb": a.volume_gb, "volumeMountPath": "/workspace",
        "ports": ["8188/http", "22/tcp"], "minRAMPerGPU": a.min_ram, "minVCPUPerGPU": 4,
        "allowedCudaVersions": ["12.8", "12.9", "13.0", "13.1", "13.2"],
        "env": {"PIPELINE_HEARTBEAT_MIN": str(a.heartbeat_min), "PIPELINE_MAX_MIN": str(a.max_min)},
    }
    if a.bootstrap:  # stock image, docker/bootstrap.sh from the repo does the Dockerfile steps at boot
        body["imageName"] = BASE_IMAGE
        body["dockerEntrypoint"] = ["bash", "-c", f"curl -fsSL https://raw.githubusercontent.com/{REPO}/{a.bootstrap}"
                                    f"/docker/bootstrap.sh | bash -s {REPO} {a.bootstrap}"]
    if a.min_download_mbps:
        body["minDownloadMbps"] = a.min_download_mbps
    if a.network_volume:
        nv = rp.get_network_volume(a.network_volume)
        body.update({"networkVolumeId": a.network_volume, "volumeMountPath": "/runpod-volume",
                     "volumeInGb": 0, "containerDiskInGb": 40, "dataCenterIds": [nv["dataCenterId"]],
                     "cloudType": "SECURE"})
    return body


def create_pod(a, cands, name):
    errors = []
    for q in cands:
        try:
            pod = rp.create_pod(pod_body(a, q["gpu"], q["cloud"], name))
            log(f"pod {pod['id']}: {q['name']} {q['cloud']} ${pod.get('costPerHr') or q['price']}/ч")
            return pod, q
        except rp.RunPodError as e:
            errors.append(f"{q['name']} {q['cloud']}: {str(e)[:160]}")
            log(f"нет машины: {errors[-1]}")
    raise RuntimeError("не удалось создать под ни на одной карте:\n  " + "\n  ".join(errors))


def wait_boot(comfy, pod_id, timeout=25 * 60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = comfy.status()
        if st:
            return st
        pod = rp.get_pod(pod_id)
        if pod.get("desiredStatus") not in ("RUNNING", None):
            raise RuntimeError(f"pod is {pod.get('desiredStatus')} while booting")
        time.sleep(15)
    raise TimeoutError("ComfyUI did not come up in 25 min (image pull / start failed?)")


def wait_models(comfy, timeout=120 * 60):
    t0, last, last_b = time.time(), 0, None
    while time.time() - t0 < timeout:
        st = comfy.status()
        if st:
            if st["ready"]:
                return st
            if st["fetch_state"].startswith("failed"):
                raise RuntimeError(f"model download {st['fetch_state']}")
            if time.time() - last > 60:
                b = st["bytes_done"]
                rate = f", {(b - last_b[0]) / (time.time() - last_b[1]) / 1e6:.0f} MB/s" if last_b else ""
                log(f"  модели: {b / 1e9:.1f} / {st['bytes_total'] / 1e9:.1f} GB{rate}")
                last, last_b = time.time(), (b, time.time())
        time.sleep(15)
    raise TimeoutError("models not ready in 120 min")


def run_phase(comfy, steps, jobs, clips_dir, label):
    """jobs: [(clip_id, graph)] queued at once; each result is downloaded as soon as it finishes."""
    if not jobs:
        return []
    pids = {comfy.submit(g): cid for cid, g in jobs}
    log(f"  в очереди {len(pids)} ({label})")
    failed = []

    def done(pid, res):
        cid = pids[pid]
        if res["ok"] and res["files"]:
            comfy.download(res["files"][-1], os.path.join(clips_dir, f"{cid}.mp4"))
            log(f"  ✓ {cid}: {res['exec_s']} с")
        else:
            failed.append(cid)
            log(f"  ✗ {cid}: {res['error']}")
        steps.add(f"{label}: {cid}", res["exec_s"] or 0)

    comfy.wait(list(pids), on_done=done)
    return failed


def cleanup(a):
    pods = [p for p in rp.list_pods() if (p.get("name") or "").startswith(POD_PREFIX)]
    if not pods:
        print("нет подов reel-*")
        return
    for p in pods:
        print(f"  {p['id']} {p['name']} {p.get('desiredStatus')} ${p.get('costPerHr')}/ч")
    if a.yes or confirm("Удалить (terminate) эти поды?"):
        for p in pods:
            rp.terminate_pod(p["id"])
            print("  terminated", p["id"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", help="reel spec JSON, or 'cleanup'")
    ap.add_argument("--yes", action="store_true", help="do not ask before creating the pod")
    ap.add_argument("--quote", action="store_true", help="only show GPU prices and exit")
    ap.add_argument("--gpu", action="append", help="'NAME:COMMUNITY|SECURE', repeatable, in priority order")
    ap.add_argument("--max-price", type=float, default=0.80, help="max $/h for the GPU (default 0.80)")
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--bootstrap", metavar="GIT_REF", nargs="?", const="main",
                    help="no custom image: stock runpod/comfyui + docker/bootstrap.sh at boot (+3-5 min)")
    ap.add_argument("--network-volume", help="Network Volume id with the models (Secure Cloud only)")
    ap.add_argument("--volume-gb", type=int, default=80, help="pod volume for models when no Network Volume")
    ap.add_argument("--min-ram", type=int, default=46, help="min RAM per GPU, GB (InfiniteTalk peaks ~43 GB)")
    ap.add_argument("--min-download-mbps", type=int, help="only hosts with at least this download speed")
    ap.add_argument("--heartbeat-min", type=int, default=20, help="pod stops itself after this long without us")
    ap.add_argument("--max-min", type=int, default=240, help="hard pod lifetime limit, min")
    ap.add_argument("--tts", action="store_true", help="voice the script with ElevenLabs (voice in the spec)")
    ap.add_argument("--music", help="music bed mp3 for the montage")
    ap.add_argument("--out-dir", help="run directory (default runs/<name>_<time>)")
    ap.add_argument("--resume", action="store_true", help="skip clips already in <out-dir>/clips")
    ap.add_argument("--keep-pod", action="store_true", help="stop instead of terminate (volume keeps billing!)")
    ap.add_argument("--no-montage", action="store_true")
    a = ap.parse_args()

    if a.spec == "cleanup":
        return cleanup(a)

    spec = reel_spec.load(a.spec)
    gen = spec.get("gen", {})
    W, H = gen.get("width", 480), gen.get("height", 832)
    up = tuple(gen["upscale"]) if gen.get("upscale") else None
    if a.music:
        spec["music"] = {"file": os.path.abspath(a.music), "db": -24}
    run_dir = os.path.abspath(a.out_dir or os.path.join(HERE, "runs", f"{spec['name']}_{time.strftime('%Y%m%d_%H%M%S')}"))
    clips_dir = os.path.join(run_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    steps = Steps()
    report = {"spec": os.path.abspath(a.spec), "run_dir": run_dir, "started": dt.datetime.now().isoformat()}

    # 1. voice-over (optional)
    audio_dir = None
    if a.tts:
        audio_dir = os.path.join(run_dir, "audio")
        with steps.step("озвучка ElevenLabs", billed=False):
            _, chars = tts.generate(spec, audio_dir, reel_spec.audio_keys(spec), reel_spec.talk_audio_keys(spec))
        steps.rows[-1]["usd"] = chars * tts.USD_PER_CHAR
        steps.rows[-1]["step"] += f" ({chars} симв.)"
        report["tts_chars"] = chars
    audio = reel_spec.audio_paths(spec, audio_dir)
    reel_spec.check_files(spec, audio)

    todo = [c for c in spec["clips"] if not (a.resume and os.path.exists(os.path.join(clips_dir, f"{c['id']}.mp4")))]
    log(f"клипов к генерации: {len(todo)} из {len(spec['clips'])}")

    pod_id, pod_t0, pod_t1, err = None, None, None, None
    if todo:
        print("Цены и наличие:")
        cands = quote(parse_gpus(a.gpu or DEFAULT_GPUS), a.max_price, a.network_volume)
        if a.quote:
            return
        if not cands:
            sys.exit("нет доступных карт в пределах --max-price")
        first = cands[0]
        text = (f"\nПод: {first['name']} {first['cloud']} — ${first['price']:.2f}/ч"
                + (f" (если занят — следующий из списка, максимум ${max(q['price'] for q in cands):.2f}/ч)" if len(cands) > 1 else "")
                + f"\nОценка: ~60–90 мин работы пода ≈ ${first['price'] * 1.5:.2f}; под удаляется в конце, "
                  f"сам останавливается через {a.heartbeat_min} мин без связи и через {a.max_min} мин в любом случае.")
        if not a.yes and not confirm(text):
            sys.exit("отменено")

        def on_signal(signum, _frame):
            raise KeyboardInterrupt(f"signal {signum}")
        for s in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(s, on_signal)

        try:
            bal0 = rp.balance()
            with steps.step("создание пода"):
                pod, q = create_pod(a, cands, f"{POD_PREFIX}{spec['name']}")
                pod_id, pod_t0 = pod["id"], time.time()
                json.dump({"pod_id": pod_id}, open(os.path.join(run_dir, "pod.json"), "w"))
                rate = float(pod.get("costPerHr") or q["price"])
                disk = (a.volume_gb + 20) if not a.network_volume else 40
                steps.rate = rate + disk * DISK_USD_PER_GB_H
            report.update(pod_id=pod_id, gpu=q["name"], cloud=q["cloud"], rate_per_h=round(steps.rate, 4),
                          start_iso=dt.datetime.now(dt.timezone.utc).isoformat())
            comfy = comfy_remote.Comfy(f"https://{pod_id}-8188.proxy.runpod.net")
            with steps.step("загрузка образа и старт ComfyUI"):
                wait_boot(comfy, pod_id)
            with steps.step("модели (aria2 / Network Volume)"):
                st = wait_models(comfy)
                log(f"  модели готовы: {st['bytes_total'] / 1e9:.1f} GB в {st['models_root']}")
            vram = comfy.system_stats()["devices"][0]["vram_total"] / 2**30
            swap = 0 if vram >= 40 else 20
            log(f"  VRAM {vram:.0f} GB → block swap {swap}")
            report["vram_gb"] = round(vram, 1)

            with steps.step("загрузка входов на под"):
                up_names = {}
                for c in todo:
                    up_names[c["_frame"]] = comfy.upload(c["_frame"], f"{spec['name']}_{os.path.basename(c['_frame'])}")
                for c in todo:
                    if c["kind"] == "talk":
                        p = audio[c["audio"]]
                        if p not in up_names:
                            up_names[p] = comfy.upload(p, f"{spec['name']}_{os.path.basename(p)}")

            prefix = lambda c: f"{spec['name']}/{c['id']}"
            acts = [(c["id"], graphs.wan22(up_names[c["_frame"]], c["prompt"], prefix(c), W, H,
                                           c.get("frames", gen.get("action_frames", 81)),
                                           c.get("seed", gen.get("action_seed", 42)), upscale=up))
                    for c in todo if c["kind"] == "action"]
            talks = [(c["id"], graphs.talk(up_names[c["_frame"]], up_names[audio[c["audio"]]], prefix(c), W, H,
                                           c.get("prompt", graphs.TALK_PROMPT), c.get("seed", gen.get("talk_seed", 2)),
                                           swap, upscale=up))
                     for c in todo if c["kind"] == "talk"]
            failed = run_phase(comfy, steps, acts, clips_dir, "Wan 2.2")
            if acts and talks:
                comfy.free()
                time.sleep(5)
            failed += run_phase(comfy, steps, talks, clips_dir, "InfiniteTalk")
            if failed:
                raise RuntimeError(f"не получились клипы: {', '.join(failed)} (перезапуск с --resume --out-dir {run_dir})")
        except BaseException as e:  # noqa: BLE001 — the pod must go down whatever happened
            err = e
            log(f"ОШИБКА: {type(e).__name__}: {e}")
        finally:
            if pod_id:
                signal.signal(signal.SIGINT, signal.SIG_IGN)  # a second Ctrl-C must not abort the teardown
                with steps.step("остановка пода", billed=False):
                    for attempt in range(5):
                        try:
                            (rp.stop_pod if a.keep_pod else rp.terminate_pod)(pod_id)
                            break
                        except Exception as e:  # noqa: BLE001
                            log(f"  не удалось остановить ({e}), повтор")
                            time.sleep(10)
                    else:
                        log(f"!!! ПОД {pod_id} НЕ ОСТАНОВЛЕН — останови вручную: run_reel.py cleanup")
                    pod_t1 = time.time()
                log(f"под {pod_id} {'остановлен' if a.keep_pod else 'удалён'}")
                try:
                    bal1 = rp.balance()
                    report["balance_delta"] = round(bal0["clientBalance"] - bal1["clientBalance"], 4)
                except Exception:  # noqa: BLE001
                    pass
    if err:
        print(steps.table(pod_t1 - pod_t0 if pod_t0 and pod_t1 else None)[0])
        raise SystemExit(f"остановлено из-за ошибки: {err}")

    out = None
    if not a.no_montage:
        with steps.step("монтаж (Mac)", billed=False):
            out, length = montage.build(spec, clips_dir, audio, os.path.join(run_dir, f"{spec['name']}.mp4"),
                                        reel_spec.script(spec))
        log(f"ролик: {out} ({length} с)")

    table, total = steps.table(pod_t1 - pod_t0 if pod_t0 and pod_t1 else None)
    print("\n" + table)
    if report.get("balance_delta") is not None:
        print(f"Списание по балансу RunPod на момент окончания: ${report['balance_delta']:.3f} "
              f"(баланс обновляется с задержкой, точная сумма — в Billing)")
    report.update(steps=steps.rows, total_usd=round(total, 4), out=out,
                  pod_seconds=round(pod_t1 - pod_t0, 1) if pod_t0 and pod_t1 else None)
    json.dump(report, open(os.path.join(run_dir, "report.json"), "w"), ensure_ascii=False, indent=1)
    print(f"отчёт: {os.path.join(run_dir, 'report.json')}")


if __name__ == "__main__":
    main()
