"""Reel spec (JSON) loading: resolves asset paths and validates clips / layout references.

Paths inside the spec are relative to spec["assets"], which itself is relative to the spec file.
See reels/nina_shchi.json for the format.
"""
import json
import os


class SpecError(ValueError):
    pass


def load(path):
    spec = json.load(open(path, encoding="utf-8"))
    base = os.path.dirname(os.path.abspath(path))
    assets = os.path.normpath(os.path.join(base, spec.get("assets", ".")))
    spec["_assets"] = assets
    rel = lambda p: p if os.path.isabs(p) else os.path.join(assets, p)
    if "script_file" in spec:
        spec["script"] = json.load(open(rel(spec["script_file"]), encoding="utf-8"))
    for c in spec["clips"]:
        c["_frame"] = rel(c["frame"])
    spec["_audio"] = {k: rel(v) for k, v in spec.get("audio", {}).items()}
    if spec.get("music"):
        spec["music"]["file"] = rel(spec["music"]["file"])
    validate(spec)
    return spec


def validate(spec):
    ids = {c["id"]: c for c in spec["clips"]}
    if len(ids) != len(spec["clips"]):
        raise SpecError("duplicate clip ids")
    for c in spec["clips"]:
        if c["kind"] not in ("action", "talk"):
            raise SpecError(f"{c['id']}: kind must be action or talk")
        if c["kind"] == "action" and not c.get("prompt"):
            raise SpecError(f"{c['id']}: action clip needs a prompt")
        if c["kind"] == "talk" and not c.get("audio"):
            raise SpecError(f"{c['id']}: talk clip needs an audio key")
    for seg in spec["layout"]:
        refs = [seg.get("clip"), seg.get("bottom"), *seg.get("clips", []), *seg.get("top", [])]
        for r in filter(None, refs):
            if r not in ids:
                raise SpecError(f"layout references unknown clip {r}")


def script(spec):
    return spec.get("script", {})


def talk_audio_keys(spec):
    return {c["audio"] for c in spec["clips"] if c["kind"] == "talk"}


def audio_keys(spec):
    keys = talk_audio_keys(spec) | {s["audio"] for s in spec["layout"] if s.get("audio")}
    return sorted(keys)


def audio_paths(spec, audio_dir=None):
    """{key: file}. A TTS run's audio_dir/<key>.mp3 wins over the spec's own files."""
    out = {}
    for k in audio_keys(spec):
        gen = os.path.join(audio_dir, f"{k}.mp3") if audio_dir else None
        if gen and os.path.exists(gen):
            out[k] = gen
        elif k in spec["_audio"]:
            out[k] = spec["_audio"][k]
        else:
            raise SpecError(f"no audio for '{k}': add it to spec.audio or run with --tts")
    return out


def check_files(spec, audio):
    missing = [c["_frame"] for c in spec["clips"] if not os.path.exists(c["_frame"])]
    missing += [p for p in audio.values() if not os.path.exists(p)]
    if spec.get("music") and not os.path.exists(spec["music"]["file"]):
        missing.append(spec["music"]["file"])
    if missing:
        raise SpecError("missing files:\n  " + "\n  ".join(missing))
