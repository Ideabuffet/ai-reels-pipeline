"""ElevenLabs voice-over from the reel script (key in ELEVENLABS_API_KEY, never printed).

Each script key becomes <out_dir>/<key>.mp3. Audio for talking parts gets voice.pad_s seconds of
silence in front: InfiniteTalk "warms up" on the first frames, the montage trims them (trim_start).
"""
import os
import subprocess

import requests

API = "https://api.elevenlabs.io/v1"
USD_PER_CHAR = 22 / 100_000  # Creator plan: $22 / 100k credits, multilingual v2 = 1 credit per char


def synth(text, voice_id, model, out, settings=None):
    r = requests.post(f"{API}/text-to-speech/{voice_id}", params={"output_format": "mp3_44100_128"},
                      headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"], "Content-Type": "application/json"},
                      json={"text": text, "model_id": model, **({"voice_settings": settings} if settings else {})},
                      timeout=300)
    if r.status_code != 200:
        raise RuntimeError(f"ElevenLabs HTTP {r.status_code}: {r.text[:300]}")
    open(out, "wb").write(r.content)
    return int(r.headers.get("x-character-count") or r.headers.get("character-cost") or len(text))


def pad(src, out, seconds):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-t", str(seconds), "-i",
                    "anullsrc=r=44100:cl=mono", "-i", src, "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1[a]",
                    "-map", "[a]", "-c:a", "libmp3lame", "-b:a", "128k", out], check=True)


def generate(spec, out_dir, keys, talk_keys):
    """Returns ({key: mp3}, chars)."""
    if not os.environ.get("ELEVENLABS_API_KEY"):
        raise RuntimeError("ELEVENLABS_API_KEY is not set")
    v = spec["voice"]
    os.makedirs(out_dir, exist_ok=True)
    files, chars = {}, 0
    for k in keys:
        text = spec["script"].get(k)
        if not text:
            raise RuntimeError(f"script has no text for audio key '{k}'")
        raw = os.path.join(out_dir, f"{k}.raw.mp3")
        chars += synth(text, v["voice_id"], v.get("model", "eleven_multilingual_v2"), raw, v.get("settings"))
        dst = os.path.join(out_dir, f"{k}.mp3")
        if k in talk_keys and v.get("pad_s"):
            pad(raw, dst, v["pad_s"])
            os.remove(raw)
        else:
            os.replace(raw, dst)
        files[k] = dst
    return files, chars
