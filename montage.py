#!/usr/bin/env python3
"""Montage of a 9:16 reel from generated clips, driven by the reel spec's "layout" (Mac side, ffmpeg).

Generalises nina-hybrid/tools/build_nina2.py. Segment types:
  talk   — talking clip full screen with its own sound            {"clip", "trim_start"}
  broll  — silent action clips cut evenly under a voice-over track {"clips", "audio"}
  split  — action clips on top (slowed to fit), talking clip below {"top", "bottom", "bottom_crop_y"}
Then: big subtitles of 1–3 words (text from the script, word timings from whisper-cli), text overlays
(hook title, "AI-персонаж" label), optional music bed.

usage: montage.py <reel.json> <clips_dir> <out.mp4> [--audio-dir DIR]
"""
import argparse, difflib, json, os, re, subprocess, sys, tempfile

VF = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,fps=30,setsar=1"
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", os.path.expanduser("~/development/whisper-models/ggml-small.bin"))
FONT = os.environ.get("REEL_FONT", "/System/Library/Fonts/Helvetica.ttc")


def run(cmd):
    subprocess.run(cmd, check=True)


def dur(f):
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                          "-of", "csv=p=0", f]).decode())


def enc(args, out):
    run(["ffmpeg", "-y", "-loglevel", "error", *args, "-c:v", "libx264", "-preset", "fast", "-crf", "18",
         "-c:a", "aac", "-ar", "44100", "-ac", "2", "-pix_fmt", "yuv420p", out])


def concat(files, out, t):
    lst = os.path.join(t, os.path.basename(out) + ".txt")
    open(lst, "w").write("".join(f"file '{f}'\n" for f in files))
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", out])


def norm(w):
    return re.sub(r"[^а-яёa-z0-9іїєґ]", "", w.lower().replace("ё", "е").replace("́", ""))


def parse_srt(path):
    blocks = open(path, encoding="utf-8").read().strip().split("\n\n")
    ts = lambda x: sum(float(v) * m for v, m in zip(x.replace(",", ".").split(":"), (3600, 60, 1)))
    cues = []
    for b in blocks:
        lines = b.strip().splitlines()
        if len(lines) < 2:
            continue
        m = re.match(r"(\S+) --> (\S+)", lines[1])
        cues.append((ts(m.group(1)), ts(m.group(2)), " ".join(l.strip() for l in lines[2:])))
    return cues


def render_text(text, out, y, size, box=False, width=1080, height=1920):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    font = ImageFont.truetype(FONT, size, index=1)
    words, lines, cur = text.split(), [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if d.textlength(test, font=font) > width - 140 and cur:
            lines.append(cur); cur = w
        else:
            cur = test
    lines.append(cur)
    lh = int(size * 1.18)
    for i, line in enumerate(lines):
        tw = d.textlength(line, font=font); x = (width - tw) / 2; yy = y + i * lh
        if box:
            d.rounded_rectangle((x - 18, yy - 10, x + tw + 18, yy + size + 12), 12, fill=(0, 0, 0, 115))
            d.text((x, yy), line, font=font, fill=(255, 255, 255, 235))
        else:
            d.text((x, yy), line, font=font, fill="white", stroke_width=6, stroke_fill="black")
    im.save(out)


def script_cues(wcues, script_words, max_chars=16):
    """Aligns script words to whisper word timings (difflib) and groups them by 1–3 words."""
    words = script_words
    ww = [(s, e, w) for s, e, w in wcues if norm(w)]
    sm = difflib.SequenceMatcher(None, [norm(w) for w in words], [norm(w) for _, _, w in ww], autojunk=False)
    times = [None] * len(words)
    for a0, b0, n in sm.get_matching_blocks():
        for k in range(n):
            times[a0 + k] = (ww[b0 + k][0], ww[b0 + k][1])
    known = [i for i, x in enumerate(times) if x]
    for i in range(len(words)):  # interpolate words whisper missed
        if times[i] is None:
            prev = max([j for j in known if j < i], default=None); nxt = min([j for j in known if j > i], default=None)
            s = times[prev][1] if prev is not None else 0.0
            e = times[nxt][0] if nxt is not None else s + 0.4
            span = (nxt if nxt is not None else len(words)) - (prev if prev is not None else -1)
            pos = i - (prev if prev is not None else -1)
            st = s + (e - s) * (pos - 1) / span; times[i] = (st, st + (e - s) / span)
    cues, cur, cs, last_e = [], [], None, 0.0
    for w, (s, e) in zip(words, times):
        if w in ("—", "-"):
            continue
        txt = " ".join(cur + [w])
        short = len(norm(cur[-1])) <= 2 if cur else False  # don't leave «с», «не», «в» at a line end
        if cur and not short and (len(txt) > max_chars or re.search(r"[.!?:]$", cur[-1])):
            cues.append((cs, s, " ".join(cur))); cur, cs = [], None
        if cs is None:
            cs = s
        cur.append(w); last_e = e
    if cur:
        cues.append((cs, last_e + 0.3, " ".join(cur)))
    return [(s, max(e, s + 0.35), x) for s, e, x in cues]


def build(spec, clips_dir, audio, out, script):
    """spec: reel dict; audio: {key: path}; script: {key: text} in spoken order."""
    t = tempfile.mkdtemp(prefix="montage_")
    clip = lambda cid: os.path.join(clips_dir, f"{cid}.mp4")
    segs, split_ranges, pos = [], [], 0.0
    for i, seg in enumerate(spec["layout"]):
        o = f"{t}/s{i}.mp4"
        if seg["type"] == "talk":
            enc(["-ss", str(seg.get("trim_start", 0)), "-i", clip(seg["clip"]), "-vf", VF], o)
        elif seg["type"] == "broll":
            vo = audio[seg["audio"]]
            each = dur(vo) / len(seg["clips"])
            parts = []
            for k, cid in enumerate(seg["clips"]):
                enc(["-i", clip(cid), "-t", f"{each:.3f}", "-vf", VF, "-an"], f"{t}/s{i}_{k}.mp4")
                parts.append(f"{t}/s{i}_{k}.mp4")
            concat(parts, f"{t}/s{i}_v.mp4", t)
            enc(["-i", f"{t}/s{i}_v.mp4", "-i", vo, "-map", "0:v", "-map", "1:a", "-shortest"], o)
        elif seg["type"] == "split":
            bottom = clip(seg["bottom"])
            d = dur(bottom)
            part = d / len(seg["top"])
            parts = []
            for k, cid in enumerate(seg["top"]):
                slow = max(part / dur(clip(cid)), 1)
                enc(["-i", clip(cid), "-vf",
                     f"setpts={slow:.3f}*PTS,scale=1080:960:force_original_aspect_ratio=increase,crop=1080:960,fps=30,setsar=1",
                     "-t", f"{part:.3f}", "-an"], f"{t}/s{i}_{k}.mp4")
                parts.append(f"{t}/s{i}_{k}.mp4")
            concat(parts, f"{t}/s{i}_top.mp4", t)
            cy = seg.get("bottom_crop_y", 180)
            enc(["-i", f"{t}/s{i}_top.mp4", "-i", bottom, "-filter_complex",
                 f"[1:v]scale=1080:1920,crop=1080:960:0:{cy},fps=30,setsar=1[b];[0:v][b]vstack=inputs=2[v]",
                 "-map", "[v]", "-map", "1:a", "-shortest"], o)
        else:
            raise ValueError(f"unknown layout segment type: {seg['type']}")
        d = dur(o)
        if seg["type"] == "split":
            split_ranges.append((pos, pos + d))
        pos += d
        segs.append(o)
    concat(segs, f"{t}/joined.mp4", t)
    total = dur(f"{t}/joined.mp4")

    pngs = []
    sub = spec.get("subtitles")
    if sub:
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{t}/joined.mp4", "-vn", "-ar", "16000", "-ac", "1", f"{t}/a.wav"])
        run(["whisper-cli", "-m", WHISPER_MODEL, "-l", sub.get("lang", "ru"), "-ml", "1", "-sow", "-osrt",
             "-of", f"{t}/words", "-f", f"{t}/a.wav", "-np"])
        words = " ".join(script.values()).replace("́", "").split()
        cues = script_cues(parse_srt(f"{t}/words.srt"), words, sub.get("max_chars", 16))
        for k, (s, e, txt) in enumerate(cues):
            txt = re.sub(r"[.,!?]+$", "", txt.strip()).upper()
            in_split = any(a <= s < b for a, b in split_ranges)
            y = sub.get("split_y", 900) if in_split else sub.get("y", 1300)  # in a split screen — on the seam
            f = f"{t}/c{k}.png"; render_text(txt, f, y=y, size=sub.get("size", 92)); pngs.append((f, s, e))
    for k, ov in enumerate(spec.get("overlays", [])):
        f = f"{t}/o{k}.png"
        render_text(ov["text"], f, y=ov.get("y", 260), size=ov.get("size", 100), box=ov.get("box", False))
        pngs.append((f, ov.get("start", 0), ov.get("end", total)))

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", f"{t}/joined.mp4"]
    for f, _, _ in pngs:
        cmd += ["-i", f]
    chain, last = [], "0:v"
    for k, (_, s, e) in enumerate(pngs):
        chain.append(f"[{last}][{k + 1}:v]overlay=0:0:enable='gte(t,{s:.2f})*lt(t,{e:.2f})'[v{k}]"); last = f"v{k}"
    amap = "0:a"
    music = spec.get("music")
    if music:
        cmd += ["-stream_loop", "-1", "-i", music["file"]]
        chain.append(f"[{1 + len(pngs)}:a]volume={music.get('db', -24)}dB,afade=t=out:st={total - 1.5:.2f}:d=1.5[m];"
                     f"[0:a][m]amix=inputs=2:duration=first:normalize=0[ax]")
        amap = "[ax]"
    if chain:
        cmd += ["-filter_complex", ";".join(chain), "-map", f"[{last}]" if pngs else "0:v", "-map", amap]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-c:a", "aac", "-b:a", "160k",
            "-movflags", "+faststart", out]
    run(cmd)
    return out, round(dur(out), 1)


def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import reel_spec
    ap = argparse.ArgumentParser()
    ap.add_argument("spec"); ap.add_argument("clips_dir"); ap.add_argument("out")
    ap.add_argument("--audio-dir", help="use generated audio from here (TTS run) instead of the spec's files")
    a = ap.parse_args()
    spec = reel_spec.load(a.spec)
    audio = reel_spec.audio_paths(spec, a.audio_dir)
    print(*build(spec, a.clips_dir, audio, a.out, reel_spec.script(spec)))


if __name__ == "__main__":
    main()
