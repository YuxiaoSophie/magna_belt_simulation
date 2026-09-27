#!/usr/bin/env python3
"""One MP4 of every motion-primitive prediction video, each after a short title card.

Reads ``<run>/index.json`` (order, effective amplitudes), ``<run>/eval.json`` (one-step RMSEs)
and ``<run>/videos/<episode>_<primitive>.mp4``; writes ``<run>/videos/all_primitives.mp4``
(re-encoded as the renderer does: libx264, crf 20, yuv420p) and ``..._cards/<name>.png``.

Run:
    uv run python scripts/lcs/compile_motion_primitive_videos.py \\
        --run data/lcs/motion_primitives/<run>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from PIL import Image, ImageDraw

from task_common.prediction_video import _font
from task_common.replay_video import CRF, FFMPEG, LEGEND_TEXT, Encoder, probe

WHAT = {
    "both_up_down": "both arms down and back up, twice",
    "both_fwd_back": "both arms along the belt's long axis: +, then -",
    "both_sideways": "both arms along the grasp axis: +, then -",
    "opposite_fwd_back": "arms opposite along the long axis (shear): +/-, then -/+",
    "opposite_sideways": "arms opposite along the grasp axis: stretch, then slack",
    "opposite_up_down": "one arm lower than the other, alternating (tilt), both below start",
    "franka_only": "Franka only: grasp axis, long axis, then down and back",
    "ur_only": "UR only: grasp axis, long axis, then down and back",
    "wrist_roll": "both wrists roll about the tool z, twice each way",
    "random_mix": "smooth random motion on all 12 axes",
    "good_mix": "both arms together: smooth random down/up + along the long axis only",
    "good_mix_tilt": "good_mix + small independent vertical offsets per arm",
}
OOD_LINE = "ACTION OUT OF DISTRIBUTION: per-step actions up to {:g}x the training bounds"
OOD_RGB = (180, 30, 30)


def card(name: str, lines: list[str], W: int, H: int, warn: str | None = None) -> np.ndarray:
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    big, small = _font(round(H * 0.07)), _font(round(H * 0.03))
    y = round(H * 0.34)
    items = [(name, big, LEGEND_TEXT)] + [(ln, small, LEGEND_TEXT) for ln in lines]
    if warn:
        items.append((warn, small, OOD_RGB))
    for text, font, rgb in items:
        x0, y0, x1, y1 = d.textbbox((0, 0), text, font=font)
        d.text(((W - (x1 - x0)) // 2 - x0, y - y0), text, fill=rgb, font=font)
        y += (y1 - y0) + round(H * (0.05 if font is big else 0.025))
    return np.asarray(img)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--card-s", type=float, default=1.5)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--first", default=None, help="primitive shown first")
    p.add_argument("--exclude", default="", help="comma list of primitives left out")
    args = p.parse_args()
    index = json.loads((args.run / "index.json").read_text())
    ev = json.loads((args.run / "eval.json").read_text())["primitives"]
    vids = args.run / "videos"
    out = args.out or vids / "all_primitives.mp4"
    cards_dir = out.with_name(out.stem + "_cards")
    cards_dir.mkdir(parents=True, exist_ok=True)
    skip = {x for x in args.exclude.split(",") if x}
    rows = [r for r in index["episodes"] if r["status"] == "ok" and r["primitive"] not in skip]
    rows.sort(key=lambda r: r["primitive"] != args.first)
    segs = []
    with tempfile.TemporaryDirectory(prefix=".compile-", dir=vids) as tmp:
        for r in rows:
            name = r["primitive"]
            v = vids / f"{Path(r['file']).stem}_{name}.mp4"
            info = probe(v)
            fps = info["frames"] / info["duration_s"]
            lob = r["plan"].get("lobe_amplitudes")
            if lob:  # achieved + / - peaks; one-sided (down-only) axes show one number
                amp = ", ".join(
                    f"{c['arm']} {c['name'].split('_', 1)[1]} "
                    + (f"{lob[c['name']][0]:.0f}" if abs(lob[c['name']][1]) < 1e-9 else
                       f"+{lob[c['name']][0]:.0f}/{lob[c['name']][1]:.0f}")
                    + f" {c['unit']}" for c in r["plan"]["components"])
            else:
                amp = ", ".join(f"{c['arm']} {c['name'].split('_', 1)[1]} "
                                f"{c['amp_effective']:.1f} {c['unit']}"
                                for c in r["plan"]["components"])
            a = ev[name]["all"]
            rmse = (f"one-step RMSE {a['model']['mean']:.2f} mm (reconstruction "
                    f"{a['recon']['mean']:.2f}, no motion {a['nomotion']['mean']:.2f})")
            lines = [WHAT.get(name, ""), f"peak amplitude: {amp}" if len(amp) < 110 else
                     "peak amplitude: see eval.json", rmse]
            if r["plan"].get("motion_s") is not None:
                lines.insert(1, f"motion {r['plan']['motion_s']:.1f} s")
            frac = r["plan"].get("bound_frac", 0.8)
            warn = OOD_LINE.format(frac) if r["plan"].get("action_ood") else None
            img = card(name, lines, info["width"], info["height"], warn)
            Image.fromarray(img).save(cards_dir / f"{name}.png")
            seg = Path(tmp) / f"card_{name}.mp4"
            enc = Encoder(seg, info["width"], info["height"], fps)
            for _ in range(round(args.card_s * fps)):
                enc.write(img)
            enc.close()
            segs += [seg, v]
        lst = Path(tmp) / "list.txt"
        lst.write_text("".join(f"file '{s.resolve()}'\n" for s in segs))
        tmp_out = Path(tmp) / out.name
        subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat",
                        "-safe", "0", "-i", str(lst), "-c:v", "libx264", "-crf", str(CRF),
                        "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        str(tmp_out)], check=True)
        tmp_out.replace(out)
    info = probe(out)
    print(f"wrote {out}: {info['frames']} frames {info['width']}x{info['height']}, "
          f"{info['duration_s']:.2f} s, {info['bytes'] / 1e6:.2f} MB; cards in {cards_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
