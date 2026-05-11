"""Quick test: which ffmpeg filter chain produces the best colors."""
import subprocess, sys
from pathlib import Path
import cv2
import numpy as np

video = '/Users/ValeryD/Documents/Отслеживание веса/20260505_213731.mp4'

def test(label, vf):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-ss", "1", "-i", video, "-vframes", "1",
           "-vf", vf, "-f", "image2pipe", "-vcodec", "png", "pipe:1"]
    r = subprocess.run(cmd, capture_output=True, timeout=20)
    if r.returncode != 0 or len(r.stdout) < 100:
        err = r.stderr[-300:].decode(errors='replace')
        print(f"{label}: FAILED rc={r.returncode}  {err}")
        return
    bgr = cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"{label}: imdecode returned None")
        return
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(float)
    print(f"{label}:")
    print(f"  luma  mean={g.mean():.1f}  p5={np.percentile(g,5):.0f}  "
          f"p50={np.percentile(g,50):.0f}  p95={np.percentile(g,95):.0f}")
    print(f"  B={bgr[:,:,0].mean():.1f}  G={bgr[:,:,1].mean():.1f}  R={bgr[:,:,2].mean():.1f}")
    # Save for visual inspection
    out = Path("hdr_debug") / f"ffmpeg_{label}.png"
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), bgr)
    print(f"  saved → {out}")

# ──────────────────────────────────────────────────────────────────────────
# Rewritten tests — run with TS=3 for a more representative frame
TS = "3"
video = '/Users/ValeryD/Documents/Отслеживание веса/20260505_213731.mp4'

# Baseline: what cv2/VideoToolbox gives
test("1_plain",       "format=rgb24")

# tonemap alone (does PQ EOTF + tone-map, outputs linear BT.2020)
test("2_tm_hable",     "tonemap=hable,format=rgb24")
test("3_tm_reinhard",  "tonemap=reinhard,format=rgb24")
test("4_tm_mobius",    "tonemap=mobius:desat=0,format=rgb24")
test("5_tm_bt.709",    "tonemap=bt.709,format=rgb24")

# tonemap + colorspace: apply BT.2020→BT.709 gamut matrix + sRGB gamma
test("6_tm_hable_cs",  "tonemap=hable,colorspace=primaries=bt709:trc=bt709,format=rgb24")
test("7_tm_reinh_cs",  "tonemap=reinhard,colorspace=primaries=bt709:trc=bt709,format=rgb24")
test("8_tm_mobius_cs", "tonemap=mobius:desat=0,colorspace=primaries=bt709:trc=bt709,format=rgb24")
test("9_tm_all_bt709", "tonemap=reinhard,colorspace=all=bt709,format=rgb24")
