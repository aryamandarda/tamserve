"""The videos Tracker is scored on, laid out like MOTS20 for bench/named.py (--data <out>):
part of SA-Co/VEval, Meta's own video benchmark for SAM 3, the validation videos filmed
with smart glasses, so the camera moves with the wearer's head.

SA-Co/VEval pairs each video (20 seconds, 6 frames a second) with phrases such as "the
white scarf" or "the spiral staircase", and for each pair outlines every matching object
on every frame. Some pairs name nothing in the video: there a tracker should stay silent.
One sequence per video and phrase, the phrase in prompt.txt: every pair with something
to find, and up to --negatives of each video's pairs with nothing.

The first --skip videos are left out: Tracker's settings were chosen on them, so a
score on them flatters it. What is laid out is 224 pairs from 117 other videos.

The dataset is gated on Hugging Face (facebook/SACo-VEval): accept its licence and log
in with `hf auth login`. Its frames are one 30 GB archive, read here as a stream that
stops after --videos complete videos. Scores are TrackEval's HOTA over the pairs, as for
the other benchmarks here, not Meta's own cgF1.
"""
import argparse
import json
import random
import tarfile
from pathlib import Path

import numpy as np
import pycocotools.mask as rle
import requests
from huggingface_hub import get_token, hf_hub_download, hf_hub_url

REPO = "facebook/SACo-VEval"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/saco")
    ap.add_argument("--frames", default="data/saco_frames")
    ap.add_argument("--videos", type=int, default=160)
    ap.add_argument("--skip", type=int, default=40, help="leave out the first N videos, by name")
    ap.add_argument("--negatives", type=int, default=1, help="pairs with nothing to find kept per video")
    args = ap.parse_args()
    out, shared = Path(args.out), Path(args.frames)
    ann = json.load(open(hf_hub_download(REPO, "annotation/saco_veval_smartglasses_val.json", repo_type="dataset")))
    videos = {v["video_name"]: v for v in ann["videos"]}

    # 1. frames: stream the archive, keep validation videos, stop once enough are complete
    done = sorted(p.name for p in shared.iterdir() if (p / ".complete").exists()) if shared.exists() else []
    if len(done) < args.videos:
        r = requests.get(hf_hub_url(REPO, "media/saco_sg.tar.gz", repo_type="dataset"),
                         headers={"Authorization": "Bearer " + get_token()}, stream=True, timeout=120)
        current, got = None, set(done)
        with tarfile.open(fileobj=r.raw, mode="r|gz") as tar:
            for m in tar:
                parts = m.name.split("/")
                if not m.isfile() or len(parts) < 4:
                    continue
                name = parts[2]
                if name != current:  # the archive lists a video's frames together
                    if current in videos and current not in got:
                        (shared / current / ".complete").touch()
                        got.add(current)
                        if len(got) >= args.videos:
                            break
                    current = name
                if name in videos and name not in got and Path(parts[3]).name == parts[3]:
                    (shared / name).mkdir(parents=True, exist_ok=True)
                    (shared / name / parts[3]).write_bytes(tar.extractfile(m).read())
        r.close()
        done = sorted(got)

    # 2. one sequence per video and phrase
    by_video = {videos[n]["id"]: n for n in done}
    masklets = {}
    for a in ann["annotations"]:
        if a["video_id"] in by_video:
            masklets.setdefault((a["video_id"], a["category_id"]), []).append(a)
    rng = random.Random(0)
    pairs, per_video = [], {}
    for p in ann["video_np_pairs"]:
        if p["video_id"] in by_video:
            per_video.setdefault(p["video_id"], []).append(p)
    for vid, ps in sorted(per_video.items(), key=lambda t: int(t[0])):
        positive = [p for p in ps if (vid, p["category_id"]) in masklets]
        negative = [p for p in ps if (vid, p["category_id"]) not in masklets]
        pairs += positive + rng.sample(negative, min(args.negatives, len(negative)))
    skipped = set(done[:args.skip])  # chosen after the sampling, so the pairs do not depend on --skip
    pairs = [p for p in pairs if by_video[p["video_id"]] not in skipped]
    n_pos = n_neg = 0
    for p in pairs:
        name = by_video[p["video_id"]]
        v = videos[name]
        h, w = int(v["height"]), int(v["width"])
        d = out / "train" / f"{name[8:]}-{p['category_id']}"
        (d / "gt").mkdir(parents=True, exist_ok=True)
        if not (d / "img1").exists():
            (d / "img1").symlink_to((shared / name).resolve())
        gt = []
        tracks = masklets.get((p["video_id"], p["category_id"]), [])
        for k in range(len(v["file_names"])):
            taken = np.zeros((h, w), bool)
            for a in tracks:
                seg = a["segmentations"][k]
                if not seg:
                    continue
                m = rle.decode({"size": seg["size"], "counts": seg["counts"].encode() if isinstance(seg["counts"], str)
                                else seg["counts"]}).astype(bool) & ~taken
                if not m.any():
                    continue
                taken |= m
                counts = rle.encode(np.asfortranarray(m.astype(np.uint8)))["counts"].decode()
                gt.append(f"{k + 1} {2000 + int(a['id'])} 2 {h} {w} {counts}")
        (d / "gt" / "gt.txt").write_text("\n".join(gt) + "\n" if gt else "")
        (d / "prompt.txt").write_text(p["noun_phrase"] + "\n")
        (d / "seqinfo.ini").write_text(f"[Sequence]\nname={d.name}\nseqLength={len(v['file_names'])}\n"
                                       f"imWidth={w}\nimHeight={h}\n")
        n_pos += bool(tracks)
        n_neg += not tracks
    print(json.dumps({"videos": len({p["video_id"] for p in pairs}), "pairs": len(pairs),
                      "with something to find": n_pos, "with nothing": n_neg}))


if __name__ == "__main__":
    main()
