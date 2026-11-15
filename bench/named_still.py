"""The cameras that do not move that Tracker is scored on, laid out like MOTS20 for
bench/named.py (--data <out>): security-camera video from MEVA (mevadata.org, CC BY 4.0),
5-minute 1080p clips from a bus station, a school, an office and a hospital, in a public
bucket that needs no login.

One clip from each of six cameras; from each, two windows of 150 frames at 10 frames a
second, scaled to 1280 pixels wide; every window asked for the same five things, whatever
it shows. One sequence per window and prompt, 60 in all. (Six more cameras, the ones
between these in the bucket's listing, are where Tracker's settings were looked at; they
are left out.)

Nobody has labelled these videos. The reference is SAM 3's own video tracker: this
script runs it on every sequence and saves what it tracks as the labels, so a score here
says how closely a tracker reproduces SAM 3, not how right either is. A sequence SAM 3
runs out of GPU memory on is left without labels and is not scored.

  --again   SAM 3 once more, started one frame earlier, scored against those labels:
            how far SAM 3 agrees with itself, the most a tracker can be expected to reach

Needs ffmpeg.
"""
import argparse
import json
import os
import re
import subprocess
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote

os.environ.setdefault("TQDM_DISABLE", "1")

import torch
from PIL import Image

from people import DEVICE, mots_lines, score, sequences
from tamserve.reference import clock

BUCKET = "https://mevadata-public-01.s3.amazonaws.com"
FOLDER = "drops-123-r13/2018-03-11/16/"  # one date and hour
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
PROMPTS = ("person", "car", "backpack", "chair", "door")
STARTS, FRAMES, FPS, WIDTH = (30, 180), 150, 10, 1280  # seconds into the clip; frames; frames a second; pixels


def clips():
    """(site, camera, key) of the six cameras: the earliest clip of each camera in the
    folder, cameras taken from each site in turn, every second one of the first twelve."""
    keys, token = [], None
    while True:
        url = f"{BUCKET}/?list-type=2&prefix={quote(FOLDER)}" + (f"&continuation-token={quote(token, safe='')}" if token else "")
        root = ET.fromstring(urllib.request.urlopen(url, timeout=60).read())
        keys += [c.find(NS + "Key").text for c in root.iter(NS + "Contents")]
        token = root.findtext(NS + "NextContinuationToken")
        if not token:
            break
    first = {}
    for key in sorted(keys):
        m = re.search(r"\.([a-z]+)\.(G\d+)\.r13\.avi$", key)
        if m and m.group(2) not in first:
            first[m.group(2)] = (m.group(1), key)
    sites = {}
    for cam, (site, key) in sorted(first.items()):
        sites.setdefault(site, []).append((cam, key))
    chosen = []
    while len(chosen) < 12 and any(sites.values()):
        for site in sorted(sites):
            if sites[site] and len(chosen) < 12:
                chosen.append((site, *sites[site].pop(0)))
    return chosen[1::2]


def lay_out(out, store):
    """Fetch the clips and cut the windows. Each keeps one frame more than a sequence
    uses, the one before its first: img0 holds all of them, img1 the sequence's own."""
    store.mkdir(parents=True, exist_ok=True)
    for site, cam, key in clips():
        clip = store / Path(key).name
        if not clip.exists():
            urllib.request.urlretrieve(f"{BUCKET}/{quote(key)}", clip)
        for start in STARTS:
            window = f"{site}{cam}-{start:03d}"
            frames = out.parent / (out.name + "_frames") / window
            if len(list(frames.glob("*.jpg"))) != FRAMES + 1:
                frames.mkdir(parents=True, exist_ok=True)
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(start), "-i", str(clip),
                                "-vf", f"fps={FPS},scale={WIDTH}:-2", "-frames:v", str(FRAMES + 1),
                                "-q:v", "2", str(frames / "%06d.jpg")], check=True)
            names = sorted(frames.glob("*.jpg"))
            w, h = Image.open(names[0]).size
            for prompt in PROMPTS:
                d = out / "train" / f"{window}-{prompt}"
                for folder, use in (("img0", names), ("img1", names[1:])):
                    (d / folder).mkdir(parents=True, exist_ok=True)
                    for k, src in enumerate(use, start=1):
                        if not (d / folder / f"{k:06d}.jpg").exists():
                            (d / folder / f"{k:06d}.jpg").symlink_to(src.resolve())
                (d / "prompt.txt").write_text(prompt + "\n")
                (d / "seqinfo.ini").write_text(f"[Sequence]\nname={d.name}\nimDir=img1\nframeRate={FPS}\n"
                                               f"seqLength={FRAMES}\nimWidth={w}\nimHeight={h}\nimExt=.jpg\n")


def track(predictor, folder, prompt, skip=0):
    """SAM 3's tracks on a folder of frames, as MOTS rows without its first `skip` frames, and the seconds it took."""
    w, h = Image.open(next(folder.glob("*.jpg"))).size
    lines = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        sid = predictor.handle_request(dict(type="start_session", offload_video_to_cpu=True,
                                            offload_state_to_cpu=True, resource_path=str(folder)))["session_id"]
        try:
            t = clock(DEVICE)
            predictor.handle_request(dict(type="add_prompt", session_id=sid, frame_index=0, text=prompt))
            for r in predictor.handle_stream_request(dict(type="propagate_in_video", session_id=sid)):
                o = r["outputs"]
                if r["frame_index"] >= skip:
                    lines += mots_lines(r["frame_index"] + 1 - skip,
                                        {int(i): m for i, m in zip(o["out_obj_ids"], o["out_binary_masks"])}, h, w)
            return lines, clock(DEVICE) - t
        finally:
            predictor.handle_request(dict(type="close_session", session_id=sid))


def main():
    from sam3.model_builder import build_sam3_video_predictor

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/still")
    ap.add_argument("--clips", default="data/meva")
    ap.add_argument("--sam3", help="path to sam3.pt; without it the weights come from Hugging Face")
    ap.add_argument("--again", action="store_true", help="SAM 3 started one frame earlier, scored against its labels")
    ap.add_argument("--results", default="results/still")
    args = ap.parse_args()
    out, train = Path(args.out), Path(args.out) / "train"
    lay_out(out, Path(args.clips))
    predictor = build_sam3_video_predictor(checkpoint_path=args.sam3)
    for d in sorted(p for p in train.iterdir() if not (p / "gt").exists()):
        (d / "gt").mkdir()
        try:
            lines, _ = track(predictor, d / "img1", (d / "prompt.txt").read_text().strip())
            (d / "gt" / "gt.txt").write_text("\n".join(lines) + "\n" if lines else "")
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
        print(json.dumps({"sequence": d.name, "labelled": (d / "gt" / "gt.txt").exists()}), flush=True)
    if args.again:
        seqs = sequences(out)
        lines, seconds = {}, 0.0
        for s in seqs:
            lines[s], t = track(predictor, train / s / "img0", (train / s / "prompt.txt").read_text().strip(), skip=1)
            seconds += t
        name = "sam3-one-frame-earlier"
        frames = len(seqs) * (FRAMES + 1)
        print(json.dumps({"system": name, "sequences": len(seqs), "ms_per_frame": round(seconds / frames * 1e3, 2),
                          **score(name, lines, seqs, out, Path(args.results))}), flush=True)


if __name__ == "__main__":
    main()
