"""The home videos PeopleTracker is scored on, laid out like MOTS20 for bench/people.py
(--data <out>): BURST's validation videos from Charades, people at home filming
themselves, mostly one or two to a video, labelled once a second.

Keeps the 116 videos where every person is labelled. Frames: every 3rd (10 a second)
plus every labelled frame, taken from TAO's 82 GB validation archive, where Charades is
one unbroken 12.7 GB stretch, fetched with a single request. Only labelled frames are
scored; scored.txt lists them by run frame number.

BURST's server refuses some cloud machines: if the annotations do not download, fetch
annotations.zip elsewhere and copy it to <cache>/burst_annotations.zip.
"""
import argparse
import json
import socket
import struct
import urllib.request
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pycocotools.mask as rle

ANNOTATIONS = "https://omnomnom.vision.rwth-aachen.de/data/BURST/annotations.zip"
TAO_VAL = "https://motchallenge.net/data/2-TAO_VAL.zip"
PERSON = 805  # BURST category id
socket.setdefaulttimeout(120)  # a dead server fails the run instead of hanging it


def sequences(cache):
    path = cache / "burst_annotations.zip"  # or copy BURST's annotations.zip here
    if not path.exists():
        urllib.request.urlretrieve(ANNOTATIONS, path)
    with zipfile.ZipFile(path) as z:
        val = json.loads(z.read(next(n for n in z.namelist() if n.endswith("val/all_classes.json"))))
    return [s for s in val["sequences"] if s["dataset"] == "Charades" and s["seq_name"].isalnum()
            and PERSON not in s["not_exhaustive_category_ids"]
            and PERSON in s["track_category_ids"].values()]


def charades_frames(cache):
    """Local copy of the archive's Charades stretch, and where each frame sits in it."""
    from remotezip import RemoteZip

    with RemoteZip(TAO_VAL) as z:
        infos = [i for i in z.infolist() if i.filename.startswith("frames/val/Charades/") and not i.is_dir()]
    lo = min(i.header_offset for i in infos)
    hi = max(i.header_offset + i.compress_size for i in infos) + 1024  # room for the last local header
    blob = cache / "tao_val_charades.bin"
    if not blob.exists() or blob.stat().st_size != hi - lo:
        part = blob.with_suffix(".part")
        req = urllib.request.Request(TAO_VAL, headers={"Range": f"bytes={lo}-{hi - 1}"})
        with urllib.request.urlopen(req) as r, part.open("wb") as f:
            while chunk := r.read(1 << 24):
                f.write(chunk)
        part.rename(blob)
    return blob, lo, {i.filename: i for i in infos}


def extract(f, lo, info):
    """One member's bytes: skip its local header, then inflate."""
    f.seek(info.header_offset - lo)
    head = f.read(30)
    name_len, extra_len = struct.unpack("<HH", head[26:30])
    f.seek(name_len + extra_len, 1)
    data = f.read(info.compress_size)
    return zlib.decompress(data, -15) if info.compress_type == zipfile.ZIP_DEFLATED else data


def _people(s, a, frame):
    """MOTS rows for the people labelled on annotated frame a. MOTS masks may not
    overlap and 37 BURST frames' do, slightly: earlier tracks keep shared pixels."""
    h, w = s["height"], s["width"]
    taken, rows = np.zeros((h, w), bool), []
    for tid, seg in sorted(s["segmentations"][a].items(), key=lambda t: int(t[0])):
        if s["track_category_ids"][tid] == PERSON:
            m = rle.decode({"size": [h, w], "counts": seg["rle"].encode()}).astype(bool) & ~taken
            taken |= m
            counts = rle.encode(np.asfortranarray(m.astype(np.uint8)))["counts"].decode()
            rows.append(f"{frame} {2000 + int(tid)} 2 {h} {w} {counts}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/homes")
    ap.add_argument("--cache", default="data/burst")
    args = ap.parse_args()
    out, cache = Path(args.out), Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    seqs = sequences(cache)
    blob, lo, members = charades_frames(cache)
    with blob.open("rb") as f:
        for s in seqs:
            d = out / "train" / s["seq_name"]
            (d / "img1").mkdir(parents=True, exist_ok=True)
            (d / "gt").mkdir(exist_ok=True)
            labelled = {p: a for a, p in enumerate(s["annotated_image_paths"])}
            run = [p for i, p in enumerate(s["all_image_paths"]) if i % 3 == 0 or p in labelled]
            scored, gt = [], []
            for k, p in enumerate(run, start=1):
                img = d / "img1" / f"{k:06d}.jpg"
                if not img.exists():
                    img.write_bytes(extract(f, lo, members[f"frames/val/Charades/{s['seq_name']}/{p}"]))
                if p in labelled:
                    scored.append(k)
                    gt += _people(s, labelled[p], len(scored))
            (d / "gt" / "gt.txt").write_text("\n".join(gt) + "\n")
            (d / "scored.txt").write_text("\n".join(map(str, scored)) + "\n")
            (d / "seqinfo.ini").write_text(f"[Sequence]\nname={s['seq_name']}\nseqLength={len(scored)}\n"
                                           f"imWidth={s['width']}\nimHeight={s['height']}\n")
    print(json.dumps({"sequences": len(seqs), "frames": sum(1 for _ in out.glob("train/*/img1/*.jpg"))}))


if __name__ == "__main__":
    main()
