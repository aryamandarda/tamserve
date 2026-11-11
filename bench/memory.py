"""Memory against video length, one object: the stock predictor, which loads a
whole video and keeps every frame's outputs, and the engine, which keeps only
the memory the model reads. Synthetic frames.
"""
import argparse
import json
import os
import tempfile

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import psutil
import torch
from PIL import Image

import meta_sam2
from tamserve.engine import Engine, prepare
from tamserve.models import build_video_predictor
from tamserve.reference import precision

H, W = 360, 640
gb = lambda b: round(b / 2**30, 2)
rss = lambda: psutil.Process().memory_info().rss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="edgetam")
    ap.add_argument("--stock", default="edgetam", choices=["edgetam", "sam2"],
                    help="whose tracker stock is: EdgeTAM's repository's, or Meta's current sam2 package")
    ap.add_argument("--frames", default="200,400,800")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    pool = [Image.fromarray(rng.integers(0, 255, (H, W, 3), dtype=np.uint8)) for _ in range(8)]
    seed = torch.zeros(H, W, dtype=torch.bool)
    seed[40:80, 60:120] = True

    for n in (int(f) for f in args.frames.split(",")):
        row = {"model": args.model, "stock": args.stock, "frames": n}
        with tempfile.TemporaryDirectory() as d:
            for i in range(n):
                pool[i % 8].save(f"{d}/{i}.jpg")
            # Stock as it comes, then with its video and its outputs kept in host memory instead.
            for name, kw in (("stock", {}), ("stock_offloaded", {"offload_video_to_cpu": True,
                                                                 "offload_state_to_cpu": True})):
                pred = meta_sam2.build(args.model) if args.stock == "sam2" else build_video_predictor("cuda", args.model)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                base_gpu, base_host = torch.cuda.memory_allocated(), rss()
                with torch.inference_mode(), precision("cuda"):
                    state = pred.init_state(d, **kw)
                    pred.add_new_mask(state, frame_idx=0, obj_id=1, mask=seed)
                    for _ in pred.propagate_in_video(state):
                        pass
                row[f"{name}_gpu_gb"] = gb(torch.cuda.max_memory_allocated() - base_gpu)
                row[f"{name}_host_gb"] = gb(rss() - base_host)
                del state, pred
                torch.cuda.empty_cache()
        engine = Engine(build_video_predictor("cuda", args.model))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base_gpu, base_host = torch.cuda.memory_allocated(), rss()
        engine.step({0: prepare(pool[0], device="cuda")}, {0: {1: seed}})
        for i in range(1, n):
            engine.step({0: prepare(pool[i % 8], device="cuda")})
        row["engine_gpu_gb"] = gb(torch.cuda.max_memory_allocated() - base_gpu)
        row["engine_host_gb"] = gb(rss() - base_host)
        del engine
        torch.cuda.empty_cache()
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
