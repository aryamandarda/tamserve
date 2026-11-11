# tamserve: results

Every measurement behind the README, with the command that produces it. All on
one machine: an RTX 4090 (24 GB), an AMD EPYC 9354 limited to 8.8 cores, PyTorch
2.15 with CUDA 13.2, Python 3.11, bf16 autocast (how these models normally run
on a GPU), EdgeTAM at commit `7711e01`. How the engine works is in
[how-it-works.md](how-it-works.md).

"Stock" is the tracker in Meta's EdgeTAM repository, used as it comes: Meta's SAM 2
video predictor as it stood in late 2024, which also runs the SAM 2.1 models.
Meta's separate SAM 2 repository has since changed its predictor (it can add
objects mid-video and has its own compiled mode); SAM 2.1 is measured against
that newer code too, in [its own section](#against-metas-current-sam-2-code).
"Compiled" stock is its
modules compiled the way Meta's `vos_optimized` mode compiles SAM 2 (EdgeTAM's
repository ships no compiled mode), in `torch.compile`'s `max-autotune` mode,
with the one compiler fix the engine also uses (see
[Compiling](how-it-works.md#compiling)); without that fix compiled stock ran at
126 frames/s instead of 186. The models' optional step that fills pinhole gaps in
masks needs a CUDA extension built at install time; it is not built here, so
neither stock nor tamserve applies it.

## Throughput

`bench/throughput.py`. Synthetic 640x360 frames already on the GPU (cost depends
on tensor shapes, not on what the pixels show), one object per camera, every
object's memory full. Frames per second are summed over cameras.

### Against stock's strongest setup

Stock runs one compiled process per camera, sharing the GPU through NVIDIA MPS,
which lets the processes' work overlap on the GPU. The engine is one compiled
process (`Engine(compile=True)`). Both are measured by the same code: a worker
process runs for a 20-second window after a 10-second lead-in, synchronises with
the GPU after every frame (stock) or step (engine), and counts the frames it
finishes and the time they took. Stock's frames with a memory that is still
filling, at the start of each pass over its video, are left out because they
are cheaper.

```bash
python bench/throughput.py --arms procs-compiled --procs 1,2,4,8                    # stock
python bench/throughput.py --arms engine-procs --procs 1 --sizes 1,2,4,8,16,32,64   # tamserve
```

| cameras | stock, frames/s | tamserve, frames/s | | stock GPU / host memory | tamserve GPU / host memory |
|---|---|---|---|---|---|
| 1 | 177 | 219 | 1.24x | 1.1 / 2.6 GB | 1.1 / 3.2 GB |
| 2 | 258 | 286 | 1.11x | 2.2 / 5.2 GB | 1.3 / 3.2 GB |
| 4 | 294 | 333 | 1.13x | 4.3 / 10.5 GB | 1.6 / 3.2 GB |
| 8 | 280 | 323 | 1.15x | 8.6 / 21.1 GB | 2.2 / 3.2 GB |
| 16 | | 322 | | | 2.9 / 3.2 GB |
| 32 | | 319 | | | 3.3 / 3.2 GB |
| 64 | | 321 | | | 5.7 / 3.2 GB |

GPU memory is what `nvidia-smi` reports in the middle of the window. Each stock
process holds its 64-frame video on the GPU (0.75 GB), which is subtracted here.
Stock was not run past 8 processes: 16 did not start on this host, whose
container limits threads, and would need 17 GB of GPU memory before any video.
Two earlier runs of each side landed within 4% of these: stock at 177 to 178,
259 to 260, 297 to 300 and 282 to 289, the engine at 217, 283, 331 and 322.

The engine copies every mask to the CPU in these numbers and stock leaves its
masks on the GPU. With `--gpu-masks` the engine does the same and gains about
1% (220, 287, 332, 325, 320, 323 and 326 frames/s, measured on the build just
before the last code review, when the engine's column read 217, 283, 331, 322,
320, 319 and 326).

### One process, median step

The same engine measured inside one process, as the median of 30 steps, with
peak GPU memory actually allocated. The uncompiled engine and stock on one
camera are measured the same way. This table, the two after it, start-up and
memory were measured on the build just before the last code review; the
like-for-like and accuracy tables were measured again after it.

```bash
python bench/throughput.py --arms stock-eager,stock-compiled,engine-eager,engine-compiled --sizes 1,2,4,8,16,32,64
python bench/throughput.py --arms engine-compiled --engine-mode default --sizes 1,2,4,8,16,32,64
```

| cameras | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| stock, uncompiled | 57 | | | | | | |
| stock, compiled | 186 | | | | | | |
| tamserve, uncompiled | 55 | 109 | 212 | 207 | 195 | 193 | 186 |
| tamserve, `compile="default"` | 115 | 233 | 323 | 315 | 315 | 317 | 316 |
| tamserve, `compile=True` | 217 | 291 | 350 | 339 | 337 | 332 | 328 |
| peak GPU memory, `compile=True` | 0.3 GB | 0.4 GB | 0.6 GB | 1.1 GB | 1.5 GB | 2.4 GB | 4.6 GB |

A quarter of the memory at 64 cameras is the benchmark's own frames (8 per
camera). When steps are queued without waiting for each one's masks
(`gpu_masks=True`, `bench/engine_profile.py --gpu-masks`), the engine reaches
255, 320, 354, 343 and 343 frames/s at 1 to 16 cameras.

### What each change bought

The compiled engine at 1, 2, 4, 8 and 64 cameras, frames/s, in the order the
changes were made. None of them changes a mask: uncompiled, one camera is still
bit-identical to stock after all of them.

| engine | 1 | 2 | 4 | 8 | 64 |
|---|---|---|---|---|---|
| batched and compiled, as it stood | 104 | 158 | 239 | 284 | 269 |
| no waiting inside a step | 114 | 227 | 305 | 302 | 275 |
| tuned kernels; CUDA graphs for calls of 1 or 2 rows; encoder in pieces of 4 | 221 | 275 | 322 | 326 | 309 |
| one position encoding, not one per frame; 16 objects per pass | 220 | 288 | 347 | 340 | 334 |

One and two cameras depend on how busy the host's CPU is: the second row
measured 98 and 188 on another run.

### Where a step goes

`python bench/engine_profile.py --stages`. Milliseconds per camera per step,
each stage timed with the GPU synchronised around it, so the stages add up to
more than the step where they overlap.

| cameras | frames/s | step, per camera | image encoder | memory attention | mask decoder | memory encoder | GPU idle |
|---|---|---|---|---|---|---|---|
| 1 | 220 | 4.55 | 2.57 | 1.65 | 1.16 | 0.71 | 14% |
| 2 | 288 | 3.48 | 1.82 | 1.19 | 0.65 | 0.44 | 11% |
| 4 | 347 | 2.88 | 1.40 | 1.00 | 0.57 | 0.38 | 6% |
| 8 | 340 | 2.94 | 1.48 | 0.96 | 0.33 | 0.27 | 3% |
| 16 | 338 | 2.96 | 1.45 | 0.98 | 0.28 | 0.26 | 4% |
| 32 | 337 | 2.97 | 1.42 | 0.98 | 0.28 | 0.27 | 2% |
| 64 | 334 | 2.99 | 1.41 | 0.98 | 0.27 | 0.27 | 3% |

## Start-up

Time until the engine is ready, from `ready_s` in `bench/throughput.py`. "First
time" is with PyTorch's on-disk caches empty
(`TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` pointed at an empty folder).

| | first time on a machine | later |
|---|---|---|
| `Engine(compile=True)` | 1,190 s | about 95 s |
| `Engine(compile="default")` | 164 s | 45 to 75 s |
| uncompiled | 2 s | 2 s |
| stock, compiled, one process | 325 s | 16 to 28 s |

The engine compiles five batch sizes, each for every memory length an object
passes through, so that nothing compiles later; stock compiles one.

## Accuracy

### The bar: stock against itself

`bench/noise_floor.py`, DAVIS 2017 validation, 30 videos, 61 objects. How much
stock's own masks change when nothing about the model changes:

| stock against stock | mean overlap (IoU) | worst 1% of frames | lost objects |
|---|---|---|---|
| the same run twice | 1.000 | 1.000 | 0 |
| an object tracked alone, then next to the video's other objects | 0.990 | 0.40 | 0 |
| bf16, then fp32 | 0.983 | 0.24 | 0 |

Against the hand-drawn masks the two middle runs score 0.7988 and 0.7991: stock
moves by 0.0002 from batch size alone, and single objects by up to 0.02. That is
rounding, and it is the size of change batching is allowed to make.

### The engine against stock

`bench/accuracy.py` (`--compile best` for the compiled engine). The 30 DAVIS
videos are cameras that join 3 frames apart and leave when their video ends, so
every step mixes objects that are starting, filling their memory, and in steady
state: 10.7 cameras at once on average, 26 at most. Stock tracks each video on
its own. Accuracy is mean overlap with the hand-drawn masks, averaged over
videos. "Lost" counts frames where stock had an object and the engine did not;
"extra" the reverse.

| model | stock | tamserve | change | videos better / worse | lost / extra | overlap with stock's masks |
|---|---|---|---|---|---|---|
| EdgeTAM | 0.8686 | 0.8684 | -0.0002 | 14 / 16 | 0 / 2 | 0.988 |
| EdgeTAM, `compile=True` | 0.8686 | 0.8685 | -0.0002 | 14 / 16 | 0 / 0 | 0.987 |
| SAM 2.1 tiny | 0.8807 | 0.8807 | +0.00005 | 16 / 14 | 0 / 0 | 0.997 |
| SAM 2.1 tiny, `compile=True` | 0.8807 | 0.8806 | -0.00003 | 14 / 16 | 0 / 0 | 0.995 |
| SAM 2.1 small | 0.8850 | 0.8846 | -0.0004 | 15 / 15 | 0 / 0 | 0.996 |
| SAM 2.1 base+ | 0.8893 | 0.8893 | +0.0001 | 12 / 18 | 0 / 0 | 0.998 |
| SAM 2.1 large | 0.8919 | 0.8918 | -0.0001 | 18 / 12 | 0 / 1 | 0.996 |

The extra frames are all in one video (kite-surf), where stock drops a small
object for a frame or two and the engine keeps it. Compiled results are not
repeatable to the last digit from one build to the next: an earlier build of
the compiled EdgeTAM engine scored 0.8689 (+0.0002, 17 videos better, 13 worse),
and of SAM 2.1 tiny 0.8805. The largest single-video
changes are in the videos where stock varies most against itself: -0.019
(EdgeTAM, paragliding-launch) and -0.009 (SAM 2.1 small, pigs), which is most
of that model's -0.0004.

### Bit for bit

`pytest` (`tests/test_engine.py`): with `Engine(exact=True)` one camera gives
masks identical to stock's on every frame, on the CPU and the GPU, for EdgeTAM
and SAM 2.1 tiny, for mask, box and click prompts. 28 tests passed on the GPU
machine; 23 run without a GPU.

## Memory

### Against video length

`python bench/memory.py`. One object; peak GPU memory allocated while tracking.

| frames | stock | stock, video and outputs kept in host memory | tamserve |
|---|---|---|---|
| 200 | 2.7 GB | 0.35 GB on the GPU, 2.4 GB on the host | 0.28 GB |
| 400 | 5.1 GB | 0.35 GB on the GPU, 4.8 GB on the host | 0.28 GB |
| 800 | 9.8 GB | 0.35 GB on the GPU, 9.5 GB on the host | 0.28 GB |

Stock grows by about 12 MB per frame wherever it keeps it. The engine's memory
does not depend on how long a camera has been running.

### Against cameras

In the throughput tables above: 2.2 GB of GPU memory and 3.2 GB of host memory
for 8 cameras in one process, against 8.6 GB and 21.2 GB for eight stock
processes.

## SAM 2.1

SAM 2.1 keeps 4,096 memory tokens per frame where EdgeTAM keeps 512, so memory
attention is most of a step (6 of 11 ms per camera for SAM 2.1 tiny) and the GPU
is the limit from one camera on. Batching cameras cannot raise its frame rate;
the engine's lead over stock is what removing the waits and the per-frame
position encodings is worth.

SAM 2.1 tiny, measured exactly as EdgeTAM above (`--model sam2.1-tiny`).
Compiled, one process per camera for stock, one process for the engine:

| cameras | stock, frames/s | tamserve, frames/s | | stock GPU / host memory | tamserve GPU / host memory |
|---|---|---|---|---|---|
| 1 | 77 | 83 | 1.08x | 1.4 / 2.2 GB | 1.6 / 3.7 GB |
| 4 | 85 | 87 | 1.01x | 5.4 / 8.9 GB | 2.4 / 3.7 GB |
| 8 | 83 | 87 | 1.05x | 10.7 / 17.8 GB | 3.4 / 3.7 GB |
| 16 | | 85 | | | 6.0 / 3.7 GB |
| 32 | | 86 | | | 6.6 / 3.7 GB |

One process, median step:

| cameras | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| stock, uncompiled | 58 | | | | | |
| stock, compiled | 80 | | | | | |
| tamserve, uncompiled | 74 | 76 | 77 | 74 | 72 | 71 |
| tamserve, `compile=True` | 89 | 92 | 92 | 89 | 89 | 88 |

At 8 cameras a step costs 10.8 ms per camera: memory attention 5.9, image
encoder 4.3, mask decoder 0.5, memory encoder 0.2. The compiled engine took 918
seconds to be ready the first time and about 75 after that. The larger SAM 2.1
models were measured for accuracy only.

### Against Meta's current SAM 2 code

The tracker in EdgeTAM's repository is a late-2024 copy of Meta's. Meta's own
SAM 2 repository has moved on since: its tracker follows each object
independently, can add an object after tracking has started, and has a compiled
mode (`vos_optimized`). So SAM 2.1 was measured again against that code, the
`sam2` package at version 1.1.0, installed next to tamserve with
`uv pip install sam2` and selected in the benchmarks with `--stock`.

Everything in this section comes from one session on a second machine: an RTX
4090 with an AMD EPYC 9684X, PyTorch 2.15 with CUDA 13.2. SAM 2.1 tiny, one
object per camera, counted as in [Throughput](#throughput).

Meta's compiled mode does not run on PyTorch 2.15. It stops with "accessing
tensor output of CUDAGraphs that has been overwritten by a subsequent run". It
was therefore measured two ways: as it comes, in an environment with PyTorch
2.5.1 (the oldest its repository supports), and on PyTorch 2.15 with its
modules compiled by the recipe used for the other stock.

One process per camera sharing the GPU through NVIDIA MPS for the three stock
columns, one process for tamserve, all compiled:

| cameras | Meta's code, its own compiled mode (PyTorch 2.5.1) | Meta's code, compiled as the other stock is | EdgeTAM repository's tracker | tamserve |
|---|---|---|---|---|
| 1 | 91 | 88 | 78 | 88 |
| 2 | 93 | 90 | 91 | 89 |
| 4 | 92 | 89 | 90 | 92 |
| 8 | 89 | 86 | 87 | 91 |
| 16 | | | | 91 |
| 32 | | | | 91 |

Frames per second, summed over cameras. One camera in one process, median
frame: Meta's code 75 uncompiled, 94 in its own compiled mode and 91 compiled
as the other stock is; the EdgeTAM repository's tracker 56 and 78; tamserve
uncompiled 71, 73, 75 and 73 at 1, 2, 4 and 8 cameras.

| cameras | Meta's code, GPU / host memory | tamserve, GPU / host memory |
|---|---|---|
| 1 | 1.3 / 1.5 GB | 1.8 / 4.3 GB |
| 2 | 2.5 / 3.0 GB | 1.6 / 4.3 GB |
| 4 | 5.0 / 6.0 GB | 2.4 / 4.3 GB |
| 8 | 10.0 / 12.0 GB | 3.4 / 4.3 GB |
| 16 | | 6.0 / 4.3 GB |
| 32 | | 6.6 / 4.3 GB |

Meta's figures are for its own compiled mode and leave out the 0.75 GB of video
each process holds on the GPU. On PyTorch 2.15 the same processes took 21.0 GB
of host memory at 8 cameras.

Against video length (`bench/memory.py --model sam2.1-tiny --stock sam2`): Meta's
code holds 2.7 GB of GPU memory after 200 frames and 10.2 GB after 800, or 0.27
GB on the GPU and 9.8 GB on the host when told to keep its video and outputs
there. tamserve holds 0.32 GB.

Accuracy on DAVIS 2017, as in [The engine against stock](#the-engine-against-stock)
but with Meta's current code as the reference (`bench/accuracy.py --stock sam2`):

| model | Meta's code | tamserve | lost / extra |
|---|---|---|---|
| SAM 2.1 tiny | 0.8809 | 0.8807 | 0 / 0 |
| SAM 2.1 tiny, `compile=True` | 0.8809 | 0.8806 | 0 / 0 |
| SAM 2.1 small | 0.8850 | 0.8846 | 0 / 0 |
| SAM 2.1 base+ | 0.8894 | 0.8893 | 0 / 0 |
| SAM 2.1 large | 0.8919 | 0.8918 | 0 / 1 |

For SAM 2.1, then, speed is level with Meta's current code, because the GPU is
the limit for both. What tamserve changes is memory, which stays put as cameras
are added and as time passes, and the interface: frames arrive one at a time
from live cameras instead of from a video loaded in full.

## Waiting inside a step

How often the CPU stops until the GPU has finished everything queued, in one
frame of steady tracking, as reported by
`torch.cuda.set_sync_debug_mode("warn")` (PyTorch notes that this mode does not
yet catch every such operation). Uncompiled.

| | EdgeTAM | SAM 2.1 tiny |
|---|---|---|
| stock predictor, per frame | 10 | 11 |
| tamserve, per step of 4 cameras | 1 | 1 |
| tamserve with `gpu_masks=True` | 0 | 0 |

Stock's ten are the masked writes in its prompt encoder, which run on every
frame whether or not there is a prompt; SAM 2.1's eleventh is a small copy to
the GPU while it builds the memory. The engine's one is the copy of the finished
masks to the CPU.

## Tried and not kept

| idea | what happened |
|---|---|
| CUDA graphs for every call | 220 instead of 130 frames/s at one camera, but 4 to 8% slower from 4 cameras up. Kept for calls of one or two rows only. |
| encoder in pieces of 8 or 16 beyond 8 frames | 3 to 8% slower than pieces of 4 from 16 cameras up (EdgeTAM); within 3% either way for SAM 2.1 |
| 64 objects per pass instead of 16 | the same speed, two more sizes to compile |
| two engines sharing the GPU through MPS | no faster than one engine with all the cameras |
| sharing the first self-attention across one camera's objects | exact, and 1.3x on memory attention with 10 objects per camera, but about 7% of such a step, for a copy of stock's attention loop. Not built. |
| folding the image encoder's branches and batch norms together | 13% faster, accuracy down 0.0022, one object lost |
| channels-last layout with cuDNN autotuning | 3% faster at a fixed batch size, slower when it changes; accuracy down 0.0013, two objects lost |
