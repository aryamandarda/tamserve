# How tamserve works

tamserve is a serving engine for SAM 2-family video trackers (EdgeTAM and
SAM 2.1). It runs the models' own code, with the weights Meta published; what it
changes is how that code is fed. Every number here is measured, and the
measurements are in [results.md](results.md).

## The model

EdgeTAM and SAM 2.1 follow an object through a video from one prompt: a box, a
few clicks or a mask on one frame. For each new frame they:

1. encode the image into a grid of feature vectors (the **image encoder**),
2. look up the object's **memory** to find it in this frame (**memory attention**),
3. draw this frame's mask (the **mask decoder**),
4. write this frame into the object's memory (the **memory encoder**).

Each object has its own memory: the frame it was prompted on, its last 6 frames,
and a short summary vector (an "object pointer") from each of its last 15
frames. Objects never read each other's memory.

The predictor in EdgeTAM's repository, which this project calls stock, is built
for a finished video file:

- One video per predictor. Many cameras mean many copies of the model, each in
  its own process.
- Every frame is loaded before tracking starts and every frame's outputs are
  kept, so memory grows with the length of the video.
- Every object has to be named before tracking starts. Adding one later raises
  "Cannot add new object id after tracking starts". (Meta's separate SAM 2
  repository lifted this in December 2024.)
- About a dozen times per frame, the CPU stops and waits for the GPU.

## Continuous batching

LLM servers such as vLLM serve many conversations on one GPU by stacking every
conversation that needs its next token into one pass; conversations join and
leave between passes. The engine does the same for video:

| LLM serving | tamserve |
|---|---|
| a conversation | an object on a camera |
| the prompt | a box, clicks or a mask |
| one generated token | one new frame |
| the KV cache | the object's memory |

Every object on every camera is one row of a batch. `step()` takes one frame
from each camera that has a new one, encodes those frames together, then runs
memory attention, the mask decoder and the memory encoder once over all rows.
Each row reads its own camera's frame and its own memory. Because objects never
interact, this is the same arithmetic as the stock predictor tracking several
objects in one video. Only floating-point rounding can differ, because a GPU
rounds slightly differently at different batch sizes.

### The same masks

- The engine calls the model's own image encoder, memory attention, mask decoder
  and memory encoder, and builds each object's memory the way stock's
  `_prepare_memory_conditioned_features` does: same frames, same order, same
  position codes.
- With `Engine(exact=True)`, one camera gives masks that are bit-for-bit
  identical to the stock predictor's, on the CPU and on the GPU, for EdgeTAM and
  SAM 2.1, for mask, box and click prompts (`tests/test_engine.py`). Getting
  there took reproducing details such as how stock lays its tensors out in
  memory and how it divides pixel values by 255.
- By default all objects that start in one step are prompted in one pass instead
  of one at a time. That changes rounding only.
- Across a whole dataset the difference from stock is smaller than the
  difference stock shows against itself when only its batch size changes
  ([results.md](results.md#accuracy)).

### Memory that does not grow

An object keeps only what the model reads: its prompt frame, 6 recent frames and
15 pointers. Older frames are dropped as new ones arrive, so an object can be
followed for as long as the camera runs, in fixed memory. The frames themselves
are never stored.

### Objects come and go

- A new object can start on any frame of any camera, from a box, clicks or a
  mask, without disturbing the objects already followed. `drop()` stops one
  object or a whole camera.
- A new object's memory fills over its first 16 frames, and rows whose memories
  have different sizes cannot share one attention call. The engine groups rows
  by memory size. Sixteen frames after the last object joined, that is one
  group.
- Prompting an object that is already followed starts it again from the new
  prompt: its memory is replaced.
- `add()` starts an object on the frame the engine has just processed, without
  advancing the other objects a second time. That is for prompts decided after
  looking at a frame's masks. The frame is not encoded again.
- Cameras need not share a resolution or a frame rate. Pass whichever cameras
  have a new frame; masks come back at the size of the frame they are for.

### Bounded GPU memory

However many cameras and objects there are, one call never grows past a fixed
size: the image encoder reads a few frames per call; memory attention, the mask
decoder and the memory encoder take at most 16 objects (SAM 2.1 attends over
about 29,000 memory tokens for each); and all objects share one copy of the
position encodings. More cameras make a step longer, not larger.

## Speed

A step has two costs: the GPU doing the arithmetic, and the CPU telling it what
to do, one small job ("kernel") at a time. A tracking step is about a thousand
kernels. With many cameras the GPU is the limit; with one or two, the CPU is.
The engine attacks both, without touching the model's code: where stock does
work nobody reads, the engine swaps in an equivalent for the duration of a step
and puts stock's back afterwards.

### No waiting inside a step

The GPU works through a queue that the CPU fills. As long as the CPU only adds
to the queue, both work at once. Some operations make the CPU stop until the GPU
has emptied the queue: an ordinary copy of data onto the GPU, or anything whose
size depends on values that live on the GPU.

Stock's prompt encoder does this ten times per frame. It writes its embeddings
with masked assignments (`x[labels == k] += w`), and it does so on every frame,
including the frames that have no prompt at all. The engine added two waits of
its own: copying a small index list to the GPU, and copying the masks back.
`torch.cuda.set_sync_debug_mode("warn")` reports each one.

The engine now computes the prompt encoder's "no prompt" output once, since it
is the same for every object on every frame, and sends its index lists from
pinned memory, which does not wait. That leaves one wait per step, the copy of
the finished masks to the CPU, and `gpu_masks=True` removes that one too. The
waits PyTorch reports per step went from 12 to 1. The masks are unchanged, bit
for bit.

### One position encoding, not one per frame

Stock's image encoder attaches a position encoding to every frame it encodes:
88 MB per frame at 1024 x 1024, identical for every frame, of which the model
reads one small part. The engine asks for a single frame's worth. That alone
made a step 4 to 7% faster and nearly halved GPU memory at 64 cameras (9.4 GB to
5.2 GB). The masks are unchanged, bit for bit.

### Compiling

`compile=True` runs the four modules through `torch.compile`, which fuses small
kernels into fewer, tuned ones. Four things had to be solved for that to work on
a batch whose size changes whenever an object comes or goes:

- **A few sizes only.** Compiled code is specific to its tensor sizes. The
  engine cuts every batch into power-of-two pieces, so each module only ever
  sees 1, 2, 4, 8 or 16 rows, and compiles all of them when it is created. No
  step stalls later to compile.
- **One graph per module.** Stock chooses its attention kernel through a context
  manager the compiler cannot follow, which made it cut the mask decoder into 18
  separately compiled pieces and memory attention into 8. Stock has a switch
  that lets PyTorch choose the kernel instead; with it each module compiles
  whole. The engine turns it on only while its compiled modules run, so stock
  run in the same process is unaffected.
- **One layout per size.** How a tensor is laid out in memory depends on how the
  batch was cut, and the compiler compiles again for every layout it meets. The
  engine hands compiled modules compact copies.
- **Two kinds of compiled code.** A CUDA graph records a module's kernels once
  and replays them with a single launch. With one or two rows, where launching
  is the limit, that is the difference between 130 and 220 frames per second at
  one camera. With more rows the GPU is the limit and replaying a graph costs it
  4 to 8%. The engine uses CUDA graphs for calls of one or two rows and plain
  tuned code for larger ones.

Prompting stays uncompiled: it is rare, and its shapes vary with the prompt.

Python's garbage collector is paused for the length of each step. If it runs
while compiled code is being recorded as a CUDA graph, it can free something
that touches the GPU, and CUDA abandons the recording. This happened at start-up
in some processes and never in others, depending on how many objects each had
made; with the collector paused it stopped.

### Encoder pieces

EdgeTAM's image encoder does the least GPU work per frame when it is given 4
frames at a time, but every call has to be launched. Up to 8 frames go in one
call; more are cut into pieces of 4.

### Getting frames in

Everything above starts from a frame that is already resized and on the GPU. A
live loop starts from a decoder, and what happens in between can cost more than
the tracking: resizing a 720p frame to the model's 1024 x 1024 on the CPU takes
16 ms, five times what the engine needs to track it.

So `step()` takes frames as the decoder gives them. The engine copies a round's
frames into one buffer of page-locked memory (memory the system will not move,
so the GPU can read it directly), uploads them in one go, and resizes them on
the GPU: 0.44 ms a frame at 720p. The resize follows PIL's step for step
(bicubic, width then height, rounded in between), so every pixel matches stock's
or differs by 1, and accuracy does not change ([results.md](results.md#raw-frames)).
`prepare()` still does stock's exact resize on the CPU, for when masks must
match stock's bit for bit.

Decoding stays with the caller. It is best done on a thread per camera, so that
the next round's frames are decoded while the GPU tracks this one
([results.md](results.md#a-live-loop)).

### Where the time goes

EdgeTAM on an RTX 4090, compiled, one object per camera, per camera per step:

| cameras | image encoder | memory attention | mask decoder | memory encoder | whole step | GPU idle |
|---|---|---|---|---|---|---|
| 1 | 2.57 ms | 1.65 ms | 1.16 ms | 0.71 ms | 4.5 ms | 14% |
| 8 | 1.48 ms | 0.96 ms | 0.33 ms | 0.27 ms | 2.9 ms | 3% |
| 64 | 1.41 ms | 0.98 ms | 0.27 ms | 0.27 ms | 3.0 ms | 3% |

The stages are timed one at a time, so at one camera they add up to more than
the whole step, where they overlap. At 8 cameras and beyond the GPU is busy
nearly all the time and the image encoder is half the cost. SAM 2.1 is different:
its memory holds 8 times as many tokens per frame, memory attention dominates,
and the GPU is already the limit at one camera, so batching cameras cannot raise
its frame rate ([results.md](results.md#sam-21)).

## Where the model code comes from

EdgeTAM is not published as a package, so on first use tamserve downloads its
code from GitHub at a pinned commit and checks it against a pinned SHA-256
digest before importing anything. It applies a two-line fix (upstream crashes
as soon as a second object is tracked) and keeps the result, with the weights,
in `~/.cache/tamserve`; `TAMSERVE_CACHE` moves that folder. SAM 2.1 weights come
from Meta's servers and are checked against pinned digests too; they are 149 to
856 MB. tamserve will not import code from a cache folder that another user owns
or that anyone can write to. The code is stored under the package name
`tamserve_sam2`, so it can share a process with Meta's own `sam2` package.

## How it was checked

- **Against stock, bit for bit.** One camera through the engine against the
  stock predictor on the same frames and prompts, on the CPU and the GPU, both
  models (`tests/test_engine.py`).
- **Objects joining, leaving and restarting.** Each object must be followed as
  if it were alone, whatever the others do (same file).
- **Accuracy on DAVIS 2017.** All 30 validation videos run as cameras that join
  3 frames apart, so every step mixes new and old objects. Each is compared with
  stock tracking that video alone and with the hand-drawn masks
  (`bench/accuracy.py`).
- **Speed against stock's best.** Stock compiled the way Meta's `vos_optimized`
  mode compiles SAM 2, one process per camera, sharing the GPU through NVIDIA
  MPS, counted exactly as the engine is (`bench/throughput.py`).

## Limits

- Forward only. Stock can also track backwards through a stored video.
- A new prompt for a followed object restarts it. Stock's refinement clicks,
  which adjust the current mask and keep the old memory, and prompts spread over
  several frames are not supported.
- The engine follows what it is told to follow. It does not find objects, and if
  one leaves the view its mask comes back empty until the model picks it up
  again, if it does.
- One engine drives one GPU and is called from one thread. A `Router` spreads
  cameras over the GPUs of one machine, a worker process on each
  ([results.md](results.md#through-a-router)).
- The timings are from one RTX 4090. The two tuning choices made from them, CUDA
  graphs for calls of one or two rows and encoder pieces of 4, are safe on any
  GPU but may not be the fastest choice on another.
