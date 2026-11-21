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
kernels into fewer, tuned ones. Five things had to be solved for that to work on
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
- **One setting for float32 arithmetic.** Compiled code is tied to PyTorch's
  setting for multiplying float32 matrices (exact, or the faster TF32 format),
  and is compiled again the first time it runs after that setting changes. Other
  code changes it: importing SAM 3 turns TF32 on for the whole process. The
  engine puts the setting back as it was when it compiled for the length of each
  step, then returns the caller's.
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

## Several GPUs

Using a router is covered in [several-gpus.md](several-gpus.md). This is how it is built.

A `Router` places cameras on workers, one per GPU, and a camera stays where it is
put, because its objects' memory lives on that GPU. Each worker opens and decodes
its own cameras: moving decoded frames between processes cost more than tracking
them. The router sends commands and receives masks stored as runs of equal pixels,
a few KB each. A new camera goes to the worker with the most room, judged by how
busy each worker says its GPU was over the last second.

Workers on this machine are child processes, each seeing only its own GPU so that
all of them reuse the code the first compiled. A worker on another machine is a
`tamserve worker` reached over TCP. Before anything else, router and worker each
prove they know a shared token, by answering a random challenge with an HMAC-SHA256
of it, so the token never crosses the network. After that every message is plain
data: a JSON description and the raw bytes of any arrays in it. Only plain values
and tamserve's own types (Box, Clicks, Mask, Update) can be rebuilt from a message,
so neither side ever runs anything the other sent; a message is also limited in
size (64 MB per array, 256 MB in all). The router also compares its clock with
each worker's when it connects, so the times on updates are on its own clock
wherever the frame was decoded, and both sides check they speak the same version
of the protocol.

The router sends every worker a sign of life every 2 seconds, and each worker
reports how busy it is every second, from threads of their own so that a long
step does not delay them. Either side that hears nothing for 15 seconds treats
the other as gone: a worker then forgets that router's cameras and waits for the
next, which is what lets a program whose machine lost power start again. A router
moves a lost worker's cameras to the others, each object starting from its last
mask (so it loses its memory of the frames before), and keeps trying to start or
reach that worker again. A worker's front door handles several newcomers at once,
each with five seconds to prove it knows the token, so nobody can hold it shut.

Inside a worker, each camera's reader runs on a thread of its own. A network
stream or reader that fails is opened again with growing waits, and its objects
are kept. Frames are checked as they arrive and prompts before they are used, so
a broken camera or a wrong prompt is reported for that camera alone instead of
failing the step for every camera on the GPU. A worker whose steps fail three
times in a row decides its GPU is broken and stops, rather than reporting the same
error forever.

## On a Mac

An Apple-silicon Mac has three processors on one chip: the CPU, a GPU, and the
Neural Engine, which is built only to run neural networks. PyTorch uses the GPU
and leaves the Neural Engine idle.

On an NVIDIA card one camera leaves the GPU waiting most of the time, and
batching cameras fills it. A Mac's GPU is slower, and one camera already fills
it: on an M4 Max each object costs 18 ms of GPU time a step, whether there is
one or eight ([results.md](results.md#on-a-mac)). Batching buys nothing there.
What a Mac has to spare is the Neural Engine.

**The image encoder runs on the Neural Engine.** It is the one stage that needs
only the frame, so it can run while the GPU does something else. It is the
model's own encoder with the same weights, converted to Core ML, Apple's format
and the only way onto the Neural Engine (`src/tamserve/apple/`). The Neural
Engine works in 16-bit numbers, its features come out 2% different from
PyTorch's, and that costs 0.002 of accuracy on DAVIS 2017.
`Engine(predictor, neural_engine=False)` keeps everything on the GPU, and so does
a macOS older than 15, which cannot load the encoder.

**The encoder needs a head start.** Handed one frame at a time, the two
processors take turns and little is gained. They work at once only if the
Neural Engine has a frame while the GPU is still following objects on the one
before.

A live camera gives it that by itself: its next frame arrives on the camera's
clock, usually before the GPU is done. So the engine can be told of a frame the
moment it arrives, from the thread that reads the camera (`start_encoding()`),
and a Router's workers do. One camera at 30 frames a second then has every
frame tracked, against 24 of them, with its masks sooner. When tracking is
behind its cameras, a worker takes each camera's newest frame that is already
encoded, not the very newest, so the GPU does not wait for the Neural Engine.
A frame that is passed over before its turn is never encoded.

A recorded video has no clock, but its next frame is already there, so `step()`
takes it as `next_frames`, and one camera goes from 24 frames a second to 40.

Either way, a frame given early is recognised by its pixels when it comes back.
If a different frame comes, that one is encoded and followed.

**The other stages stay on the GPU.** Following an object is a chain: each stage
needs the one before, and the next frame needs the last. A second processor
could only help by running a stage as exactly as the GPU does, and the Neural
Engine runs the memory encoder no faster than the GPU and 9% off.

The GPU runs in 16-bit numbers on a Mac too (fp16 there, bf16 on NVIDIA).

## Following people

The engine follows what it is pointed at. `PeopleTracker`
(`src/tamserve/people.py`) does the pointing for people, with a second model.

SAM 3 is Meta's model that outlines everything in a picture that matches a word
or a phrase, here "person". Running it on one frame is called a look below. SAM 3
also has a video tracker of its own, which carries its outlines through a
recorded video; that tracker is what `PeopleTracker` is measured against in
[results.md](results.md#people).

**A look every few frames.** A look is slow and following is cheap, so a camera
is looked at once every `look_every` frames and followed in between. The cost of
a frame is about L / k + F: a look (L) shared over the k frames between looks,
plus following (F). On the home videos that is 65 / 15 + 4.4, about 8.9 ms, where
SAM 3's own tracker takes 126 ms.

**A leaner look.** As SAM 3 ships, a look at a crowded street takes 113 ms on an
RTX 4090. Two changes bring it to 68 ms with the same outlines. The word "person" is encoded
once instead of on every look (95 ms). And the decoder, which is hundreds of
small GPU jobs and was limited by the CPU launching them, is compiled (68 ms).
The image encoder keeps the GPU busy by itself and gains nothing from compiling;
it is three quarters of what is left.

**Keeping the same number on a person.** On a look frame the engine first
predicts where everyone it follows is. SAM 3's outlines are then matched to
those predictions by overlap, the best overlap first and one to one. A matched
outline is that person; an unmatched one is someone new. Both start again from
SAM 3's outline, which fits better than a mask carried for 15 frames. A person
no outline matched is followed for `hold` more frames, out of the answer, and
has their number back if a later look matches them.

**Two thresholds.** An outline needs a confidence of 0.7 to start a new person
but only 0.4 to confirm one who is already followed, and the confident outlines
choose first. In a crowd this was worth 2.7 HOTA with a look on every frame. In
homes it changed nothing.

**A camera's first frame is always looked at.** After that, looks are rationed to
about cameras / `look_every` per step, the most overdue camera first, so they do
not all land on one step. Making new cameras wait their turn as well cost 2.9
HOTA on the home videos: people on a camera's first frames went unseen.

**Cameras can have their own `look_every`.** `look_every={"plaza": 1}` looks at
that camera on every frame and leaves the rest at 15. The ration is then the sum
of one over each camera's interval, and the cameras that look most often go
first, so a camera set to 1 never waits behind the others. `Tracker` takes the
same dict.

**Looks are not batched.** SAM 3's image encoder is pure arithmetic on a full
GPU, so two frames in one call take as long as two calls.

## Following what you name

`Tracker` (`src/tamserve/tracker.py`) is the same pairing for anything that can
be put in words: "dog", "the red mug", "forklift". Each camera has one or more
prompts, SAM 3 looks at the camera every `look_every` frames, 3 by default, and
the engine follows what it outlined in between. People have their own class
because their settings were tuned on people and score better there; `Tracker`
is for everything else, and for people next to other things.

**Off NVIDIA, SAM 3 comes through other code.** Meta's SAM 3 package names
the NVIDIA device throughout and does not import on a Mac. Hugging Face's
`transformers` has the same model with the same weights, and `Finder` uses it on
an Apple GPU or a CPU. Called the way its documentation shows, it is not the
same finder: on 152 questions it outlined 14 things that Meta's code scored
under the threshold. Three differences were behind that, and `Finder` removes
each one ([results.md](results.md#the-same-sam-3)):

- Meta's code puts every prompt through its box encoder, even a prompt of words
  alone, and the token that comes out joins the words. Hugging Face's skips the
  step when there is no box. `Finder` gives it one box marked "ignore", which
  makes it run.
- The two resize a picture to SAM 3's 1,008 pixels differently. `Finder` resizes
  as Meta's code does.
- Meta's code enlarges a mask's scores to the frame and then cuts them into a
  mask. Hugging Face's cuts after squashing them, which moves the edges.
  `Finder` does it Meta's way.

**The rules are SAM 3's own.** With SAM 3 looking at every single frame, a
first version found more than SAM 3's video tracker did and was right half as
often: SAM 3 on one frame reports many things that are not there, and its video
tracker filters them out. Its rules for that are in its code
(`sam3_video_base.py`), and `Tracker` follows them:

- An outline that overlaps a more confident one of the same look by a tenth is
  a repeat of it, and is dropped.
- An outline and a followed object are the same object from a tenth of overlap.
  For people the bar is three tenths.
- An outline needs a confidence of 0.7 to start something new and 0.5 to
  confirm something already followed.
- A followed object starts again from SAM 3's outline only when SAM 3 scores
  that outline 0.8 or more. Otherwise it keeps the mask the engine carried,
  which holds a thin or half-hidden object better than a doubtful outline does.
- A new object is on trial for its first 15 frames. If more than half of the
  looks in that time do not see it, it was a false find and is removed.
- After its trial an object is never hidden because a look missed it. It stays
  in the answer for as long as the engine has a mask for it, and is forgotten
  once no look has seen it for `hold` frames.

**Several prompts, one encoding.** Three quarters of a look is SAM 3's image
encoder, and it does not depend on the prompt. A camera with five prompts has
its picture encoded once and each prompt answered from that, where SAM 3's video
tracker, which takes one prompt a session, would run five times over.

**Answers at once.** SAM 3's video tracker holds every answer back 15 frames,
which is how an object that fails its trial never appears in its output.
`Tracker` answers each frame as it comes, so a false find can show for up to 15
frames before it is removed. `bench/named.py --late 15` scores both.

**Cameras that do not move.** With `still=True` a camera's picture is compared
with the one it last looked at, as the mean colour of each 16-pixel block. While
no block has moved by more than 8 grey levels the camera looks every 30th frame
only, and on frames that did not change since the last one followed, its objects
are not followed either: their masks stay as they were. A camera with a new
object on trial keeps looking. On a hand-held camera every frame changes, so the
setting does nothing there.

**What it does not match.** A look judges one frame alone. SAM 3's video tracker
scores every frame with the objects it already follows as context, and that is
where the last few points of accuracy are: looking more often does not recover
them, because recall rises and precision falls by as much.

## Where the code is

| path | what is there |
|---|---|
| [`src/tamserve/engine.py`](../src/tamserve/engine.py) | the engine: batching, per-object memory, compiling |
| [`src/tamserve/router.py`](../src/tamserve/router.py), [`worker.py`](../src/tamserve/worker.py), [`wire.py`](../src/tamserve/wire.py) | cameras spread over several GPUs: placing them, the worker on each GPU, and how router and workers on other machines talk |
| [`src/tamserve/tracker.py`](../src/tamserve/tracker.py), [`people.py`](../src/tamserve/people.py) | following what you name on the engine, and deciding what to keep from each look; the same for people |
| [`src/tamserve/finder.py`](../src/tamserve/finder.py) | the look itself: the one file that calls SAM 3 |
| [`src/tamserve/models.py`](../src/tamserve/models.py) | fetching, checking and loading EdgeTAM and SAM 2.1 |
| [`src/tamserve/reference.py`](../src/tamserve/reference.py), [`compare.py`](../src/tamserve/compare.py) | the stock predictor as a reference, and mask comparison |
| [`tests/`](../tests/) | bit-for-bit checks against stock; objects joining, leaving and restarting |
| [`bench/`](../bench/) | the scripts behind every number here |

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
- **On a Mac.** The Neural Engine path against the plain engine, bit for bit,
  with the model's own encoder standing in for the Neural Engine; and on the real
  one, by how far the masks overlap (`tests/test_apple.py`, `bench/mac.py`).
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
- `Tracker` and `PeopleTracker` find what they are asked for with SAM 3. They
  number things per camera, run on one engine, so one GPU, and their looks are
  not batched across cameras.
- One engine drives one GPU and is called from one thread. A `Router` spreads
  cameras over several GPUs, on one machine or many, a worker on each
  ([results.md](results.md#through-a-router)).
- Router and remote workers authenticate each other but do not encrypt or
  protect their traffic; run them on a private network or a VPN.
- A camera that moves to another worker starts its frame numbers again from 0,
  and its objects start again from their last masks.
- The timings are from one RTX 4090. The two tuning choices made from them, CUDA
  graphs for calls of one or two rows and encoder pieces of 4, are safe on any
  GPU but may not be the fastest choice on another.
- The lead over stock comes from work the engine skips, so it should carry over
  to other GPUs, but it has only been measured on that one card. GPUs without
  hardware support for bf16, such as the T4 that Colab gives for free, are far
  slower than their age suggests: there the quickstart notebook tracks 7 frames
  a second, and one look by SAM 3 takes 3 seconds.
