# Several GPUs

One engine drives one GPU. A `Router` spreads cameras over several, on one machine or many, and the [README](../README.md#several-gpus) has the basic loop. This page is the rest of the guide: what an update carries, what happens when a camera or a worker fails, and workers on other machines. How the router is built is in [how-it-works.md](how-it-works.md#several-gpus), and its measurements are in [results.md](results.md#two-gpus).

## Cameras and masks

The router reads the cameras itself, so you give it each camera's address instead of handing it frames. Opening streams by address needs OpenCV (`pip install opencv-python-headless`). You can also pass a function that returns your own reader. `start()`, `stop()` and `remove_camera()` work as on an engine.

Masks arrive in a compact form, a few KB each instead of a full grid of pixels. `mask.box` and `mask.area` are ready to use, and `mask.array()` gives the full True/False grid when you need it. Each update also carries the frame's number. If a person or a detector pointed at something on a frame the router has already moved past, pass that number as `start(..., frame=n)`: the worker starts the object on that frame and catches it up through the frames it kept.

On two RTX 4090s, with one object per camera, a router tracked 476 frames a second from 32 cameras of 720p video, against 241 on one GPU. Below what the GPUs can do, every camera is tracked at its full 30 frames a second and masks reach the router 20 ms after the frame was decoded.

## When something goes wrong

Cameras drop, machines restart, and a router keeps going through both. It also tells you, so a camera is never silently missing:

- **A camera whose stream drops** is opened again by its worker, waiting 1, 2, 4 seconds and so on, at most a minute, between tries. Its objects are kept, and tracking carries on when the picture returns. It keeps trying until it succeeds, unless you give it a limit with `add_camera(..., retry_for=600)`. A stream that stays connected but stops sending pictures counts as dropped after 10 seconds.
- **A worker that stops answering**, because it crashed or its machine went away, is noticed within 15 seconds. Its cameras move to the other workers, each object starting again from its last mask, and the router keeps trying to bring the worker back.

Each of these arrives in `results()` as an update with a `status`, and `error` says why:

```python
for update in router.results():
    if update.status:  # "disconnected", "reconnecting", "reconnected", "moved", "waiting" or "ended"
        alert(update.camera, update.status, update.error)
        continue
    ...  # masks, as before
```

The same events go to Python's standard `logging` under the name `tamserve`, so `logging.basicConfig(level=logging.INFO)` is enough to see them. For a dashboard or a health check, `router.status()` gives each camera's worker, its state and since when, frames per second and delay over the last 5 seconds, and the time of its last frame, and each worker's state, how busy it is and when it was last heard from.

## GPUs on other machines

On each machine with GPUs, start one worker per GPU. A worker builds and compiles its engine once, then waits for a router to connect:

```bash
python -c "import secrets; print(secrets.token_hex(32))" > token   # once; copy the same file to every machine
tamserve worker --gpu 0 --listen 0.0.0.0:7300 --token-file token --compile
tamserve worker --gpu 1 --listen 0.0.0.0:7301 --token-file token --compile
```

Your program then lists the workers' addresses, and nothing else in it changes. Addresses and this machine's own GPUs can be mixed:

```python
router = Router(workers=["10.0.0.11:7300", "10.0.0.11:7301", "10.0.0.12:7300"], token=open("token").read())
```

A few things to know:

- **The workers open the cameras**, so each camera's address has to be reachable from the worker machines, not just from your program. Cameras on a worker machine are given by address (`rtsp://...`), not as a reader function.
- **Keep workers on a private network or a VPN** such as Tailscale or WireGuard, or reach them through an SSH tunnel. The token keeps strangers from connecting: router and worker each prove they know it, without sending it, and it must be at least 32 characters. But the traffic is neither encrypted nor protected from tampering: anyone on the network path could read camera addresses, which often carry the camera's password, and could slip commands into a session. Messages are plain data, so neither side ever runs anything it is sent.
- **One router per worker at a time.** A second one is turned away with a message saying so. When the router disconnects, for example because your program restarted, or goes silent for 15 seconds, the worker forgets its cameras and waits for the next one. If its GPU keeps failing, the worker exits, so a service manager such as systemd can restart it.

Measured with two machines, one in California and one in Utah, each with an RTX 4090 and 16 RTSP cameras at 720p: a router on the California machine tracked 494 frames a second from all 32, which is 98% of what the two did on their own. Below capacity, every camera was tracked at its full 30 frames a second, and masks from the Utah machine arrived 21 ms after those from the local one.
