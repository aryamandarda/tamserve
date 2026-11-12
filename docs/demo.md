# The demo video

[`media/tamserve-demo.mp4`](media/tamserve-demo.mp4), 34 seconds. Three runs recorded through one
`Router` on two RTX 4090s, with EdgeTAM compiled. Every mask, frame rate, delay and status on screen is
read from the recordings. Only the GPU memory scene repeats numbers measured separately, from the
[README](../README.md#results).

## What is on screen

| scene | what was recorded |
|---|---|
| Camera wall | 18 live cameras (nine clips and their mirror images) with 30 objects between them, nine cameras and 15 objects per GPU. Each camera streams at 30 frames/s, more than two GPUs can track, so every step takes each camera's newest frame. Over the 5.5 seconds shown: 360 frames/s in total (351 to 369), 20 per camera, and a median of 67 ms from a frame being decoded to its masks reaching the program (95th percentile 81 ms). |
| GPU memory | The README's figures. Eight cameras at once: 8.6 GB with the model's own code, which loads one copy per camera, and 2.2 GB with tamserve. One camera at frame 800: 9.8 GB against 0.28 GB. |
| Start and stop tracking | One live camera. The blue forklift is followed from frame 0. The orange forklift's box was drawn on frame 80 but only sent at frame 100, as a slow detector would: `router.start(camera, "orange", box, frame=80)`. The worker started it on frame 80 from the recent frames it keeps and caught up; its first mask arrived around frame 104. The blue forklift is stopped at frame 230. Nothing else on the camera is disturbed. |
| Failures | Eight live cameras, four per GPU, 224 frames/s in total. At 2.5 s one camera's stream stops for 2.5 seconds. Its worker keeps retrying, and the camera reconnects at 5.7 s with the objects it had. At 4.0 s the worker process on GPU 1 is killed with SIGKILL. 0.4 s later its four cameras are running on GPU 0, each object restarting from its last mask, and with one GPU serving all eight the total settles at 200 frames/s. |

## How it was recorded

- **Cameras.** Each clip was cut to 300 frames of 1280x720 and played as a live camera: frame k is due
  k/30 seconds after the start, whether or not anyone is reading, and frames nobody read in time are
  skipped. Sources slower than 30 frames/s are played at 30, so some motion is up to a quarter faster
  than life. Every camera logged which frame it sent and when, so each mask is drawn on the exact
  picture it was computed from.
- **Starting boxes.** A COCO detector (torchvision's Faster R-CNN) on each clip's first frame, checked
  by eye, and on frame 80 for the orange forklift. tamserve does not find objects; this is the
  "something else supplies the box" step.
- **Serving.** `Router(workers=["cuda:0", "cuda:1"], compile=True)`: one worker process per GPU, each
  opening and decoding its own cameras. The wall is shown from 4 s into its run, once every camera has
  been tracked for at least two seconds.
- **The picture.** Drawn afterwards from the saved updates: masks, statuses and when each arrived. The
  frame-rate counters count the updates that reached the program in the previous second.

## Footage

Nine clips from [Pexels](https://www.pexels.com), used under the
[Pexels licence](https://www.pexels.com/license/), which allows free use and modification.

| camera | source |
|---|---|
| pond | [Ducks in a pond](https://www.pexels.com/video/ducks-in-a-pond-1343236/) |
| park | [Dogs running on grass](https://www.pexels.com/video/dogs-running-on-grass-12363705/) |
| street | [Backside of people walking on a street sidewalk](https://www.pexels.com/video/backside-of-people-walking-on-a-street-sidewalk-2836342/) |
| warehouse | [Forklifts moving boxes in warehouse interior](https://www.pexels.com/video/forklifts-moving-boxes-in-warehouse-interior-37651089/) |
| pasture | [Horse running on grassland](https://www.pexels.com/video/horse-running-on-grassland-3987730/) |
| pitch | [Soccer players training](https://www.pexels.com/video/soccer-players-training-6077710/) |
| lake | [Ducks swimming and feeding in serene pond](https://www.pexels.com/video/ducks-swimming-and-feeding-in-serene-pond-32812147/) |
| cycle path | [Man riding bike on country road](https://www.pexels.com/video/man-riding-bike-on-country-road-13742199/) |
| paddock | [A horse in running in the farm](https://www.pexels.com/video/a-horse-in-running-in-the-farm-6264718/) |

The clips themselves are not in this repository, only the video made from them.
