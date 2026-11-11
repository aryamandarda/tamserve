# The demo video

[`media/tamserve-demo.mp4`](media/tamserve-demo.mp4), 37 seconds. Three runs recorded on one RTX 4090
with EdgeTAM. Every mask, speed and memory figure on screen is read from the recording; only the last
card, which repeats the README's benchmark table, is typed in.

## What is on screen

| scene | what was recorded |
|---|---|
| Camera wall | Nine clips played as nine cameras with 15 objects between them, through one compiled engine: 208 frames/s in total, 23 per camera, 43 ms per step, 1.0 GB of GPU memory for the engine. The wall plays at the speed it was processed, which is a little slower than the 30 frames/s footage. |
| The same masks | One 300-frame clip with three objects, through the stock predictor and through tamserve with `exact=True`. Of 829 million mask pixels, 0 differ. Stock holds 3.75 to 4.0 GB of GPU memory, because it loads the whole clip first; tamserve holds 0.17 GB. |
| Objects come and go | One camera. A forklift is followed from the first frame; a second is added on frame 80, when it has driven into view; the first is dropped on frame 230. EdgeTAM's own predictor raises `Cannot add new object id ... after tracking starts` at the second step. |
| Benchmark card | The like-for-like table from the [README](../README.md#results), measured separately. |

## How it was recorded

- **Clips.** Each clip was cut to 300 frames of 1280x720. Sources slower than 30 frames/s are played at
  30, so some motion is up to a quarter faster than life.
- **Starting boxes.** A COCO detector (torchvision's Faster R-CNN) on each clip's first frame, checked
  by eye. tamserve does not find objects; this is the "something else supplies the box" step.
- **Camera wall.** `Engine(compile=True)`. Frames were decoded and placed on the GPU beforehand, as in
  the benchmarks, and every step was timed with the GPU synchronised before and after, masks copied to
  the CPU each time. Median step 43.2 ms, 95th percentile 43.7 ms; the first 19 steps, while the
  objects' memories fill, average 59 ms. GPU memory is PyTorch's peak allocation minus the 7.9 GB of
  frames waiting on the GPU. The same run with 12 objects instead of 15 took 36 ms per step: 250
  frames/s, 28 per camera.
- **The same masks.** Both sides uncompiled, bf16. Stock loads the clip and tracks it; tamserve is
  given one frame at a time. Differing pixels are counted at 1280x720 on every frame. Per frame, stock
  took 21.3 ms and tamserve 20.8 ms.
- **Objects come and go.** An uncompiled engine. The second forklift's box is the detector's, run on
  frame 80.
- **The picture.** Drawn afterwards from the saved masks and timings. The counters on the wall replay
  the recorded step times; masks were saved at half resolution for drawing.

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
