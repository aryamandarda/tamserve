# The demo video

[`media/tamserve-demo.mp4`](media/tamserve-demo.mp4), 32 seconds. Recorded through a `Router` with one worker
on one RTX 4090, with EdgeTAM compiled. Every mask, frame rate, delay and status on screen is read from the
recordings. Only the GPU memory scene repeats numbers measured separately, from the
[README](../README.md#results).

## What is on screen

| scene | what was recorded |
|---|---|
| Camera wall | Six live cameras with 9 objects between them: a welder, a cat, a jet and the marshal guiding it in, three people at an office table, a pig, and a barista. Each camera streams at 30 frames/s. Over the 5.5 seconds shown, every camera was tracked at 30 frames/s and never fewer than 29 in any second: 180 frames/s in total, with a median of 32 ms from a frame being decoded to its masks reaching the program (95th percentile 38 ms). |
| GPU memory | The README's figures. Eight cameras at once: 8.6 GB with the model's own code, which loads one copy per camera, and 2.2 GB with tamserve. One camera at frame 800: 9.8 GB against 0.28 GB. |
| Start and stop tracking | One live camera, recorded for the 0.2.0 demo. The blue forklift is followed from frame 0. The orange forklift's box was drawn on frame 80 but only sent at frame 100, as a slow detector would: `router.start(camera, "orange", box, frame=80)`. The worker started it on frame 80 from the recent frames it keeps and caught up; its first mask arrived around frame 104. The blue forklift is stopped at frame 230. Nothing else on the camera is disturbed. |
| A camera drops | The same six cameras. At 2.5 s camera 4's stream stops for 2.5 seconds. Its worker keeps retrying, and the camera reconnects at 5.6 s with its three people still followed. The other five cameras carry on at 30 frames/s. |

## How it was recorded

- **Choosing the clips.** Each camera is a fixed or slowly panning shot where every moving thing is in
  view from the first frame to the last, so nothing on screen goes unmasked. Every clip was tracked end to
  end on a laptop before recording, and its masks were checked at 640x360 on eight frames spread across
  the clip.
- **Cameras.** Each clip was cut to 300 frames of 1280x720 and played as a live camera: frame k is due
  k/30 seconds after the start, whether or not anyone is reading, and frames nobody read in time are
  skipped. The office and café clips were filmed at 25 frames/s and are played at 30, so their motion is a
  fifth faster than life; faster sources were brought down to 30. The café clip has 80 pixels cut from its
  right edge, where someone outside the shot pokes an elbow in. Every camera logged which frame it sent
  and when, so each mask is drawn on the exact picture it was computed from.
- **Starting boxes.** A COCO detector (torchvision's Faster R-CNN) on each clip's first frame, checked by
  eye, and on frame 80 for the orange forklift. tamserve does not find objects; this is the "something
  else supplies the box" step.
- **Serving.** `Router(workers=["cuda:0"], compile=True)`: one worker process on the GPU, opening and
  decoding all six cameras. The wall is shown from 4 s into its run; every camera's first mask had arrived
  by 0.6 s.
- **The picture.** Drawn afterwards from the saved updates: masks, statuses and when each arrived. Each
  camera's frame rate counts its updates over the previous two seconds; the total counts the previous
  second.

## The naming clip

[`media/tamserve-name-it.mp4`](media/tamserve-name-it.mp4), 17 seconds: a `Tracker` on one live camera at a time, on one
RTX 4090, with tamserve 0.3.0 as installed from PyPI. SAM 3 finds what is named and EdgeTAM follows it. Every mask, name,
frame rate and delay on screen is read from the recordings.

| scene | what was recorded |
|---|---|
| Warehouse | The camera starts with nothing asked of it. `tracker.prompts = ["forklift"]` is set 0.7 s in, and the forklift in view has its first mask 0.16 s later. The second forklift drives into view and has its first mask at 1.13 s, with nobody pointing at it. "person" is added at 4.5 s, and both drivers have masks 0.12 s later. The camera is tracked at 23 frames/s with one prompt and 18 with two, with a median of 37 and 31 ms from a frame arriving to its masks. |
| Office | The same on another camera: "person" at 0.7 s, "glasses" at 3.0 s and "plant" at 5.2 s. Three people, two pairs of glasses and a plant each have a mask within 0.16 s of being asked for. The camera is tracked at 22, 19 and 15 frames/s as it is asked for one, two and three things. |

- **The camera.** Each clip is played as a live camera, as above: frame k is due k/30 seconds after the start, a thread
  decodes it, and the loop takes the newest frame each round. Frames that arrive while the GPU is busy are skipped, which
  is why the camera sends 30 frames a second and about 20 are tracked.
- **The words.** The typing is drawn afterwards. Each word finishes on screen at the moment `tracker.prompts` was set in
  the recording.
- **Not compiled.** A compiled engine pauses the first few times a tracker meets a new number of objects
  ([Limits](../README.md#limits)), and those pauses would have fallen inside these few seconds.
- **The picture.** Each mask is drawn on the frame it was computed from. The frame rate counts the rounds finished in
  the previous second.

## Footage

Seven clips from [Pexels](https://www.pexels.com), used under the
[Pexels licence](https://www.pexels.com/license/), which allows free use and modification.

| camera | source |
|---|---|
| workshop | [Man Doing Welding Work](https://www.pexels.com/video/man-doing-welding-work-14314645/) |
| garden | [Close Up Video of a Cat on the Grass](https://www.pexels.com/video/close-up-video-of-a-cat-on-the-grass-7727397/) |
| airport | [Man Assisting An Airplane In Runway](https://www.pexels.com/video/man-assisting-an-airplane-in-runway-856984/) |
| office | [People Working in the Office](https://www.pexels.com/video/people-working-in-the-office-6914684/) |
| pig pen | [A Pig Covered in Mud](https://www.pexels.com/video/a-pig-covered-in-mud-9179082/) |
| café | [A Man Making Coffee](https://www.pexels.com/video/a-man-making-coffee-7171097/) |
| warehouse | [Forklifts moving boxes in warehouse interior](https://www.pexels.com/video/forklifts-moving-boxes-in-warehouse-interior-37651089/) |

The clips themselves are not in this repository, only the video made from them.
