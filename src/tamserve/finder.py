"""SAM 3, Meta's model that outlines whatever you name, wrapped for the trackers.

This is the only file in tamserve that calls SAM 3. SAM 3 is Meta's, under Meta's
own licence, and none of it is in this package: its code is installed only when
you ask (`pip install "tamserve[sam3]"`) and imported only when a finder is built.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

# The word list SAM 3 reads prompts with. Its PyPI package leaves the file out, so it is
# fetched once from Meta's repository and checked against this SHA-256.
_VOCAB = ("https://raw.githubusercontent.com/facebookresearch/sam3/f1647e8d0c1d280991f4d1b7d974c2704537fe66"
          "/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
          "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a")


class Finder:
    """Everything you name in a frame, outlined by SAM 3.

    Needs Meta's SAM 3 code (`pip install "tamserve[sam3]"`) and its weights,
    which are gated: accept Meta's licence for facebook/sam3 on Hugging Face and
    log in with `hf auth login`, or pass a downloaded `checkpoint`.

    threshold: the confidence, 0 to 1, below which SAM 3's outlines are ignored.
    compile: a torch.compile mode such as "default", or None. Each prompt is
        encoded once here, not on every look as SAM 3 ships (113 ms a look down to
        95), and compiling brings a look to about 65 ms, with the same outlines up
        to rounding. Compiling takes a few minutes, done here.
    """

    def __init__(self, *, checkpoint: str | None = None, threshold: float = 0.5, compile: str | None = None,
                 device=None):
        try:
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ModuleNotFoundError as e:
            if (e.name or "").split(".")[0] != "sam3":
                raise  # SAM 3 is there and something it needs is not
            raise ModuleNotFoundError(
                'Tracker and PeopleTracker find things with Meta\'s SAM 3, which is not installed: '
                'pip install "tamserve[sam3]". Its weights are gated: accept the licence at '
                'https://huggingface.co/facebook/sam3 and run `hf auth login` once.', name="sam3") from None

        from tamserve.models import CACHE, _download

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        vocab = _download(_VOCAB[0], CACHE / "bpe_simple_vocab_16e6.txt.gz", _VOCAB[1])
        # Built on the CPU and moved: SAM 3's builder only knows "cuda", the first GPU.
        model = build_sam3_image_model(bpe_path=str(vocab), device="cpu", checkpoint_path=checkpoint,
                                       load_from_HF=checkpoint is None).to(self.device)
        if compile:
            # The decoder gains the most: it is hundreds of small GPU jobs, and the CPU
            # launching them was its limit (20 ms of a look).
            trunk = model.backbone.vision_backbone.trunk
            trunk.forward = torch.compile(trunk.forward, mode=compile, fullgraph=True)
            for part in (model.transformer.encoder, model.transformer.decoder, model.segmentation_head):
                part.forward = torch.compile(part.forward, mode=compile)
        self.model = model
        self.processor = Sam3Processor(model, device=self.device, confidence_threshold=threshold)
        self._words = {}  # prompt -> its encoding
        self.scores = []  # per frame of the latest call, each outline's confidence
        self.names = []   # the same, the prompt each outline was found for
        if compile:  # compile now, on a made-up picture, not during the first real looks
            noise = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
            for _ in range(4):
                Finder.__call__(self, [noise], [["person"]])

    @torch.inference_mode()
    def __call__(self, frames: list, prompts: list) -> list:
        """frames: H x W x 3 uint8 RGB arrays. prompts: for each frame, the list of things
        to find in it. Returns, for each frame, a list of bool masks; `scores` and `names`
        then hold each mask's confidence and the prompt it was found for."""
        found, self.scores, self.names = [], [], []
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            for frame, wanted in zip(frames, prompts):
                # The picture is encoded once, whatever the number of prompts: most of a look.
                state = self.processor.set_image(Image.fromarray(frame))
                masks, scores, names = [], [], []
                for prompt in wanted:
                    if prompt not in self._words:
                        self._words[prompt] = self.model.backbone.forward_text([prompt], device=self.device)
                    state["backbone_out"].update(self._words[prompt])
                    state["geometric_prompt"] = self.model._get_dummy_prompt()
                    state = self.processor._forward_grounding(state)
                    masks += list(state["masks"][:, 0].cpu())
                    scores += state["scores"].float().cpu().tolist()
                    names += [prompt] * (len(masks) - len(names))
                found.append(masks)
                self.scores.append(scores)
                self.names.append(names)
        return found


class PeopleFinder(Finder):
    """Every person in a frame, outlined by SAM 3: Finder, asked for "person".

    threshold: the confidence, 0 to 1, below which SAM 3's outlines are ignored.
    """

    def __init__(self, *, checkpoint: str | None = None, threshold: float = 0.4, compile: str | None = None,
                 device=None):
        super().__init__(checkpoint=checkpoint, threshold=threshold, compile=compile, device=device)

    def __call__(self, frames: list) -> list:
        """frames: H x W x 3 uint8 RGB arrays. Returns, for each, a list of bool masks."""
        return super().__call__(frames, [["person"]] * len(frames))
