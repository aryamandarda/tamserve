"""Meta's current SAM 2 tracker, as a second baseline (pip install sam2).

The benchmarks' default stock is the tracker in EdgeTAM's repository, which is how
tamserve loads every model. That code predates Meta's December 2024 changes to the
SAM 2 repository: its own compiled mode, and objects that can be added after
tracking starts. This builds that newer tracker with the same weights.
"""
from tamserve.models import MODELS, checkpoint


def build(model, device="cuda", compiled=False):
    """Meta's sam2 package's video predictor for a SAM 2.1 model; compiled by its own vos_optimized mode."""
    from sam2.build_sam import build_sam2_video_predictor

    if not model.startswith("sam2.1"):
        raise ValueError("Meta's sam2 package has the SAM 2.1 models only; EdgeTAM lives in its own repository")
    return build_sam2_video_predictor(MODELS[model][0], str(checkpoint(model)), device=device, vos_optimized=compiled)
