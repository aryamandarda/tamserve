"""The models tamserve serves, loaded through EdgeTAM's codebase (a SAM 2 fork that
also carries the SAM 2.1 models). EdgeTAM is not published as a package: its code
is fetched at a pinned commit on first use, like the checkpoints."""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Pinned, not a branch: the stock predictor is the reference every batched result
# is checked against, so it must not move underneath us. The digest covers the
# files tamserve keeps from that commit, checked before any of it is imported.
EDGETAM_REF = "7711e012a30a2402c4eaab637bdb00a521302c91"
EDGETAM_SHA256 = "e8d535efee90b11d7f5dc6cdb4ec98f099db6fd07d2cbd037e034bc66e751a57"
# EdgeTAM's package is named sam2, like Meta's own SAM 2 package, and one process
# cannot import two packages of one name: tamserve stores its copy under this one.
PACKAGE = "tamserve_sam2"
CACHE = Path(os.environ.get("TAMSERVE_CACHE", Path.home() / ".cache" / "tamserve"))
EDGETAM_ROOT = Path(os.environ.get("TAMSERVE_EDGETAM_ROOT", CACHE / f"{PACKAGE}-{EDGETAM_REF[:7]}")).resolve()

_SAM21 = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
MODELS = {  # name -> (config, checkpoint URL and its SHA-256; None = ships with EdgeTAM)
    "edgetam": ("configs/edgetam.yaml", None, None),
    "sam2.1-tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", _SAM21 + "sam2.1_hiera_tiny.pt",
                    "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"),
    "sam2.1-small": ("configs/sam2.1/sam2.1_hiera_s.yaml", _SAM21 + "sam2.1_hiera_small.pt",
                     "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38"),
    "sam2.1-base-plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", _SAM21 + "sam2.1_hiera_base_plus.pt",
                         "a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5"),
    "sam2.1-large": ("configs/sam2.1/sam2.1_hiera_l.yaml", _SAM21 + "sam2.1_hiera_large.pt",
                     "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"),
}

# Upstream bug: .view() on a non-contiguous tensor crashes as soon as a second
# object is tracked, and batching is all about many objects.
_PERCEIVER_FIX = [
    (".expand(B, -1, -1).view(-1, 1, C)", ".expand(B, -1, -1).reshape(-1, 1, C)"),
    ("latents_2d.view(B, num_window, num_window, C)", "latents_2d.reshape(B, num_window, num_window, C)"),
]
# Only these leave the archive: the rest of the repository would sit on sys.path.
_KEEP = ("sam2/", "checkpoints/edgetam.pt", "LICENSE")


def _sha256(paths) -> str:
    h = hashlib.sha256()
    for path in paths:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def _tree_sha256(root: Path) -> str:
    """One digest over every kept file's relative path and bytes."""
    files = sorted(p for p in root.rglob("*") if p.is_file())
    h = hashlib.sha256()
    for p in files:
        h.update(p.relative_to(root).as_posix().encode() + b"\0" + _sha256([p]).encode())
    return h.hexdigest()


def _download(url: str, path: Path, sha256: str | None = None) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(f"{path.name}.{os.getpid()}.part")  # its own, if several processes start at once
        try:
            with urllib.request.urlopen(url, timeout=60) as r, part.open("wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            if sha256 and _sha256([part]) != sha256:
                raise RuntimeError(f"{url} did not match its pinned SHA-256; refusing to use it")
            part.replace(path)
        finally:
            part.unlink(missing_ok=True)
    return path


def _patched(perceiver: Path) -> bool:
    text = perceiver.read_text()
    return all(new in text for _, new in _PERCEIVER_FIX)


def _rename(src: Path) -> None:
    """sam2 -> PACKAGE: the folder, and every import and config target naming it."""
    (src / "sam2").rename(src / PACKAGE)
    for path in (src / PACKAGE).rglob("*"):
        if path.suffix in (".py", ".yaml"):
            text = re.sub(r"(?<![\w.])sam2\.(?=[A-Za-z_])", PACKAGE + ".", path.read_text())  # not sam2.1
            path.write_text(re.sub(r"^(\s*(?:from|import)\s+)sam2(?=\s|$)", r"\1" + PACKAGE, text, flags=re.M))
    # Stock claims Hydra's one process-wide configuration here, as Meta's package
    # does for itself; build_video_predictor reads the configs without it.
    (src / PACKAGE / "__init__.py").write_text("")


def _only_ours(path: Path) -> bool:
    """Whether `path` is this user's or root's and not writable by everyone. A guard
    against a shared or planted folder, not against users who share a group."""
    st = path.stat()
    return not hasattr(os, "getuid") or (st.st_uid in (0, os.getuid()) and not st.st_mode & 0o002)


def _fetch() -> None:
    """Download EDGETAM_REF, check its digest, apply the fix and the rename, and move
    the result to EDGETAM_ROOT."""
    EDGETAM_ROOT.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=EDGETAM_ROOT.name + ".", dir=EDGETAM_ROOT.parent))
    try:
        tar = _download(f"https://github.com/facebookresearch/EdgeTAM/archive/{EDGETAM_REF}.tar.gz", tmp / "source.tar.gz")
        with tarfile.open(tar) as t:
            # Regular files and folders under _KEEP only: no links, no absolute or ".." paths.
            keep = [m for m in t.getmembers() if (m.isfile() or m.isdir()) and "/" in m.name
                    and m.name.split("/", 1)[1].startswith(_KEEP)
                    and not m.name.startswith("/") and ".." not in Path(m.name).parts]
            t.extractall(tmp / "tree", members=keep, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
        (src,) = (tmp / "tree").iterdir()
        digest = _tree_sha256(src)
        if digest != EDGETAM_SHA256:
            raise RuntimeError(f"EdgeTAM {EDGETAM_REF[:7]} did not match its pinned digest (got {digest})")
        perceiver = src / "sam2" / "modeling" / "perceiver.py"
        text = perceiver.read_text()
        for old, new in _PERCEIVER_FIX:
            text = text.replace(old, new)
        perceiver.write_text(text)
        _rename(src)
        try:
            src.rename(EDGETAM_ROOT)
        except OSError:
            if not EDGETAM_ROOT.is_dir():  # otherwise another process finished first
                raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def edgetam_root() -> Path:
    """EdgeTAM's code at EDGETAM_REF, renamed and with the perceiver fix, fetched on first use."""
    if not EDGETAM_ROOT.is_dir():
        _fetch()
    perceiver = EDGETAM_ROOT / PACKAGE / "modeling" / "perceiver.py"
    if not (perceiver.is_file() and _patched(perceiver)):
        raise RuntimeError(f"{EDGETAM_ROOT} is not a copy tamserve prepared; delete it to fetch one")
    # The digest is checked when the code is fetched; after that the folder is trusted,
    # whoever fetched it, unless another user owns it or everyone can write to it.
    if not all(_only_ours(p) for p in (EDGETAM_ROOT, EDGETAM_ROOT / PACKAGE)):
        raise RuntimeError(f"{EDGETAM_ROOT} belongs to another user or can be written by everyone; "
                           "tamserve will not import code from it")
    return EDGETAM_ROOT


def checkpoint(model: str) -> Path:
    _, url, sha256 = MODELS[model]
    if url is None:
        return edgetam_root() / "checkpoints" / "edgetam.pt"
    return _download(url, CACHE / url.rsplit("/", 1)[1], sha256)


def build_video_predictor(device: str = "cuda", model: str = "edgetam"):
    """The stock video predictor: the reference every batched result is checked against."""
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {sorted(MODELS)}")
    root = edgetam_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # What stock's build_sam2_video_predictor composes through Hydra, read directly.
    cfg = OmegaConf.merge(OmegaConf.load(root / PACKAGE / MODELS[model][0]), OmegaConf.from_dotlist([
        f"model._target_={PACKAGE}.sam2_video_predictor.SAM2VideoPredictor",
        "model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
        "model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
        "model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        "model.binarize_mask_from_pts_for_mem_enc=true",
        "model.fill_hole_area=8",
    ]))
    OmegaConf.resolve(cfg)
    predictor = instantiate(cfg.model, _recursive_=True)
    predictor.load_state_dict(torch.load(checkpoint(model), map_location="cpu", weights_only=True)["model"])
    return predictor.to(device).eval()
