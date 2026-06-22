"""RAFT optical-flow wrapper.

Computes dense optical flow for a pair of *consecutive* rendered frames.
Everything downstream (flow magnitude, popping detection) consumes the
(H, W, 2) flow field produced here, so this module is the single place that
knows how to talk to RAFT. It does no metric computation itself.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large


# Design decision: a class (not free functions) so the expensive weight load
# happens exactly once in __init__ and is reused for every pair — directly
# satisfying the constraint "RAFT weights loaded once at startup, not per
# frame pair". Alternative considered: a module-level lazy singleton; rejected
# because an explicit object makes the lifetime obvious to the pipeline and
# keeps device choice testable.
class RaftFlow:
    def __init__(self, device: str | None = None) -> None:
        # Design decision: pick CUDA automatically but allow an explicit
        # override, so the same code runs on the laptop (CPU) and a GPU box
        # without a hardcoded device. Alternative: default to CPU always —
        # rejected as needlessly slow where a GPU exists.
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        # DEFAULT pins a specific pretrained checkpoint, which is what makes
        # runs reproducible: same weights -> same flow -> same numbers.
        self._weights = Raft_Large_Weights.DEFAULT
        self.model = raft_large(weights=self._weights, progress=False)
        self.model = self.model.to(self.device).eval()

    # Design decision: read with cv2 then convert BGR->RGB here, so the rest of
    # the module only ever sees RGB and the easy-to-forget OpenCV channel-order
    # bug is contained to one function. Alpha is dropped (IMREAD_COLOR) because
    # RAFT is a 3-channel model and the renders' alpha carries no motion.
    @staticmethod
    def read_frame(path: str | Path) -> np.ndarray:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read frame: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Design decision: normalise to [0,1] float32 at the wrapper boundary (as
    # the project notes require), then map to RAFT's trained [-1,1] range
    # internally. Doing the [-1,1] step here rather than reusing
    # weights.transforms() avoids that preset's optional *resize*, which would
    # rescale flow magnitudes and break reproducibility of the popping metric.
    def _preprocess(self, frame_rgb: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0  # [0,1]
        t = (t - 0.5) / 0.5  # -> [-1,1], RAFT's training normalisation
        return t.unsqueeze(0).to(self.device)

    # Design decision: pad (reflect) up to a multiple of 8 instead of resizing,
    # because RAFT's feature pyramid requires /8 dimensions but resizing would
    # alter pixel-distance flow values. Padding is later cropped away so the
    # returned flow matches the original frame size exactly. For these 1280x720
    # renders this is a no-op, but it keeps the wrapper resolution-agnostic.
    @staticmethod
    def _pad_to_multiple(t: torch.Tensor, mult: int = 8) -> tuple[torch.Tensor, int, int]:
        _, _, h, w = t.shape
        pad_h = (mult - h % mult) % mult
        pad_w = (mult - w % mult) % mult
        if pad_h or pad_w:
            t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), mode="replicate")
        return t, h, w

    # Design decision: take two RGB arrays and return a plain (H,W,2) numpy
    # float32 flow (A->B order, never reversed), so callers stay in numpy and
    # never touch torch. Alternative: return a tensor — rejected to keep the
    # torch dependency from leaking into metrics.py/report.py.
    def flow(self, frame_a_rgb: np.ndarray, frame_b_rgb: np.ndarray) -> np.ndarray:
        a, h, w = self._pad_to_multiple(self._preprocess(frame_a_rgb))
        b, _, _ = self._pad_to_multiple(self._preprocess(frame_b_rgb))
        # inference_mode (not just eval) guarantees no autograd state and, with
        # the fixed checkpoint, deterministic output for identical inputs.
        with torch.inference_mode():
            # RAFT refines iteratively; the final element is the best estimate.
            flow_pred = self.model(a, b)[-1]
        flow_pred = flow_pred[0, :, :h, :w]  # drop batch, crop off padding
        return flow_pred.permute(1, 2, 0).cpu().numpy().astype(np.float32)

    # Design decision: a path-based convenience that enforces consecutiveness is
    # the *pipeline's* job, not this function's — this one just loads two given
    # files. It exists so callers/tests don't repeat read_frame + flow.
    def flow_from_paths(self, path_a: str | Path, path_b: str | Path) -> np.ndarray:
        return self.flow(self.read_frame(path_a), self.read_frame(path_b))
