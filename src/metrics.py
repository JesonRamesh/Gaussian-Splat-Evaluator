"""Per-frame quality metrics.

Pure-numpy metric functions: consecutive-frame PSNR, optical-flow magnitude,
and a configurable Gaussian-popping detector. This module never touches torch
or the filesystem — it consumes the RGB frames and the (H,W,2) flow fields that
flow.py produces, so each function here is independently testable.

Vocabulary note (Gaussian-Splatting specific): "popping" is when a Gaussian at
a visibility boundary appears or disappears between two otherwise near-identical
frames. In a flow-magnitude time series it is an impulsive *spike* sitting on
top of the smooth baseline set by ordinary camera motion. The detector below is
therefore an outlier finder on that series, not a fixed-threshold alarm.
"""
from __future__ import annotations

import numpy as np


# Design decision: compute in float64 with a configurable data_range rather than
# assuming uint8 internally, so the MSE can't overflow and the function also
# works if frames are ever passed as floats. Identical frames return +inf (the
# mathematically correct PSNR) instead of a sentinel — callers decide how to
# display it. Alternative: clamp to a large finite number — rejected as it would
# silently distort any later averaging/plotting.
def psnr(frame_a: np.ndarray, frame_b: np.ndarray, data_range: float = 255.0) -> float:
    if frame_a.shape != frame_b.shape:
        raise ValueError(f"PSNR shape mismatch: {frame_a.shape} vs {frame_b.shape}")
    mse = np.mean((frame_a.astype(np.float64) - frame_b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(mse))


# Design decision: keep the full (H,W) per-pixel magnitude map as its own
# function (separate from the scalar reduction) so report.py can visualise
# *where* a pop happened, not just that one did. Magnitude is the L2 norm of the
# (dx,dy) vector per pixel.
def flow_magnitude_map(flow: np.ndarray) -> np.ndarray:
    if flow.ndim != 3 or flow.shape[-1] != 2:
        raise ValueError(f"Expected flow of shape (H,W,2), got {flow.shape}")
    return np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)


# Design decision: reduce each pair's magnitude map to ONE scalar with a
# configurable reducer, defaulting to a high percentile rather than the mean.
# Popping is *localised* — a small region spikes — so a frame-wide mean dilutes
# it, while a single-pixel max is dominated by RAFT noise; the 99th percentile
# is the sensitive-but-robust middle ground. Mean is still offered for a
# "global smoothness" reading. (This is a per-PAIR reduction, not the
# whole-sequence averaging that CLAUDE.md forbids.)
def frame_flow_magnitude(flow: np.ndarray, reduce: str = "p99") -> float:
    mag = flow_magnitude_map(flow)
    if reduce == "mean":
        return float(mag.mean())
    if reduce == "max":
        return float(mag.max())
    if reduce.startswith("p"):
        q = float(reduce[1:])  # e.g. "p99" -> 99th percentile
        return float(np.percentile(mag, q))
    raise ValueError(f"Unknown reduce mode: {reduce!r}")


# Design decision: a rolling MEDIAN (not moving average) gives the smooth
# baseline, because the median ignores the very spikes we are hunting whereas a
# mean would be pulled upward by them and mask the event. Edge frames reuse the
# nearest value (edge padding) so the series length is preserved.
def _rolling_median(x: np.ndarray, window: int) -> np.ndarray:
    half = window // 2
    xp = np.pad(x, half, mode="edge")
    return np.array([np.median(xp[i : i + window]) for i in range(len(x))])


# Design decision: detect popping as upper-tail outliers of the *detrended*
# series using a robust (median + MAD) z-score — i.e. a Hampel filter. We
# detrend first so a fast-but-smooth camera section doesn't trip the alarm, and
# we use MAD instead of std so the spikes don't inflate their own threshold.
# One-sided (upper) only: a pop always *raises* magnitude. Two configurable
# knobs, never hardcoded: `k` (relative sensitivity) and `min_abs` (an ABSOLUTE
# floor, in px). The floor exists because on a very clean baseline MAD shrinks
# to near-zero, so a purely relative k*sigma threshold would flag sub-pixel
# noise as "popping"; a real pop must clear both. Effective threshold is
# therefore max(k*sigma, min_abs). Alternative: relative-only — rejected, it
# manufactured 13 phantom events on the clean 900-frame capture.
def detect_popping(
    magnitudes: np.ndarray,
    window: int = 5,
    k: float = 4.0,
    min_abs: float = 1.0,
) -> dict:
    mags = np.asarray(magnitudes, dtype=np.float64)
    if mags.ndim != 1:
        raise ValueError(f"Expected a 1-D magnitude series, got shape {mags.shape}")
    if window % 2 == 0:
        window += 1  # median window must be odd to be centred

    baseline = _rolling_median(mags, window)
    residual = mags - baseline
    # 1.4826 rescales MAD to a std-equivalent for normal-ish noise.
    mad = np.median(np.abs(residual - np.median(residual)))
    sigma = 1.4826 * mad if mad > 0 else 0.0
    threshold = max(k * sigma, min_abs)  # absolute floor guards a clean baseline
    is_pop = residual > threshold

    return {
        "baseline": baseline,
        "residual": residual,
        "sigma": float(sigma),
        "threshold": float(threshold),
        "k": float(k),
        "min_abs": float(min_abs),
        "window": int(window),
        "is_pop": is_pop,
        "popping_indices": np.nonzero(is_pop)[0].tolist(),
    }
