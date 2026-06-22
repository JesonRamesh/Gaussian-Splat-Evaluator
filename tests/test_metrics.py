"""Sanity checks for the metric functions (no RAFT / no files needed)."""
import numpy as np
import pytest

from src.metrics import (
    detect_popping,
    flow_magnitude_map,
    frame_flow_magnitude,
    psnr,
)


def test_psnr_identical_is_inf():
    img = np.full((8, 8, 3), 127, dtype=np.uint8)
    assert psnr(img, img) == float("inf")


def test_psnr_decreases_with_more_noise():
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    small = img.copy(); small[0, 0, 0] = 10
    large = img.copy(); large[:, :, :] = 50
    assert psnr(img, small) > psnr(img, large)


def test_psnr_shape_mismatch_raises():
    with pytest.raises(ValueError):
        psnr(np.zeros((4, 4, 3)), np.zeros((4, 5, 3)))


def test_flow_magnitude_map_pythagoras():
    flow = np.zeros((2, 2, 2), dtype=np.float32)
    flow[..., 0] = 3.0
    flow[..., 1] = 4.0
    assert np.allclose(flow_magnitude_map(flow), 5.0)


def test_frame_flow_magnitude_reducers():
    flow = np.zeros((10, 10, 2), dtype=np.float32)
    flow[..., 0] = 1.0  # every pixel magnitude == 1
    flow[0, 0, 0] = 100.0  # one outlier pixel
    assert frame_flow_magnitude(flow, "mean") < frame_flow_magnitude(flow, "max")
    assert np.isclose(frame_flow_magnitude(flow, "max"), 100.0)


def test_detect_popping_finds_injected_spike():
    rng = np.random.default_rng(0)
    series = 1.0 + 0.02 * rng.standard_normal(60)  # smooth low baseline + noise
    series[30] += 2.0  # inject a single popping event
    out = detect_popping(series, window=5, k=4.0)
    assert 30 in out["popping_indices"]


def test_detect_popping_silent_on_smooth_series():
    series = np.linspace(1.0, 2.0, 50)  # smooth ramp, no spikes
    out = detect_popping(series, window=5, k=4.0)
    assert out["popping_indices"] == []


def test_detect_popping_absolute_floor_ignores_subpixel_noise():
    # Mirrors the real 900-frame capture: flat ~7px baseline, sub-pixel noise,
    # no real pops. The min_abs floor must keep this silent (was 13 phantoms).
    rng = np.random.default_rng(2)
    series = 7.0 + 0.03 * rng.standard_normal(900)
    out = detect_popping(series, k=4.0)  # default min_abs floor
    assert out["popping_indices"] == []


def test_detect_popping_min_abs_is_configurable():
    rng = np.random.default_rng(3)
    series = 7.0 + 0.03 * rng.standard_normal(200)
    series[100] += 0.5  # half-pixel bump: real-ish but small
    assert 100 not in detect_popping(series, min_abs=1.0)["popping_indices"]
    assert 100 in detect_popping(series, min_abs=0.2)["popping_indices"]
