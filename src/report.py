"""Report assembly: turn the results dict from pipeline.run() into plots and a
human-readable summary JSON.

This module is a pure *consumer* of results — it computes no metrics of its own,
it only visualises and summarises what pipeline already produced. Every plot is
strictly per-frame (never a single sequence-averaged value), because averaging
would hide the localised popping spikes this whole tool exists to find.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: write PNGs, never try to open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


# Design decision: derive the x-axis from the real frame numbers parsed out of
# frame_b filenames, not a 0..N counter, so a spike on the plot points straight
# at the frame you'd open in the viewer. Falls back to an index if names aren't
# numeric. Alternative: plot pair-index — rejected as it forces mental arithmetic
# to find the offending frame.
def _frame_axis(results: dict) -> np.ndarray:
    xs = []
    for i, pair in enumerate(results["pairs"]):
        stem = Path(pair["frame_b"]).stem
        xs.append(int(stem) if stem.isdigit() else i)
    return np.asarray(xs)


# Design decision: build the human summary as plain finite stats, dropping the
# +inf PSNR of any identical-frame pair before aggregating, so one repeated frame
# can't poison the mean/min. Kept separate from plotting so the numbers can be
# consumed without matplotlib.
def _summary(results: dict) -> dict:
    psnr = np.array([p for p in results["psnr"] if p is not None and math.isfinite(p)], dtype=float)
    mag = np.asarray(results["flow_magnitude"], dtype=float)
    pop = results["popping"]
    n_pop = len(results["popping_frames"])
    return {
        "num_frames": results["num_frames"],
        "num_pairs": results["num_pairs"],
        "psnr_db": {"mean": round(float(psnr.mean()), 2), "median": round(float(np.median(psnr)), 2),
                    "min": round(float(psnr.min()), 2), "max": round(float(psnr.max()), 2)},
        "flow_magnitude_px": {"reduce": results["config"]["reduce"],
                              "mean": round(float(mag.mean()), 3), "median": round(float(np.median(mag)), 3),
                              "min": round(float(mag.min()), 3), "max": round(float(mag.max()), 3),
                              "frame_to_frame_change": round(float(np.abs(np.diff(mag)).mean()), 3)},
        "popping": {"count": n_pop, "frames": results["popping_frames"],
                    "k": pop["k"], "min_abs_px": pop.get("min_abs"), "window": pop["window"],
                    "threshold_px": round(float(pop["threshold"]), 3), "sigma_px": round(float(pop["sigma"]), 3)},
        "verdict": (f"{n_pop} popping event(s) detected at this spatial scale; "
                    f"baseline flow {round(float(np.median(mag)), 2)} px"),
    }


# Design decision: one stacked figure (flow on top, PSNR below) sharing the frame
# axis, so popping spikes and PSNR dips line up vertically and the eye can confirm
# they coincide. The detector's own baseline and alarm line are drawn on the flow
# panel so the reader sees *why* a point was (or wasn't) flagged, not just the
# verdict. Alternative: two separate files — rejected, the cross-read is the point.
def _plot(results: dict, path: Path) -> None:
    x = _frame_axis(results)
    mag = np.asarray(results["flow_magnitude"], dtype=float)
    psnr = np.array([p if (p is not None and math.isfinite(p)) else np.nan for p in results["psnr"]], dtype=float)
    pop = results["popping"]
    baseline = np.asarray(pop["baseline"], dtype=float)
    alarm = baseline + pop["threshold"]  # the line a residual must clear to be a pop
    pop_idx = pop["popping_indices"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax1.plot(x, mag, color="#1f77b4", lw=1.0, label=f"flow magnitude ({results['config']['reduce']}/pair)")
    ax1.plot(x, baseline, color="#ff7f0e", lw=1.2, ls="-", label="rolling-median baseline")
    ax1.plot(x, alarm, color="#d62728", lw=1.0, ls="--", label=f"popping threshold (+{pop['threshold']:.2f}px)")
    if pop_idx:
        ax1.scatter(x[pop_idx], mag[pop_idx], color="#d62728", zorder=5, s=40, marker="x", label="popping event")
    ax1.set_ylabel("optical flow magnitude (px)")
    ax1.set_title("Temporal motion & Gaussian popping — per consecutive-frame pair")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(x, psnr, color="#2ca02c", lw=1.0, label="consecutive-frame PSNR")
    ax2.axhline(np.nanmedian(psnr), color="grey", ls=":", lw=1.0, label=f"median {np.nanmedian(psnr):.1f} dB")
    for i in pop_idx:
        ax2.axvline(x[i], color="#d62728", alpha=0.3, lw=1.0)
    ax2.set_ylabel("PSNR (dB)")
    ax2.set_xlabel("frame number")
    ax2.set_title("Temporal smoothness (higher = consecutive frames more similar)")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# Design decision: a single generate(results, out_dir) entry point exactly
# matching the contract pipeline.main() already calls, so the pipeline auto-wires
# to it with no further change. Returns the summary dict for convenience/testing.
def generate(results: dict, out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = _summary(results)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _plot(results, out_dir / "report.png")
    print(f"Report written: {out_dir/'report.png'}, {out_dir/'summary.json'}")
    print(f"Verdict: {summary['verdict']}")
    return summary
