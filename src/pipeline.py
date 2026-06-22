"""Top-level runner for the splat quality evaluator.

Orchestrates the whole pass: discover and order the rendered frames, pair them
*consecutively*, load RAFT once, compute per-pair PSNR and flow magnitude, run
the popping detector, then save a machine-readable metrics.json and (if present)
hand the assembled results to report.py for plots + a human summary.

This module owns the two facts the lower layers deliberately don't know about:
the frame ordering, and the rule that only adjacent frames may be paired.

Run from the repo root:  python -m src.pipeline --frames frames --out results
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np

from src.flow import RaftFlow
from src.metrics import detect_popping, frame_flow_magnitude, psnr

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# Design decision: sort with a natural (number-aware) key rather than plain
# lexical sort, so the ordering is correct even if frames aren't zero-padded
# (frame10 after frame9, not after frame1). These renders ARE zero-padded so it
# is currently equivalent, but this removes a silent failure mode if they change.
def _natural_key(path: Path):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", path.name)]


# Design decision: discovery is its own function returning an ordered list, so
# the "what frames, in what order" decision is testable in isolation and the run
# loop can stay about metrics. Path comes in as an argument — nothing hardcoded.
def discover_frames(frames_dir: str | Path, limit: int | None = None) -> list[Path]:
    frames_dir = Path(frames_dir)
    if not frames_dir.is_dir():
        raise NotADirectoryError(f"Frames directory not found: {frames_dir}")
    files = sorted(
        (p for p in frames_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS),
        key=_natural_key,
    )
    if len(files) < 2:
        raise ValueError(f"Need >=2 frames to form a pair, found {len(files)} in {frames_dir}")
    return files[:limit] if limit else files


# Design decision: convert numpy/inf into JSON-safe values at save time only, so
# the in-memory results keep full-precision floats (incl. +inf for identical
# frames) for report.py, while the on-disk metrics.json stays valid JSON
# (inf -> null). Alternative: store inf as the string "inf" — rejected, null is
# the conventional "no finite value" and won't be mistaken for a real number.
def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if math.isinf(f) or math.isnan(f) else f
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# Design decision: the core pass is a plain function returning the results dict
# (and writing metrics.json) so it can be driven from the CLI *or* a test/notebook
# without argparse. RAFT is constructed once, before the loop, per the
# weights-loaded-once constraint; the previous frame is cached so each file is
# read from disk only once rather than twice.
def run(
    frames_dir: str | Path,
    out_dir: str | Path,
    reduce: str = "p99",
    pop_window: int = 5,
    pop_k: float = 4.0,
    pop_min_abs: float = 1.0,
    device: str | None = None,
    limit: int | None = None,
    verbose: bool = True,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = discover_frames(frames_dir, limit=limit)

    flow_model = RaftFlow(device=device)
    psnr_series: list[float] = []
    mag_series: list[float] = []
    pairs: list[dict] = []

    prev_rgb = RaftFlow.read_frame(frames[0])
    t0 = time.time()
    # Consecutive-only pairing: pair i is (frames[i], frames[i+1]); we never skip
    # an index, satisfying "do not compute flow on non-consecutive frame pairs".
    for i in range(len(frames) - 1):
        cur_rgb = RaftFlow.read_frame(frames[i + 1])
        p = psnr(prev_rgb, cur_rgb)
        flow = flow_model.flow(prev_rgb, cur_rgb)
        m = frame_flow_magnitude(flow, reduce=reduce)
        psnr_series.append(p)
        mag_series.append(m)
        pairs.append(
            {"index": i, "frame_a": frames[i].name, "frame_b": frames[i + 1].name,
             "psnr": p, "flow_magnitude": m}
        )
        prev_rgb = cur_rgb
        if verbose and (i % 10 == 0 or i == len(frames) - 2):
            print(f"  pair {i + 1}/{len(frames) - 1}  PSNR={p:6.2f}  flowmag={m:.4f}")

    popping = detect_popping(np.array(mag_series), window=pop_window, k=pop_k, min_abs=pop_min_abs)
    # Map pair-indices back to the frame the new state appears in (frame_b).
    popping_frames = [frames[i + 1].name for i in popping["popping_indices"]]

    results = {
        "frames_dir": str(Path(frames_dir)),
        "num_frames": len(frames),
        "num_pairs": len(pairs),
        "elapsed_sec": round(time.time() - t0, 1),
        "config": {"reduce": reduce, "pop_window": pop_window, "pop_k": pop_k,
                   "pop_min_abs": pop_min_abs, "device": str(flow_model.device), "limit": limit},
        "psnr": psnr_series,
        "flow_magnitude": mag_series,
        "pairs": pairs,
        "popping": popping,
        "popping_frames": popping_frames,
    }

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(_jsonable(results), indent=2))
    if verbose:
        finite = [p for p in psnr_series if math.isfinite(p)]
        print(f"\nProcessed {len(pairs)} pairs in {results['elapsed_sec']}s")
        print(f"PSNR  mean={np.mean(finite):.2f} dB  min={min(finite):.2f} dB")
        print(f"Flow magnitude ({reduce})  mean={np.mean(mag_series):.4f}  max={max(mag_series):.4f}")
        print(f"Popping events (k={pop_k}, min_abs={pop_min_abs}px): {len(popping_frames)} -> {popping_frames or 'none'}")
        print(f"Wrote {metrics_path}")
    return results


# Design decision: keep argparse confined to main() so run() stays import-clean;
# --frames is required (no default path baked in), everything else has an
# overridable default. The report step is an optional, guarded import so the
# pipeline is fully runnable today and auto-wires report.py once it exists.
def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="Gaussian Splat temporal quality evaluator")
    ap.add_argument("--frames", required=True, help="folder of sequential rendered frames")
    ap.add_argument("--out", default="results", help="output folder for metrics/plots")
    ap.add_argument("--reduce", default="p99", help="per-pair magnitude reducer: mean|max|p<N>")
    ap.add_argument("--pop-window", type=int, default=5, help="median window for the popping detector")
    ap.add_argument("--pop-k", type=float, default=4.0, help="popping sensitivity (lower = more sensitive)")
    ap.add_argument("--pop-min-abs", type=float, default=1.0, help="absolute flow floor (px) a pop must clear")
    ap.add_argument("--device", default=None, help="force 'cpu' or 'cuda' (default: auto)")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N frames")
    args = ap.parse_args(argv)

    results = run(args.frames, args.out, reduce=args.reduce, pop_window=args.pop_window,
                  pop_k=args.pop_k, pop_min_abs=args.pop_min_abs, device=args.device, limit=args.limit)

    try:
        from src import report  # built in the next step
    except ImportError:
        print("(report.py not present yet — wrote metrics.json only)")
    else:
        report.generate(results, args.out)
    return results


if __name__ == "__main__":
    main()
