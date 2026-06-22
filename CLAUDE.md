# CLAUDE.md — Splat Quality Evaluator

## What this project is
A temporal consistency and quality evaluation tool for Gaussian Splat
render sequences. Input: a folder of rendered frames from a trained
Gaussian Splat. Output: per-frame metrics, plots, and a summary report.

## Core metrics to implement
1. PSNR between consecutive frames (temporal smoothness proxy)
2. Optical flow magnitude via RAFT (measures unwanted motion between
   frames that should be nearly identical)
3. Gaussian popping detector (high-frequency spikes in flow magnitude
   across the sequence = popping event)

## Stack
- Python 3.10+
- RAFT (already familiar — same as EdgeFlow project)
- OpenCV for frame I/O
- matplotlib for plots
- numpy for metric computation
- No training required — runs on CPU or GPU, inference only

## Input source
Rendered fly-through frames from the existing RoboScene+ Gaussian Splat
(.ply loaded in a viewer and rendered as image sequence). Frames should
be sequential viewpoints along a smooth camera path.

## Folder structure
Gaussian-Splat-Evaluator/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── frames/              # input rendered frames (not committed)
├── results/             # output metrics, plots, report (not committed)
├── src/
│   ├── flow.py          # RAFT wrapper — optical flow between frame pairs
│   ├── metrics.py       # PSNR, flow magnitude, popping detector
│   ├── report.py        # assembles plots and summary JSON
│   └── pipeline.py      # top-level runner, called from CLI
└── tests/
    └── test_metrics.py  # basic sanity checks on metric functions

## Key constraints
- Modular: each src/ file does one thing and is testable independently
- No hardcoded paths — all paths via CLI args or config
- Results must be reproducible: same frames = same numbers every run
- RAFT weights loaded once at startup, not per frame pair

## Common mistakes to avoid
- Do not compute flow on non-consecutive frame pairs
- Do not average flow magnitude across the whole sequence before plotting
  — per-frame values matter, averaging hides popping events
- RAFT expects float32 tensors normalised to [0,1], not uint8
- OpenCV loads BGR not RGB — convert before passing to RAFT
- Popping threshold should be configurable, not hardcoded

## What Gaussian popping is
When a camera moves through a splat, Gaussians near the visibility
boundary appear and disappear suddenly between frames. This shows up as
a localised spike in optical flow magnitude even though the camera moved
smoothly. It is a known quality issue in 3DGS and is what this tool
measures.