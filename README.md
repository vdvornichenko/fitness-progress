# Fitness Progress

A local Python app that turns a mixed folder of **photos and short videos** into a smooth,
body-aligned progress video. It detects shoulder landmarks with MediaPipe, aligns every frame to
the same position and scale, and lets you review and manually correct each item in a Streamlit UI
before rendering the final MP4.

Photos and videos are treated as equally valid sources. From each video the pipeline automatically
picks the single sharpest, best-posed frame — applying rotation correction for portrait clips and
HDR tone-mapping for HDR10+/HLG footage — so mixed libraries work seamlessly.

---

## Features

- **Photos and videos in one folder** — drop `.jpg`, `.png`, `.mp4`, `.mov`, and `.m4v` files
  together; the pipeline processes them all chronologically.
- **Automatic best-frame extraction** — for each video, frames are sampled at a configurable
  interval and scored on five quality components (shoulder detection, shoulder visibility, blur,
  centering, brightness); the highest-scoring frame is selected automatically.
- **Portrait-video auto-rotation** — container rotation metadata is read and applied before any
  analysis, so clips shot in portrait mode are handled correctly.
- **HDR10+/HLG support** — HDR footage is detected via `ffprobe` (with a luminance heuristic
  as fallback) and tone-mapped to SDR before processing, preventing washed-out frames.
- **Automatic alignment** — MediaPipe Pose detects shoulders in each selected frame; the frame is
  rotated, scaled, and translated so shoulders land in the same canvas spot across every item.
- **Manual correction editor** — Streamlit UI to tweak rotation, scale, and position per frame,
  approve or flag items, and regenerate aligned PNGs on the fly. Video items show their
  auto-selected timestamp, quality score, and reason; a slider lets you re-pick any timestamp.
- **Live build progress** — a real-time progress bar tracks all four pipeline phases (photo
  detection, video extraction, photo alignment, video alignment) with per-item status.
- **Video rendering** — exports the aligned slideshow video, an optional original-crop video, and
  an optional side-by-side comparison video with crossfade or hard-cut transitions and date labels.
- **Fallback for missed detections** — undetected frames get a center-cropped fallback and are
  flagged for manual review; they are excluded from the final video until approved.

---

## Requirements

- Python 3.11
- macOS (folder picker uses AppleScript; everything else is cross-platform)

---

## Setup

```bash
# 1. Clone
git clone https://github.com/vdvornichenko/fitness-progress.git
cd fitness-progress

# 2. Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install the package (editable)
pip install -e .
```

---

## Usage

### Option A — Streamlit UI (recommended)

```bash
source .venv/bin/activate
streamlit run app/streamlit_editor.py
```

1. In the sidebar, enter the path to your **input folder** (photos, videos, or both) or click 📁 to browse.
2. The **output folder** is auto-derived as `<input_name>_output` next to the input. Override if needed.
3. Click **Open / Load Project**.
   - If a project already exists, it loads immediately for editing.
   - Otherwise, a **Build** section appears with pipeline options.
4. Toggle **Include videos** (on by default) if your folder contains `.mp4`/`.mov`/`.m4v` files.
5. Click **🔨 Run Build** and watch the live progress bar track all phases:
   - **Phase 1** — shoulder detection in photos
   - **Phase 1b** — best-frame extraction from videos
   - **Phase 2** — photo alignment
   - **Phase 2b** — video-frame alignment
6. Once built, review every item in the editor — adjust sliders, approve or flag frames; for
   video items use the timestamp slider to re-pick any moment and re-extract the frame.
7. Click **🎬 Render** in the sidebar to produce the final MP4(s).

### Option B — Command line

```bash
source .venv/bin/activate

# Build from a mixed photo+video folder
fitness-progress build --input ./media --output ./output

# Photos only
fitness-progress build --input ./media --output ./output --no-videos

# Render videos from an existing project
fitness-progress render --output ./output

# Useful flags
fitness-progress build --input ./media --output ./output \
  --skip-render          # build only, no video
  --debug                # save debug overlay frames
  --aligned-only         # skip original video
  --comparison           # also render side-by-side video
  --frame-duration 1.2   # seconds per frame (default 0.8)
  --transition hard_cut  # crossfade (default) or hard_cut
  --include-unreviewed   # include frames not yet approved
  --video-sample-interval 0.5   # seconds between sampled video frames
  --min-video-score 0.6         # minimum quality score to auto-approve
```

---

## Project structure

```
fitness-progress-aligner/
├── app/
│   └── streamlit_editor.py   # Streamlit review & correction UI
├── src/
│   └── progress_aligner/
│       ├── alignment.py       # Compute alignment transform from detection
│       ├── cli.py             # CLI entry point (build / render / regenerate)
│       ├── config.py          # Config dataclass and YAML loader
│       ├── debug_render.py    # Debug overlay renderer
│       ├── media_import.py    # Image/video collection and EXIF date extraction
│       ├── pose.py            # MediaPipe shoulder detection
│       ├── project_store.py   # project.json builder
│       ├── transforms.py      # Affine matrix helpers
│       ├── video_moment_picker.py  # Select best frame from scored candidates
│       ├── video_render.py    # MP4 rendering (aligned / original / comparison)
│       ├── video_sampling.py  # Sample frames from a video at fixed intervals
│       └── video_scoring.py   # Score candidate frames (pose, blur, brightness…)
├── config.example.yaml        # Annotated configuration reference
├── pyproject.toml
└── requirements.txt
```

---

## Configuration

Copy `config.example.yaml` into your output folder as `config.yaml` to override defaults:

| Key | Default | Description |
|-----|---------|-------------|
| `output_width` / `output_height` | 1080 × 1920 | Canvas size in pixels |
| `target_shoulder_midpoint_x/y` | 540 / 620 | Where shoulders land in the canvas |
| `target_shoulder_width` | `null` (auto-median) | Force a fixed shoulder width |
| `max_rotation_degrees` | 8.0 | Max rotation correction applied |
| `min_scale` / `max_scale` | 0.75 / 1.35 | Scale clamp range |
| `frame_duration_seconds` | 0.8 | Seconds each frame is shown in the video |
| `fps` | 30 | Output video frame rate |
| `video.enabled` | `true` | Process videos alongside photos |
| `video.extensions` | `[.mp4, .mov, .m4v]` | Extensions to treat as video |
| `video.best_frame.sample_interval_seconds` | 0.25 | Seconds between sampled frames |
| `video.best_frame.avoid_start/end_seconds` | 0.5 | Trim start/end of each clip |
| `video.best_frame.min_score` | 0.5 | Auto-approve threshold |
| `video.best_frame.scoring.*` | see YAML | Per-component score weights |

---

## Output files

After a build the output folder contains:

```
output/
├── aligned_frames/       # 0001.png, 0002.png, … — aligned PNG frames (photos + video stills)
├── debug/                # *_debug.jpg — overlay frames (if --debug was used)
├── video_candidates/     # <name>_best.jpg — auto-selected stills from each video
├── project.json          # All items, transforms, statuses
├── progress_aligned.mp4
├── progress_original.mp4   (if rendered without --aligned-only)
└── progress_comparison.mp4 (if rendered with --comparison)
```

---

## How video processing works

For each video file found in the input folder the pipeline runs four steps automatically:

1. **Info** — reads duration, FPS, resolution, container rotation metadata, and checks for HDR
   colour transfer (via `ffprobe`; falls back to a luminance heuristic if ffprobe is absent).
2. **Sampling** — decodes one frame every `video.best_frame.sample_interval_seconds`, skipping
   the first and last `avoid_*_seconds` of the clip.  Portrait clips are rotated to upright before
   analysis; HDR frames are Reinhard tone-mapped to SDR.
3. **Scoring** — each candidate frame is scored on five components:
   | Component | Weight | What it measures |
   |-----------|--------|-----------------|
   | Pose detected | 0.35 | Shoulders visible and confident |
   | Shoulder visibility | 0.25 | Both landmarks clearly in frame |
   | Blur | 0.15 | Laplacian variance (sharpness) |
   | Centering | 0.15 | Shoulder midpoint near canvas centre |
   | Brightness | 0.10 | Gaussian centred on mid-grey |
4. **Selection** — the highest-scoring frame is chosen; if its score exceeds `min_score` it is
   auto-approved, otherwise it is flagged for manual review.

In the Streamlit editor, each video item shows the auto-selected timestamp, composite score, and
reason string. A timestamp slider lets you manually re-pick any moment and re-extract the frame
with the same rotation correction and HDR tone-mapping applied.

---

## License

MIT
