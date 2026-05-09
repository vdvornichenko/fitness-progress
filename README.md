# Fitness Progress

A local Python app that turns a folder of mirror selfies into a smooth, body-aligned progress video.
It detects shoulder landmarks with MediaPipe, aligns every photo to the same position and scale,
and lets you review and manually correct each frame in a Streamlit UI before rendering the final MP4.

---

## Features

- **Automatic alignment** — MediaPipe Pose detects shoulders; each photo is rotated, scaled, and
  translated so shoulders land in the same canvas spot in every frame.
- **Manual correction editor** — Streamlit UI to tweak rotation, scale, and position per frame,
  approve or flag frames, and regenerate aligned PNGs on the fly.
- **Video rendering** — exports aligned video, optional original video, and optional side-by-side
  comparison video with crossfade or hard-cut transitions and date labels.
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
git clone https://github.com/<your-username>/fitness-progress.git
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

1. In the sidebar, enter the path to your **input photos folder** or click 📁 to browse.
2. The **output folder** is auto-derived as `<input_name>_output` next to the input. Override if needed.
3. Click **Open / Load Project**.
   - If a project already exists, it loads immediately for editing.
   - Otherwise, a **Build** section appears with options to run the alignment pipeline.
4. Click **🔨 Run Build** and watch the live progress bar and stats.
5. Once built, review every frame in the editor — adjust sliders, approve or flag frames.
6. Click **🎬 Render** in the sidebar to produce the final MP4(s).

### Option B — Command line

```bash
source .venv/bin/activate

# Build (detect, align, save project.json)
fitness-progress build --input ./photos --output ./output

# Render videos from an existing project
fitness-progress render --output ./output

# Useful flags
fitness-progress build --input ./photos --output ./output \
  --skip-render          # build only, no video
  --debug                # save debug overlay frames
  --aligned-only         # skip original video
  --comparison           # also render side-by-side video
  --frame-duration 1.2   # seconds per photo (default 0.8)
  --transition hard_cut  # crossfade (default) or hard_cut
  --include-unreviewed   # include frames not yet approved
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
│       ├── media_import.py    # Image collection and EXIF date extraction
│       ├── pose.py            # MediaPipe shoulder detection
│       ├── project_store.py   # project.json builder
│       ├── transforms.py      # Affine matrix helpers
│       └── video_render.py    # MP4 rendering (aligned / original / comparison)
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
| `frame_duration_seconds` | 0.8 | Seconds each photo is shown in the video |
| `fps` | 30 | Output video frame rate |

---

## Output files

After a build the output folder contains:

```
output/
├── aligned_frames/   # 0001.png, 0002.png, … — aligned PNG frames
├── debug/            # *_debug.jpg — overlay frames (if --debug was used)
├── project.json      # All items, transforms, statuses
├── progress_aligned.mp4
├── progress_original.mp4   (if rendered without --aligned-only)
└── progress_comparison.mp4 (if rendered with --comparison)
```

---

## License

MIT
