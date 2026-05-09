"""
Phase 3 — Streamlit manual correction editor.

Launch from the project root:
    cd fitness-progress-aligner
    source .venv/bin/activate
    streamlit run app/streamlit_editor.py

Workflow
--------
1. Enter the path to your **input photos folder** in the sidebar.
2. The output folder is auto-derived (<input_name>_output, next to input).
   Override it if you prefer a different location.
3. Click **Open / Load Project**.
   - If a project already exists it loads immediately for editing.
   - Otherwise a **Build** panel appears to run the alignment pipeline.
4. After building (or loading), use the editor to review and correct frames,
   then render videos from the sidebar.

Preview note
------------
Real-time preview applies the manual delta to the already-aligned PNG
(fast, ~instant).  "Save & Regenerate" applies the full combined transform
to the original source image (slower, highest quality).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ── Make progress_aligner importable when launched from project root ───────
_src = Path(__file__).parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from progress_aligner.media_import import collect_images
from progress_aligner.pose import load_rgb
from progress_aligner.transforms import apply_affine, build_affine_matrix, center_crop_to_canvas
from progress_aligner.video_render import render_aligned, render_comparison, render_original

# ── Constants ──────────────────────────────────────────────────────────────
PREVIEW_W     = 324      # preview image width in pixels
OVERLAY_ALPHA = 0.35


# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fitness Progress",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .stSlider > div { padding-top: 0; padding-bottom: 4px; }
  div[data-testid="stHorizontalBlock"] { align-items: center; }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ───────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_aligned_bgr(path: str) -> Optional[np.ndarray]:
    return cv2.imread(path)


def _apply_manual_delta(
    aligned_bgr: np.ndarray,
    manual: dict,
    pivot_x: int,
    pivot_y: int,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    rot   = manual["rotation_degrees"]
    scale = manual["scale"]
    tx    = manual["translate_x"]
    ty    = manual["translate_y"]
    if rot == 0.0 and scale == 1.0 and tx == 0.0 and ty == 0.0:
        return aligned_bgr
    rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    M   = build_affine_matrix((pivot_x, pivot_y), rot, scale, tx, ty)
    out = apply_affine(rgb, M, out_w, out_h)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def _thumbnail(bgr: np.ndarray, width: int = PREVIEW_W) -> Image.Image:
    h, w = bgr.shape[:2]
    new_h = int(h * width / w)
    small = cv2.resize(bgr, (width, new_h), interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))


def _blend(base: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
    if base.shape != overlay.shape:
        overlay = cv2.resize(overlay, (base.shape[1], base.shape[0]))
    return cv2.addWeighted(base, 1.0 - alpha, overlay, alpha, 0)


def _save_project(project: dict, path: str) -> None:
    with open(path, "w") as fh:
        json.dump(project, fh, indent=2)


def _regenerate_aligned_png(item: dict, settings: dict) -> tuple[bool, str]:
    source_path = Path(item["source_path"])
    out_w = settings["output_width"]
    out_h = settings["output_height"]
    try:
        rgb = load_rgb(source_path)
    except Exception as exc:
        return False, f"Could not load source: {exc}"
    auto_t   = item["auto_transform"]
    manual_t = item["manual_adjustment"]
    if item["pose"]["detected"]:
        midpoint = tuple(item["pose"]["shoulder_midpoint"])
        rotation = auto_t["rotation_degrees"] + manual_t["rotation_degrees"]
        scale    = auto_t["scale"] * manual_t["scale"]
        tx       = auto_t["translate_x"] + manual_t["translate_x"]
        ty       = auto_t["translate_y"] + manual_t["translate_y"]
        M           = build_affine_matrix(midpoint, rotation, scale, tx, ty)
        aligned_rgb = apply_affine(rgb, M, out_w, out_h)
    else:
        aligned_rgb = center_crop_to_canvas(rgb, out_w, out_h)
        mid = (out_w // 2, out_h // 2)
        M   = build_affine_matrix(
            mid,
            manual_t["rotation_degrees"],
            manual_t["scale"],
            manual_t["translate_x"],
            manual_t["translate_y"],
        )
        aligned_rgb = apply_affine(aligned_rgb, M, out_w, out_h)
    out_path = Path(item["outputs"]["aligned_frame"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR))
    _load_aligned_bgr.clear()
    return True, ""


def _slider_keys(item_id: str) -> tuple[str, str, str, str]:
    return (
        f"rot_{item_id}", f"scale_{item_id}",
        f"tx_{item_id}",  f"ty_{item_id}",
    )


def _init_sliders_for(item: dict) -> None:
    iid = item["id"]
    m   = item["manual_adjustment"]
    k_rot, k_scale, k_tx, k_ty = _slider_keys(iid)
    if k_rot   not in st.session_state: st.session_state[k_rot]   = float(m["rotation_degrees"])
    if k_scale not in st.session_state: st.session_state[k_scale] = float(m["scale"])
    if k_tx    not in st.session_state: st.session_state[k_tx]    = float(m["translate_x"])
    if k_ty    not in st.session_state: st.session_state[k_ty]    = float(m["translate_y"])


def _reset_sliders_for(item: dict) -> None:
    iid = item["id"]
    k_rot, k_scale, k_tx, k_ty = _slider_keys(iid)
    st.session_state[k_rot]   = 0.0
    st.session_state[k_scale] = 1.0
    st.session_state[k_tx]    = 0.0
    st.session_state[k_ty]    = 0.0


def _auto_output(input_folder: str) -> str:
    """Derive a default output folder path next to the input folder."""
    p = Path(input_folder).expanduser()
    return str(p.parent / (p.name + "_output"))


def _load_project_from(proj_json: Path) -> None:
    with open(proj_json) as fh:
        st.session_state.project = json.load(fh)
    st.session_state.project_path = str(proj_json)
    st.session_state.idx = 0


def _pick_folder() -> str:
    """Open a native macOS folder-picker dialog via AppleScript."""
    script = 'POSIX path of (choose folder with prompt "Select folder")'
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True, text=True,
    )
    path = result.stdout.strip().rstrip("/")
    return path if result.returncode == 0 else ""


# ── Handle folder-browse actions (must run before sidebar widgets) ─────────

# Initialize session-state keys used by the folder widgets
if "input_val" not in st.session_state:
    st.session_state.input_val = ""
if "output_val" not in st.session_state:
    st.session_state.output_val = ""

if st.session_state.get("_browse_input"):
    st.session_state._browse_input = False
    picked = _pick_folder()
    if picked:
        st.session_state.input_val = picked

if st.session_state.get("_browse_output"):
    st.session_state._browse_output = False
    picked = _pick_folder()
    if picked:
        st.session_state.output_val = picked


# ── Sidebar: Project Setup ─────────────────────────────────────────────────

with st.sidebar:
    st.title("Fitness Progress")
    st.markdown("**Project Setup**")

    # Input folder — text field + native browse button
    in_row_l, in_row_r = st.columns([5, 1])
    with in_row_l:
        st.text_input(
            "Input photos folder",
            placeholder="/path/to/photos",
            key="input_val",
        )
    with in_row_r:
        st.write("")
        if st.button("📁", key="browse_input_btn", help="Browse for folder"):
            st.session_state._browse_input = True
            st.rerun()
    input_val = st.session_state.input_val

    # Output folder — auto-derived, overridable, with its own browse button
    auto_out = _auto_output(input_val) if input_val else ""
    out_row_l, out_row_r = st.columns([5, 1])
    with out_row_l:
        st.text_input(
            "Output folder",
            placeholder=auto_out or "(auto-derived from input)",
            key="output_val",
            help="Leave blank to use the auto-derived path shown as placeholder.",
        )
    with out_row_r:
        st.write("")
        if st.button("📁", key="browse_output_btn", help="Browse for folder"):
            st.session_state._browse_output = True
            st.rerun()
    output_val = st.session_state.output_val

    # If user cleared the field, fall back to auto
    effective_output = output_val.strip() or auto_out

    open_clicked = st.button("📂 Open / Load Project", use_container_width=True)

# ── Handle Open button ─────────────────────────────────────────────────────

if open_clicked:
    # Values are already in session state via widget keys; nothing to copy

    in_p  = Path(input_val).expanduser()  if input_val  else None
    out_p = Path(effective_output).expanduser() if effective_output else None

    if out_p is None:
        st.sidebar.error("Cannot determine output folder — enter an input folder first.")
    else:
        proj_json = out_p / "project.json"
        if proj_json.exists():
            _load_project_from(proj_json)
            n = len(st.session_state.project["items"])
            st.sidebar.success(f"Loaded project — {n} items.")
            # Clear any leftover build panel state
            st.session_state.pop("build_ready", None)
        elif in_p is None or not in_p.is_dir():
            st.sidebar.error(
                "Input folder not found. Enter a valid path to scan for images."
            )
        else:
            images = collect_images(in_p)
            st.session_state.build_ready = {
                "input":    str(in_p),
                "output":   str(out_p),
                "n_images": len(images),
            }
            # Clear any stale project from a previous load
            st.session_state.pop("project", None)

# ── Sidebar: Build Pipeline ────────────────────────────────────────────────

build_info = st.session_state.get("build_ready")

if build_info and "project" not in st.session_state:
    with st.sidebar:
        st.divider()
        st.markdown("**Build Pipeline**")

        n_img = build_info["n_images"]
        if n_img:
            st.info(f"{n_img} images found — no project yet.")
        else:
            st.warning("No JPG/PNG images found in that folder.")

        build_debug    = st.toggle("Save debug frames",           value=False, key="build_debug")
        build_skip_rnd = st.toggle("Skip rendering after build",  value=True,  key="build_skip_rnd")
        build_aln_only = st.toggle("Aligned only (skip original)", value=True, key="build_aln_only")

        run_build = st.button(
            "🔨 Run Build",
            use_container_width=True,
            type="primary",
            disabled=(n_img == 0),
        )

    if run_build:
        import re as _re

        cmd = [
            sys.executable, "-m", "progress_aligner.cli", "build",
            "--input",  build_info["input"],
            "--output", build_info["output"],
        ]
        if build_debug:
            cmd.append("--debug")
        if build_skip_rnd:
            cmd.append("--skip-render")
        elif build_aln_only:
            cmd.append("--aligned-only")

        total = build_info["n_images"]

        # ── UI placeholders ────────────────────────────────────────────────
        phase_ph  = st.empty()                      # current phase label
        prog_ph   = st.empty()                      # progress bar
        stats_ph  = st.empty()                      # live stats row
        file_ph   = st.empty()                      # current file
        detail_ph = st.container()                  # collapsible raw log

        with detail_ph:
            with st.expander("Raw log", expanded=False):
                log_ph = st.empty()

        phase_ph.markdown("**Phase 1 — Detecting shoulders…**")
        prog_ph.progress(0.0)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONPATH": str(_src)},
        )

        # ── Parsing state ──────────────────────────────────────────────────
        pass_num   = 1
        done       = 0           # frames processed in current pass
        p1_ok      = 0
        p1_fail    = 0
        p2_ok      = 0
        p2_fallback = 0
        raw_lines: list[str] = []

        # Patterns emitted by cli.py
        _re_frame  = _re.compile(r"\[\s*(\d+)/\s*(\d+)\]")
        _re_p1_ok  = _re.compile(r"\bOK\b.*w=")
        _re_p1_fail= _re.compile(r"\bFAIL\b")
        _re_p2_ok  = _re.compile(r"\bOK\b.*rot=")
        _re_p2_fb  = _re.compile(r"\bFALLBACK\b")
        _re_pass2  = _re.compile(r"Pass 2")

        assert proc.stdout is not None
        for raw in proc.stdout:
            raw_lines.append(raw)
            log_ph.code("".join(raw_lines[-120:]), language=None)

            line = raw.rstrip()
            m = _re_frame.search(line)

            if _re_pass2.search(line):
                pass_num = 2
                done = 0
                phase_ph.markdown("**Phase 2 — Aligning and saving frames…**")
                prog_ph.progress(0.0)

            if m:
                done   = int(m.group(1))
                total_ = int(m.group(2))

                # classify the outcome on the same line
                if pass_num == 1:
                    if _re_p1_ok.search(line):
                        p1_ok += 1
                    elif _re_p1_fail.search(line):
                        p1_fail += 1
                else:
                    if _re_p2_ok.search(line):
                        p2_ok += 1
                    elif _re_p2_fb.search(line):
                        p2_fallback += 1

                prog_ph.progress(done / total_ if total_ else 0.0)

                fname = line.split("]", 1)[-1].strip().split("…")[0].strip() if "]" in line else ""
                file_ph.caption(f"Processing: `{fname}`")

            # Update stats block on every line
            if pass_num == 1:
                stats_ph.markdown(
                    f"**Phase 1** — "
                    f"✅ Detected: **{p1_ok}** &nbsp;&nbsp; "
                    f"⚠️ Failed: **{p1_fail}** &nbsp;&nbsp; "
                    f"📸 Total: **{total}**"
                )
            else:
                stats_ph.markdown(
                    f"**Phase 1** — ✅ {p1_ok} detected &nbsp; ⚠️ {p1_fail} failed &nbsp;&nbsp;|&nbsp;&nbsp; "
                    f"**Phase 2** — ✅ Aligned: **{p2_ok}** &nbsp; 🔄 Fallback: **{p2_fallback}**"
                )

        proc.wait()
        file_ph.empty()

        if proc.returncode == 0:
            prog_ph.progress(1.0)
            phase_ph.empty()
            stats_ph.markdown(
                f"**Done!** &nbsp;&nbsp; "
                f"Phase 1: ✅ {p1_ok} detected / ⚠️ {p1_fail} failed &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"Phase 2: ✅ {p2_ok} aligned / 🔄 {p2_fallback} fallback"
            )
            st.success("Build complete — loading project…")
            proj_json = Path(build_info["output"]) / "project.json"
            if proj_json.exists():
                _load_project_from(proj_json)
                st.session_state.pop("build_ready", None)
                st.rerun()
            else:
                st.error("Build succeeded but project.json was not found.")
        else:
            phase_ph.error(f"Build failed (exit {proc.returncode})")
            st.error("Check the raw log above for details.")

# ── Gate: stop here if no project is loaded ───────────────────────────────

if "project" not in st.session_state:
    st.markdown("## Welcome to Fitness Progress")
    st.markdown(
        "Use the sidebar to get started:\n\n"
        "1. Enter your **input photos folder** path.\n"
        "2. The output folder is auto-derived (or override it).\n"
        "3. Click **Open / Load Project**.\n"
        "   - If a project already exists it will load immediately.\n"
        "   - Otherwise a **Build** panel will appear to run the pipeline.\n\n"
        "After loading, you can review frames, correct alignment, and render videos."
    )
    st.stop()

proj     = st.session_state.project
items    = proj["items"]
settings = proj["settings"]
out_w    = settings["output_width"]
out_h    = settings["output_height"]
tgt_x, tgt_y = settings["target_shoulder_midpoint"]

# ── Sidebar: Stats, filter, render ────────────────────────────────────────

with st.sidebar:
    st.divider()
    n_approved = sum(1 for it in items if it["status"] == "approved")
    n_review   = len(items) - n_approved
    mc1, mc2 = st.columns(2)
    mc1.metric("Approved",     n_approved)
    mc2.metric("Needs Review", n_review)
    st.divider()
    filter_opt = st.radio(
        "Show", ["All", "Needs Review", "Approved"], index=0,
        key="filter_radio",
    )

    # ── Render Videos panel ────────────────────────────────────────────────
    st.divider()
    st.markdown("**Render Videos**")
    render_frame_dur = st.number_input(
        "Frame duration (s)",
        min_value=0.1, max_value=30.0, step=0.1,
        value=float(settings.get("frame_duration_seconds", 0.8)),
        help="How long each photo is held. Overrides the value in project.json.",
    )
    render_aligned_only       = st.toggle("Aligned only (skip original)", value=True)
    render_with_comparison    = st.toggle("Include comparison video",      value=False)
    render_include_unreviewed = st.toggle(
        "Include unreviewed frames", value=False,
        help="When off (default), frames still marked 'Needs Review' are excluded.",
    )
    render_transition  = st.selectbox("Transition", ["crossfade", "hard_cut"], index=0)
    render_date_labels = st.toggle("Date labels", value=True)

    do_render = st.button("🎬 Render", use_container_width=True, type="primary")

if do_render:
    proj_path  = Path(st.session_state.project_path)
    output_dir = proj_path.parent
    render_args = dict(
        project_json            = proj_path,
        transition              = render_transition,
        show_date_label         = render_date_labels,
        frame_duration_override = render_frame_dur,
        skip_unreviewed         = not render_include_unreviewed,
    )
    jobs: list[tuple[str, object]] = [("progress_aligned.mp4", render_aligned)]
    if not render_aligned_only:
        jobs.append(("progress_original.mp4", render_original))
    if render_with_comparison:
        jobs.append(("progress_comparison.mp4", render_comparison))

    render_status = st.empty()
    render_status.info(f"Rendering {len(jobs)} video(s)…")
    errors: list[str] = []
    for filename, fn in jobs:
        out_path = output_dir / filename
        try:
            n = fn(output_path=out_path, **render_args)
            size_mb = out_path.stat().st_size / 1_048_576 if out_path.exists() else 0
            st.sidebar.success(f"{filename}: {n} frames ({size_mb:.1f} MB)")
        except Exception as exc:
            errors.append(f"{filename}: {exc}")
    if errors:
        render_status.error("Render errors:\n" + "\n".join(errors))
    else:
        render_status.success("All videos rendered successfully.")

# ── Filter visible items ───────────────────────────────────────────────────

if filter_opt == "Needs Review":
    visible = [i for i, it in enumerate(items) if it["status"] == "needs_manual_review"]
elif filter_opt == "Approved":
    visible = [i for i, it in enumerate(items) if it["status"] == "approved"]
else:
    visible = list(range(len(items)))

if not visible:
    st.warning("No items match the current filter.")
    st.stop()

# ── Navigation state ───────────────────────────────────────────────────────

idx = int(st.session_state.get("idx", visible[0]))
if idx not in visible:
    idx = visible[0]
    st.session_state.idx = idx

pos = visible.index(idx)

# ── Layout ─────────────────────────────────────────────────────────────────

img_col, ctrl_col = st.columns([1, 1], gap="large")

# ── Controls column ────────────────────────────────────────────────────────

with ctrl_col:
    item = items[idx]
    iid  = item["id"]
    _init_sliders_for(item)

    k_rot, k_scale, k_tx, k_ty = _slider_keys(iid)

    st.subheader(f"[{iid}] {Path(item['source_path']).name}")

    status_color = "green" if item["status"] == "approved" else "orange"
    st.markdown(
        f"Status: :{status_color}[**{item['status'].replace('_', ' ').upper()}**]"
    )

    pose = item["pose"]
    if pose["detected"]:
        st.caption(
            f"Shoulders detected — width {pose['shoulder_width_px']:.0f}px  "
            f"tilt {pose['shoulder_angle_degrees']:.1f}°  "
            f"L vis {pose['left_visibility']:.2f}  R vis {pose['right_visibility']:.2f}"
        )
    else:
        st.caption(f"⚠ Detection failed: {pose.get('fail_reason', 'unknown')}")

    at = item["auto_transform"]
    st.caption(
        f"Auto — rot {at['rotation_degrees']:.2f}°  scale {at['scale']:.3f}  "
        f"tx {at['translate_x']:.0f}  ty {at['translate_y']:.0f}"
        + (" [rot↑]"   if at.get("clamped_rotation") else "")
        + (" [scale↑]" if at.get("clamped_scale")    else "")
    )

    st.divider()
    st.markdown("**Manual adjustment**")

    man_rot   = st.slider("Rotation (°)",  -20.0, 20.0,  step=0.1,  key=k_rot)
    man_scale = st.slider("Scale",          0.5,   2.0,   step=0.01, key=k_scale)
    man_tx    = st.slider("Move X (px)",  -400.0, 400.0,  step=1.0,  key=k_tx)
    man_ty    = st.slider("Move Y (px)",  -400.0, 400.0,  step=1.0,  key=k_ty)

    current_manual = {
        "rotation_degrees": man_rot,
        "scale":            man_scale,
        "translate_x":      man_tx,
        "translate_y":      man_ty,
    }

    st.divider()

    btn_col1, btn_col2 = st.columns(2)

    with btn_col1:
        if st.button("↺ Reset sliders", use_container_width=True):
            _reset_sliders_for(item)
            st.rerun()

    with btn_col2:
        save_clicked = st.button("💾 Save & Regenerate", use_container_width=True, type="primary")

    if save_clicked:
        item["manual_adjustment"] = current_manual
        with st.spinner("Regenerating aligned frame from source…"):
            ok, err = _regenerate_aligned_png(item, settings)
        if ok:
            _save_project(proj, st.session_state.project_path)
            st.success("Saved and regenerated.")
        else:
            st.error(f"Regeneration failed: {err}")

    st.divider()

    if item["status"] == "approved":
        if st.button("⚠ Mark as Needs Review", use_container_width=True):
            item["status"] = "needs_manual_review"
            _save_project(proj, st.session_state.project_path)
            st.rerun()
    else:
        if st.button("✓ Mark as Approved", use_container_width=True):
            item["status"] = "approved"
            _save_project(proj, st.session_state.project_path)
            st.rerun()

    st.divider()
    show_overlay = st.toggle("Show previous frame overlay", value=False)

# ── Image column ───────────────────────────────────────────────────────────

with img_col:
    nav1, nav2, nav3 = st.columns([1, 3, 1])
    with nav1:
        if st.button("◀ Prev", use_container_width=True, disabled=(pos == 0)):
            st.session_state.idx = visible[pos - 1]
            st.rerun()
    with nav2:
        st.markdown(
            f"<div style='text-align:center; padding-top:6px'>"
            f"<b>{pos + 1} / {len(visible)}</b></div>",
            unsafe_allow_html=True,
        )
    with nav3:
        if st.button("Next ▶", use_container_width=True, disabled=(pos == len(visible) - 1)):
            st.session_state.idx = visible[pos + 1]
            st.rerun()

    aligned_path = item["outputs"]["aligned_frame"]
    aligned_bgr  = _load_aligned_bgr(aligned_path)

    if aligned_bgr is None:
        st.warning(f"Aligned frame not found: {aligned_path}")
    else:
        preview_bgr = _apply_manual_delta(
            aligned_bgr, current_manual, tgt_x, tgt_y, out_w, out_h
        )

        if show_overlay and pos > 0:
            prev_item = items[visible[pos - 1]]
            prev_bgr  = _load_aligned_bgr(prev_item["outputs"]["aligned_frame"])
            if prev_bgr is not None:
                preview_bgr = _blend(preview_bgr, prev_bgr, OVERLAY_ALPHA)

        st.image(_thumbnail(preview_bgr), use_container_width=True)

    cap = item.get("capture_date")
    if cap:
        st.caption(f"Captured: {cap[:10]}")
