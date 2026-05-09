"""
CLI entry point.

Phase 1 command
---------------
    python -m progress_aligner.cli build --input ./photos --output ./output
    progress-aligner build --input ./photos --output ./output

Two-pass pipeline:
  Pass 1  Detect shoulders in every image; collect shoulder widths.
  Compute  Effective target_shoulder_width = median of valid widths (or config override).
  Pass 2  Align each image, save aligned frame + debug frame, build project.json.

Undetected images are NOT discarded — they receive a center-cropped fallback
frame and are marked ``needs_manual_review`` in project.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import mediapipe as mp

from .alignment import compute_transform
from .config import Config, load_config
from .debug_render import render_debug
from .media_import import collect_images, get_capture_date
from .pose import ShoulderDetection, detect_shoulders, load_rgb
from .project_store import build_item, save_project
from .transforms import apply_affine, build_affine_matrix, center_crop_to_canvas
from .video_render import render_aligned, render_comparison, render_original


def cmd_build(args: argparse.Namespace) -> None:
    input_dir  = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    if not input_dir.is_dir():
        print(f"ERROR: --input is not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    config_path = Path(args.config).resolve() if args.config else None
    config = load_config(config_path)

    aligned_dir = output_dir / "aligned_frames"
    debug_dir   = output_dir / "debug"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(input_dir)
    if not images:
        print("No JPG/PNG images found in input folder.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(images)} image(s)  →  output: {output_dir}")
    print()

    # ── Pass 1: shoulder detection ─────────────────────────────────────────
    print("Pass 1 — detecting shoulders …")
    detections:    list[ShoulderDetection] = []
    capture_dates: list[datetime | None]   = []

    with mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.3,
    ) as pose_model:
        for i, path in enumerate(images, 1):
            print(f"  [{i:>3}/{len(images)}] {path.name} …", end=" ", flush=True)
            try:
                rgb = load_rgb(path)
            except Exception as exc:
                det = ShoulderDetection(detected=False, fail_reason=f"load error: {exc}")
                detections.append(det)
                capture_dates.append(None)
                print(f"LOAD ERROR: {exc}")
                continue

            det = detect_shoulders(rgb, pose_model, config)
            detections.append(det)
            capture_dates.append(get_capture_date(path))

            if det.detected:
                print(f"OK  (w={det.shoulder_width_px:.0f}px  tilt={det.angle_degrees:.1f}°)")
            else:
                print(f"FAIL ({det.fail_reason})")

    # ── Compute effective target shoulder width ────────────────────────────
    valid_widths = [d.shoulder_width_px for d in detections if d.detected]

    if config.target_shoulder_width is not None:
        effective_target_width = float(config.target_shoulder_width)
        width_source = f"config override → {effective_target_width:.1f}px"
    elif valid_widths:
        effective_target_width = statistics.median(valid_widths)
        width_source = (
            f"auto-median → {effective_target_width:.1f}px"
            f" (from {len(valid_widths)} detected images)"
        )
    else:
        effective_target_width = 420.0
        width_source = "fallback default → 420px (no detections)"

    n_ok = sum(1 for d in detections if d.detected)
    print()
    print(f"  Detection : {n_ok}/{len(images)} ({100 * n_ok / len(images):.1f}%)")
    print(f"  Target w  : {width_source}")
    print()

    # ── Pass 2: align, render, save ────────────────────────────────────────
    print("Pass 2 — aligning and saving frames …")
    project_items: list[dict] = []
    created_at = datetime.now()

    for i, (path, detection, cap_date) in enumerate(
        zip(images, detections, capture_dates), 1
    ):
        print(f"  [{i:>3}/{len(images)}] {path.name} …", end=" ", flush=True)

        affine_M  = None
        transform = None

        try:
            rgb = load_rgb(path)
        except Exception as exc:
            print(f"LOAD ERROR: {exc}")
            detection = ShoulderDetection(detected=False, fail_reason=f"load error: {exc}")
            import numpy as _np
            rgb = _np.zeros((config.output_height, config.output_width, 3), dtype=_np.uint8)

        if detection.detected:
            transform = compute_transform(detection, config, effective_target_width)
            affine_M  = build_affine_matrix(
                midpoint         = detection.midpoint_px,
                rotation_degrees = transform.rotation_degrees,
                scale            = transform.scale,
                translate_x      = transform.translate_x,
                translate_y      = transform.translate_y,
            )
            aligned_rgb = apply_affine(rgb, affine_M, config.output_width, config.output_height)

            clamp = ""
            if transform.clamped_rotation:
                clamp += " [rot↑]"
            if transform.clamped_scale:
                clamp += " [scale↑]"
            print(
                f"OK  rot={transform.rotation_degrees:.2f}°"
                f"  scale={transform.scale:.3f}{clamp}"
            )
        else:
            aligned_rgb = center_crop_to_canvas(rgb, config.output_width, config.output_height)
            print(f"FALLBACK ({detection.fail_reason})")

        idx          = f"{i:04d}"
        aligned_path = aligned_dir / f"{idx}.png"
        debug_path   = debug_dir   / f"{idx}_debug.jpg"

        cv2.imwrite(str(aligned_path), cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR))

        if args.debug:
            debug_bgr = render_debug(
                aligned_rgb     = aligned_rgb,
                detection       = detection,
                transform       = transform,
                affine_matrix   = affine_M,
                config          = config,
                source_filename = path.name,
                index           = i,
            )
            cv2.imwrite(str(debug_path), debug_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])

        project_items.append(build_item(
            index        = i,
            source_path  = path,
            output_base  = output_dir,
            capture_date = cap_date,
            detection    = detection,
            transform    = transform,
        ))

    # ── Save project.json ──────────────────────────────────────────────────
    project_json = output_dir / "project.json"
    save_project(
        path                          = project_json,
        config                        = config,
        items                         = project_items,
        effective_target_shoulder_width = effective_target_width,
        created_at                    = created_at,
    )

    # ── Video rendering ────────────────────────────────────────────────────
    if not args.skip_render:
        _run_render(
            project_json    = project_json,
            output_dir      = output_dir,
            transition      = args.transition,
            date_labels     = not args.no_date_labels,
            skip_original   = args.aligned_only,
            with_comparison = args.comparison,
            frame_duration  = args.frame_duration,
            skip_unreviewed = not args.include_unreviewed,
        )

    # ── Summary ───────────────────────────────────────────────────────────
    approved     = sum(1 for it in project_items if it["status"] == "approved")
    needs_review = len(project_items) - approved

    print()
    print("─" * 60)
    print(f"  Processed           : {len(images)} images")
    print(f"  Approved (aligned)  : {approved}")
    print(f"  Needs manual review : {needs_review}")
    print(f"  Aligned frames  →  {aligned_dir}")
    print(f"  Debug frames    →  {debug_dir}")
    print(f"  Project file    →  {project_json}")
    print("─" * 60)


def _run_render(
    project_json: Path,
    output_dir: Path,
    transition: str,
    date_labels: bool,
    skip_original: bool = False,
    with_comparison: bool = False,
    frame_duration: Optional[float] = None,
    skip_unreviewed: bool = True,
) -> None:
    print()
    print("Pass 3 — rendering videos …")

    videos: list[tuple[str, Any]] = [
        ("progress_aligned.mp4",    render_aligned),
    ]
    if not skip_original:
        videos.append(("progress_original.mp4", render_original))
    if with_comparison:
        videos.append(("progress_comparison.mp4", render_comparison))
    for filename, fn in videos:
        out = output_dir / filename
        print(f"  {filename} …", end=" ", flush=True)
        try:
            n = fn(
                project_json            = project_json,
                output_path             = out,
                transition              = transition,
                show_date_label         = date_labels,
                frame_duration_override = frame_duration,
                skip_unreviewed         = skip_unreviewed,
            )
            size_mb = out.stat().st_size / 1_048_576 if out.exists() else 0
            print(f"{n} frames  →  {size_mb:.1f} MB")
        except Exception as exc:
            print(f"ERROR: {exc}")


def cmd_render(args: argparse.Namespace) -> None:
    output_dir   = Path(args.output).resolve()
    project_json = output_dir / "project.json"
    if not project_json.exists():
        print(f"ERROR: project.json not found in {output_dir}", file=sys.stderr)
        sys.exit(1)
    _run_render(
        project_json    = project_json,
        output_dir      = output_dir,
        transition      = args.transition,
        date_labels     = not args.no_date_labels,
        skip_original   = args.aligned_only,
        with_comparison = args.comparison,
        frame_duration  = args.frame_duration,
        skip_unreviewed = not args.include_unreviewed,
    )


def _regenerate_item(item: dict, settings: dict) -> tuple[bool, str]:
    """Rebuild the aligned PNG for one item using the combined auto + manual transform.

    Returns (success, error_message).
    """
    from .pose import load_rgb
    from .transforms import apply_affine, build_affine_matrix, center_crop_to_canvas

    source_path = Path(item["source_path"])
    out_w = settings["output_width"]
    out_h = settings["output_height"]

    try:
        rgb = load_rgb(source_path)
    except Exception as exc:
        return False, f"load error: {exc}"

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
    return True, ""


def cmd_regenerate(args: argparse.Namespace) -> None:
    output_dir   = Path(args.output)
    project_json = output_dir / "project.json"

    if not project_json.exists():
        print(f"Error: {project_json} not found.", file=sys.stderr)
        sys.exit(1)

    with open(project_json) as fh:
        proj = json.load(fh)

    items    = proj["items"]
    settings = proj["settings"]

    n_ok = n_err = 0
    for item in items:
        iid = item["id"]
        ok, err = _regenerate_item(item, settings)
        if ok:
            n_ok += 1
            print(f"  [{iid}] regenerated")
        else:
            n_err += 1
            print(f"  [{iid}] FAILED — {err}", file=sys.stderr)

    print(f"\nRegenerated {n_ok}/{len(items)} frames"
          + (f", {n_err} errors" if n_err else "") + ".")

    if args.render:
        _run_render(
            project_json    = project_json,
            output_dir      = output_dir,
            transition      = args.transition,
            date_labels     = not args.no_date_labels,
            skip_original   = args.aligned_only,
            with_comparison = args.comparison,
            frame_duration  = args.frame_duration,
            skip_unreviewed = not args.include_unreviewed,
        )


def _add_render_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--transition", choices=["hard_cut", "crossfade"], default="crossfade",
        help="Transition style between frames (default: crossfade).",
    )
    p.add_argument(
        "--no-date-labels", action="store_true",
        help="Omit capture date labels from video frames.",
    )
    p.add_argument(
        "--aligned-only", action="store_true",
        help="Render only the aligned video; skip the original video.",
    )
    p.add_argument(
        "--comparison", action="store_true",
        help="Also render the side-by-side comparison video (off by default).",
    )
    p.add_argument(
        "--frame-duration", type=float, default=None, metavar="SECONDS",
        help="Seconds each photo is displayed (overrides project.json setting).",
    )
    p.add_argument(
        "--include-unreviewed", action="store_true",
        help="Include frames marked 'needs_manual_review' in rendered videos "
             "(by default only approved frames are included).",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="fitness-progress",
        description="Fitness progress photo aligner.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── build ──────────────────────────────────────────────────────────────
    build_p = sub.add_parser(
        "build",
        help="Detect shoulders, align frames, and render videos (full pipeline).",
    )
    build_p.add_argument("--input",  required=True, help="Folder containing input photos.")
    build_p.add_argument("--output", required=True, help="Output folder.")
    build_p.add_argument("--config", default=None,  help="Path to config.yaml (optional).")
    build_p.add_argument(
        "--skip-render", action="store_true",
        help="Skip video rendering (frames and project.json only).",
    )
    build_p.add_argument(
        "--debug", action="store_true",
        help="Save debug overlay frames to the debug/ subfolder.",
    )
    _add_render_args(build_p)

    # ── render ─────────────────────────────────────────────────────────────
    render_p = sub.add_parser(
        "render",
        help="Re-render videos from an existing output folder.",
    )
    render_p.add_argument(
        "--output", required=True,
        help="Output folder containing project.json and aligned_frames/.",
    )
    _add_render_args(render_p)

    # ── regenerate ─────────────────────────────────────────────────────────
    regen_p = sub.add_parser(
        "regenerate",
        help="Re-apply combined (auto + manual) transforms to source images and "
             "rebuild aligned PNGs.  Run after editing manual_adjustment in "
             "project.json or after using the Streamlit editor.",
    )
    regen_p.add_argument(
        "--output", required=True,
        help="Output folder containing project.json.",
    )
    regen_p.add_argument(
        "--render", action="store_true",
        help="Also re-render all three videos after regenerating frames.",
    )
    _add_render_args(regen_p)

    args = parser.parse_args()
    if args.command == "build":
        cmd_build(args)
    elif args.command == "render":
        cmd_render(args)
    elif args.command == "regenerate":
        cmd_regenerate(args)


if __name__ == "__main__":
    main()
