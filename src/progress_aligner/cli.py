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
from .media_import import collect_images, collect_videos, get_capture_date, get_video_capture_date
from .pose import ShoulderDetection, detect_shoulders, load_rgb
from .project_store import build_item, save_project
from .transforms import apply_affine, build_affine_matrix, center_crop_to_canvas
from .video_render import render_aligned, render_comparison, render_original
from .video_moment_picker import pick_best_frame, PickedMoment
from .video_sampling import get_video_info, sample_frames
from .video_scoring import score_frames


def cmd_build(args: argparse.Namespace) -> None:
    input_dir  = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    if not input_dir.is_dir():
        print(f"ERROR: --input is not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    config_path = Path(args.config).resolve() if args.config else None
    config = load_config(config_path)

    # CLI overrides for video settings
    if getattr(args, "video_sample_interval", None) is not None:
        config.video_sample_interval = args.video_sample_interval
    if getattr(args, "min_video_score", None) is not None:
        config.video_min_score = args.min_video_score

    aligned_dir = output_dir / "aligned_frames"
    debug_dir   = output_dir / "debug"
    video_candidates_dir = output_dir / "video_candidates"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    include_videos = getattr(args, "include_videos", False)
    if include_videos:
        video_candidates_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(input_dir)
    videos: list[Path] = []
    if include_videos and config.video_enabled:
        from .media_import import VIDEO_EXTENSIONS
        exts = frozenset(config.video_extensions) if config.video_extensions else VIDEO_EXTENSIONS
        videos = collect_videos(input_dir, extensions=exts)

    if not images and not videos:
        print("No JPG/PNG images or supported videos found in input folder.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(images)} image(s) and {len(videos)} video(s)  \u2192  output: {output_dir}")
    print()

    # ── Pass 1: shoulder detection on photos ─────────────────────────────────
    print("Pass 1 — detecting shoulders in photos…")
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
    # ── Pass 1b: video moment selection ───────────────────────────────────────
    picked_moments: list[PickedMoment] = []

    if videos:
        print()
        print(f"Pass 1b — extracting best frames from {len(videos)} video(s)…")
        with mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.3,
        ) as vm:
            for vi, vpath in enumerate(videos, 1):
                print(f"  [{vi:>3}/{len(videos)}] {vpath.name}", flush=True)
                try:
                    info = get_video_info(vpath)
                    rotation_label = (
                        f"  rotation={info.rotation_degrees}°" if info.rotation_degrees else ""
                    )
                    hdr_label = f"  HDR({info.hdr_transfer})" if info.is_hdr else ""
                    print(
                        f"         duration={info.duration_seconds:.1f}s  "
                        f"fps={info.fps:.0f}  {info.width}x{info.height}"
                        f"{rotation_label}{hdr_label}",
                        flush=True,
                    )
                    n_expected = max(1, int(
                        (max(0.0, info.duration_seconds
                             - config.video_avoid_start_seconds
                             - config.video_avoid_end_seconds)
                         ) / config.video_sample_interval
                    ))
                    print(f"         Sampling ~{n_expected} frame(s)…", flush=True)
                    candidates = sample_frames(
                        vpath,
                        interval_seconds=config.video_sample_interval,
                        avoid_start=config.video_avoid_start_seconds,
                        avoid_end=config.video_avoid_end_seconds,
                    )
                    print(f"         {len(candidates)} candidate frame(s) sampled", flush=True)
                    scored = score_frames(candidates, vm, config)
                    moment = pick_best_frame(scored, vpath, config)
                    if moment:
                        picked_moments.append(moment)
                        status = (
                            f"best ts={moment.timestamp_seconds:.2f}s  "
                            f"score={moment.score:.2f}"
                        )
                        print(f"         → {status}", flush=True)

                        # Save best candidate frame to video_candidates/ for inspection
                        cand_path = video_candidates_dir / f"{vpath.stem}_best.jpg"
                        import cv2 as _cv2
                        _cv2.imwrite(
                            str(cand_path),
                            _cv2.cvtColor(moment.scored_frame.candidate.rgb, _cv2.COLOR_RGB2BGR),
                            [_cv2.IMWRITE_JPEG_QUALITY, 90],
                        )
                    else:
                        print("         → no valid candidate found — skipping", flush=True)
                except Exception as exc:
                    print(f"         ERROR: {exc}", file=sys.stderr, flush=True)

    # ── Merge photo detections + video moments for width calibration ───────────
    # ── Compute effective target shoulder width ────────────────────────────
    # Photos are the stable reference baseline — they define the target body
    # size that everything (including video frames) will be normalised to.
    # Only fall back to video-derived widths when there are no photo
    # detections (e.g. a video-only project).
    photo_widths = [d.shoulder_width_px for d in detections if d.detected]
    video_widths = [
        pm.scored_frame.detection.shoulder_width_px
        for pm in picked_moments
        if pm.scored_frame.detection.detected
    ]

    if config.target_shoulder_width is not None:
        effective_target_width = float(config.target_shoulder_width)
        width_source = f"config override → {effective_target_width:.1f}px"
    elif photo_widths:
        effective_target_width = statistics.median(photo_widths)
        width_source = (
            f"auto-median → {effective_target_width:.1f}px"
            f" (from {len(photo_widths)} photo(s) — videos normalised to this)"
        )
    elif video_widths:
        effective_target_width = statistics.median(video_widths)
        width_source = (
            f"auto-median → {effective_target_width:.1f}px"
            f" (from {len(video_widths)} video frame(s) — no photos available)"
        )
    else:
        effective_target_width = 420.0
        width_source = "fallback default → 420px (no detections)"

    n_ok = sum(1 for d in detections if d.detected)
    print()
    print(f"  Photos    : {n_ok}/{len(images)} detected ({100 * n_ok / len(images):.1f}%)" if images else "  Photos    : 0")
    if videos:
        v_ok = sum(1 for pm in picked_moments if pm.scored_frame.detection.detected)
        print(f"  Videos    : {v_ok}/{len(videos)} with detected shoulders")
    print(f"  Target w  : {width_source}")
    print()

    # ── Pass 2: align photos ───────────────────────────────────────────────────
    print("Pass 2 — aligning and saving frames…")
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
            media_type   = "photo",
        ))

    # ── Pass 2b: align video best frames ──────────────────────────────────────
    if picked_moments:
        print()
        print(f"Pass 2b — aligning {len(picked_moments)} video best frame(s)…", flush=True)
        base_index = len(images) + 1
        for vi, moment in enumerate(picked_moments, base_index):
            vpath = moment.source_path
            print(f"  [{vi:>3}/{base_index + len(picked_moments) - 1}] {vpath.name} @ {moment.timestamp_seconds:.2f}s …", end=" ", flush=True)

            detection = moment.scored_frame.detection
            rgb       = moment.scored_frame.candidate.rgb
            transform = None
            affine_M  = None

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
                print(f"OK  rot={transform.rotation_degrees:.2f}°  scale={transform.scale:.3f}", flush=True)
            else:
                aligned_rgb = center_crop_to_canvas(rgb, config.output_width, config.output_height)
                print(f"FALLBACK ({detection.fail_reason})", flush=True)

            idx          = f"{vi:04d}"
            aligned_path = aligned_dir / f"{idx}.png"
            cv2.imwrite(str(aligned_path), cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR))

            if args.debug and affine_M is not None:
                debug_path = debug_dir / f"{idx}_debug.jpg"
                debug_bgr = render_debug(
                    aligned_rgb=aligned_rgb, detection=detection, transform=transform,
                    affine_matrix=affine_M, config=config,
                    source_filename=vpath.name, index=vi,
                )
                cv2.imwrite(str(debug_path), debug_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])

            cap_date = get_video_capture_date(vpath)
            project_items.append(build_item(
                index                  = vi,
                source_path            = vpath,
                output_base            = output_dir,
                capture_date           = cap_date,
                detection              = detection,
                transform              = transform,
                media_type             = "video_frame",
                video_timestamp_seconds= moment.timestamp_seconds,
                video_score            = moment.score,
                video_score_reason     = moment.reason,
            ))

    # ── Sort all items chronologically and re-number ───────────────────────
    # Photos and videos were processed separately; merge into one chronological
    # list before writing project.json so the slideshow plays in date order.
    def _item_sort_key(it: dict):
        cap = it.get("capture_date")
        if cap:
            try:
                return datetime.fromisoformat(cap)
            except ValueError:
                pass
        return datetime.max   # items with no date go to the end

    project_items.sort(key=_item_sort_key)

    # Re-number IDs and rename output files to match new order.
    # Two-phase rename avoids collisions when photo and video index ranges
    # overlap after sorting (e.g. renaming 0704.png→0339.png would silently
    # overwrite the existing photo 0339.png in a single-pass loop).
    #
    # Phase 1: move every out-of-place file to a safe temporary name.
    _to_finalize: list[tuple[dict, str]] = []
    for new_idx, item in enumerate(project_items, 1):
        new_id = f"{new_idx:04d}"
        old_id = item["id"]
        # Always update the path strings so they reflect the final names.
        item["id"] = new_id
        item["outputs"]["aligned_frame"] = str(aligned_dir / f"{new_id}.png")
        item["outputs"]["debug_frame"]   = str(debug_dir   / f"{new_id}_debug.jpg")
        if old_id == new_id:
            continue
        tmp_aligned = aligned_dir / f"_reorder_{old_id}.png"
        tmp_debug   = debug_dir   / f"_reorder_{old_id}.jpg"
        old_aligned = aligned_dir / f"{old_id}.png"
        old_debug   = debug_dir   / f"{old_id}_debug.jpg"
        if old_aligned.exists():
            old_aligned.rename(tmp_aligned)
        if old_debug.exists():
            old_debug.rename(tmp_debug)
        _to_finalize.append((item, tmp_aligned, tmp_debug))

    # Phase 2: rename temporaries to their final names (no collisions possible).
    for item, tmp_aligned, tmp_debug in _to_finalize:
        final_aligned = Path(item["outputs"]["aligned_frame"])
        final_debug   = Path(item["outputs"]["debug_frame"])
        if tmp_aligned.exists():
            tmp_aligned.rename(final_aligned)
        if tmp_debug.exists():
            tmp_debug.rename(final_debug)

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
    build_p.add_argument(
        "--include-videos", action="store_true",
        help="Also import .mp4/.mov/.m4v files and extract the best still frame.",
    )
    build_p.add_argument(
        "--video-sample-interval", type=float, default=None, metavar="SEC",
        help="Seconds between sampled video frames (overrides config).",
    )
    build_p.add_argument(
        "--min-video-score", type=float, default=None, metavar="SCORE",
        help="Minimum composite score for a video frame to be accepted (overrides config).",
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
