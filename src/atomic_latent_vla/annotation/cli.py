from __future__ import annotations

import argparse
import json
from pathlib import Path

from .client import create_annotation_client
from .lerobot import load_lerobot_episode
from .mirror import write_mirrored_episode_bundle
from .pipeline import AtomicSegmentationPipeline, PipelineConfig


DEFAULT_CR1_URDF_DIR = Path("/home/admin123/cr1_recordclient/model/CR1_UPPER/urdf")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Segment synchronized robot videos into 13 spatial atomic skills."
    )
    parser.add_argument("--video", action="append", help="Manual video path; repeat for views")
    parser.add_argument("--view-name", action="append", help="View label; repeat in video order")
    parser.add_argument(
        "--lerobot-root",
        type=Path,
        help=(
            "LeRobot dataset root; resolves state/action, task, timestamps, and either "
            "base/right or mirrored base/left videos"
        ),
    )
    parser.add_argument(
        "--episode-index", type=int, help="LeRobot episode index used with --lerobot-root"
    )
    parser.add_argument(
        "--lerobot-base-video-key",
        default=None,
        help="Override the auto-detected base-camera feature key",
    )
    parser.add_argument(
        "--lerobot-left-video-key",
        default=None,
        help="Override the left wrist camera used by --mirror-left-to-right",
    )
    parser.add_argument(
        "--lerobot-right-video-key",
        default=None,
        help="Override the auto-detected right wrist-camera feature key",
    )
    parser.add_argument("--task", help="Long task/prompt; optional override in LeRobot mode")
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, help="Optional PipelineConfig JSON file")
    trace_group = parser.add_mutually_exclusive_group()
    trace_group.add_argument(
        "--tcp-trace", type=Path, help="Optional CSV/JSON[L] base-frame TCP trace"
    )
    trace_group.add_argument(
        "--joint-trace",
        type=Path,
        help="CR1 HDF5 containing 30Hz qpos (or compatible 120Hz data); TCP uses FK",
    )
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument(
        "--urdf",
        type=Path,
        help="7-DoF arm URDF; default is CR1ARML/R under cr1_recordclient",
    )
    parser.add_argument(
        "--tcp-frame",
        default=None,
        help=(
            "explicit URDF TCP frame; by default use the arm wrist-x link plus "
            "the shared 0.10 m local-z tool offset"
        ),
    )
    parser.add_argument(
        "--mount-xyz",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Shoulder origin in CR1 body coordinates; defaults depend on --arm",
    )
    parser.add_argument("--extra-context", default="")
    parser.add_argument(
        "--provider",
        choices=("qwen", "codex"),
        default="qwen",
        help="Vision annotation backend (default: qwen)",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Qwen semantic sampling temperature; default 0.25",
    )
    parser.add_argument("--sample-fps", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--min-segment-duration-s", type=float, default=None)
    parser.add_argument("--quantity-value", type=float, default=None)
    parser.add_argument("--quantity-unit", default=None)
    parser.add_argument("--quantity-scale", type=float, default=None)
    parser.add_argument("--axis-convention", default=None)
    parser.add_argument(
        "--mirror-left-to-right",
        action="store_true",
        help=(
            "mirror base/left-wrist images and map left state/action into the CR1 "
            "right-arm convention before right-arm FK segmentation"
        ),
    )
    parser.add_argument(
        "--mirror-output-dir",
        type=Path,
        default=None,
        help="optionally materialize mirrored videos and trajectory.npz before annotation",
    )
    parser.add_argument(
        "--overwrite-mirror",
        action="store_true",
        help="replace existing files under --mirror-output-dir",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="run an optional second full-video boundary/classification review",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    quantity = (args.quantity_value, args.quantity_unit, args.quantity_scale)
    if any(value is not None for value in quantity) and not all(
        value is not None for value in quantity
    ):
        raise ValueError(
            "--quantity-value, --quantity-unit, and --quantity-scale must be supplied together"
        )
    if args.mirror_output_dir is not None and not args.mirror_left_to_right:
        raise ValueError("--mirror-output-dir requires --mirror-left-to-right")
    if args.overwrite_mirror and args.mirror_output_dir is None:
        raise ValueError("--overwrite-mirror requires --mirror-output-dir")
    if args.mirror_left_to_right and args.lerobot_root is None:
        raise ValueError("--mirror-left-to-right requires --lerobot-root")
    if args.lerobot_left_video_key is not None and not args.mirror_left_to_right:
        raise ValueError("--lerobot-left-video-key requires --mirror-left-to-right")
    config_kwargs = {}
    if args.config:
        config_kwargs.update(json.loads(args.config.read_text(encoding="utf-8")))
    for name in (
        "sample_fps",
        "max_frames",
        "min_segment_duration_s",
    ):
        value = getattr(args, name)
        if value is not None:
            config_kwargs[name] = value
    if args.review:
        config_kwargs["review"] = True
    if args.axis_convention:
        config_kwargs["axis_convention"] = args.axis_convention
    config = PipelineConfig(**config_kwargs)
    client = create_annotation_client(
        args.provider,
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
    )
    pipeline = AtomicSegmentationPipeline(client, config)

    lerobot_episode = None
    if args.lerobot_root is not None:
        if args.episode_index is None:
            raise ValueError("--episode-index is required with --lerobot-root")
        if args.video or args.tcp_trace or args.joint_trace:
            raise ValueError(
                "--lerobot-root resolves videos and motion automatically; do not also pass "
                "--video, --tcp-trace, or --joint-trace"
            )
        lerobot_episode = load_lerobot_episode(
            args.lerobot_root,
            args.episode_index,
            base_video_key=args.lerobot_base_video_key,
            left_video_key=args.lerobot_left_video_key,
            right_video_key=args.lerobot_right_video_key,
            mirror_left_to_right=args.mirror_left_to_right,
        )
        videos = list(lerobot_episode.videos)
        horizontal_flip = list(lerobot_episode.horizontal_flip)
        if args.mirror_output_dir is not None:
            bundle = write_mirrored_episode_bundle(
                output_dir=args.mirror_output_dir,
                source_root=lerobot_episode.root,
                source_episode_index=lerobot_episode.episode_index,
                task=args.task or lerobot_episode.task,
                source_base_video=lerobot_episode.videos[0],
                source_left_wrist_video=lerobot_episode.videos[1],
                timestamps=lerobot_episode.timestamps,
                right_state=lerobot_episode.right_state,
                right_action=lerobot_episode.right_action,
                overwrite=args.overwrite_mirror,
            )
            videos = list(bundle.videos)
            horizontal_flip = [False, False]
        view_names = ["base", "right_wrist"]
        task = args.task or lerobot_episode.task
        if not task:
            raise ValueError("the LeRobot episode has no task metadata; supply --task explicitly")
        default_episode_id = f"episode_{args.episode_index:06d}"
        if args.mirror_left_to_right:
            default_episode_id += "_mirror_lr"
        episode_id = args.episode_id or default_episode_id
    else:
        if args.episode_index is not None:
            raise ValueError("--episode-index requires --lerobot-root")
        if not args.video:
            raise ValueError("pass --lerobot-root/--episode-index or at least one --video")
        if not args.task:
            raise ValueError("--task is required in manual --video mode")
        videos = args.video
        view_names = args.view_name
        horizontal_flip = None
        task = args.task
        episode_id = args.episode_id or "episode_0"

    urdf = args.urdf
    if (args.joint_trace or lerobot_episode is not None) and urdf is None:
        arm = "right" if lerobot_episode is not None else args.arm
        suffix = "L" if arm == "left" else "R"
        urdf = DEFAULT_CR1_URDF_DIR / f"CR1ARM{suffix}.urdf"
    motion_trace = (
        lerobot_episode.make_trace(
            urdf_path=urdf,
            tcp_frame=args.tcp_frame,
            mount_xyz=tuple(args.mount_xyz) if args.mount_xyz else None,
        )
        if lerobot_episode is not None
        else None
    )
    result = pipeline.run(
        videos=videos,
        view_names=view_names,
        task=task,
        episode_id=episode_id,
        extra_context=args.extra_context,
        tcp_trace=args.tcp_trace,
        joint_trace=args.joint_trace,
        motion_trace=motion_trace,
        arm=args.arm,
        urdf_path=urdf,
        tcp_frame=args.tcp_frame,
        mount_xyz=tuple(args.mount_xyz) if args.mount_xyz else None,
        quantity_value=args.quantity_value,
        quantity_unit=args.quantity_unit,
        quantity_scale=args.quantity_scale,
        horizontal_flip=horizontal_flip,
        augmentation=(
            lerobot_episode.augmentation if lerobot_episode is not None else "none"
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    supervised = sum(bool(segment.atomic_targets) for segment in result.segments)
    print(f"saved {len(result.segments)} segments ({supervised} supervised) to {args.output}")


if __name__ == "__main__":
    main()
