from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .client import create_annotation_client
from .pipeline import AtomicSegmentationPipeline, PipelineConfig


class ManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str = Field(min_length=1)
    task: str = Field(min_length=1)
    videos: list[str] = Field(min_length=1)
    view_names: list[str] | None = None
    tcp_trace: str | None = None
    joint_trace: str | None = None
    arm: str = Field(default="left", pattern="^(left|right)$")
    urdf: str | None = None
    tcp_frame: str | None = None
    mount_xyz: tuple[float, float, float] | None = None
    extra_context: str = ""
    quantity_value: float | None = None
    quantity_unit: str | None = None
    quantity_scale: float | None = Field(default=None, gt=0)

    @classmethod
    def default_urdf(cls, arm: str) -> str:
        suffix = "L" if arm == "left" else "R"
        return f"/home/admin123/cr1_recordclient/model/CR1_UPPER/urdf/CR1ARM{suffix}.urdf"

    @model_validator(mode="after")
    def validate_motion_source(self) -> "ManifestEntry":
        if self.tcp_trace and self.joint_trace:
            raise ValueError("tcp_trace and joint_trace are mutually exclusive")
        quantity = (self.quantity_value, self.quantity_unit, self.quantity_scale)
        if any(value is not None for value in quantity) and not all(
            value is not None for value in quantity
        ):
            raise ValueError(
                "quantity_value, quantity_unit, and quantity_scale must be supplied together"
            )
        return self


def load_manifest(path: Path) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entries.append(ManifestEntry.model_validate_json(line))
        except Exception as error:
            raise ValueError(f"invalid manifest line {line_number}: {error}") from error
    if not entries:
        raise ValueError("manifest is empty")
    return entries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch atomic segmentation from a JSONL manifest")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path)
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
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_payload = {}
    if args.config:
        config_payload = json.loads(args.config.read_text(encoding="utf-8"))
    config = PipelineConfig(**config_payload)
    client = create_annotation_client(
        args.provider,
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
    )
    pipeline = AtomicSegmentationPipeline(client, config)
    entries = load_manifest(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures_path = args.output_dir / "failures.jsonl"
    failures: list[dict[str, str]] = []

    completed = 0
    skipped = 0
    for entry in entries:
        output_path = args.output_dir / f"{entry.episode_id}.json"
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            result = pipeline.run(
                videos=entry.videos,
                view_names=entry.view_names,
                task=entry.task,
                episode_id=entry.episode_id,
                extra_context=entry.extra_context,
                tcp_trace=entry.tcp_trace,
                joint_trace=entry.joint_trace,
                arm=entry.arm,
                urdf_path=entry.urdf or entry.default_urdf(entry.arm),
                tcp_frame=entry.tcp_frame,
                mount_xyz=entry.mount_xyz,
                quantity_value=entry.quantity_value,
                quantity_unit=entry.quantity_unit,
                quantity_scale=entry.quantity_scale,
            )
            output_path.write_text(
                json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            completed += 1
            print(f"[{completed + skipped}/{len(entries)}] saved {output_path}")
        except Exception as error:
            failure = {"episode_id": entry.episode_id, "error": str(error)}
            failures.append(failure)
            print(f"failed {entry.episode_id}: {error}")
            if args.stop_on_error:
                break

    if failures:
        failures_path.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in failures),
            encoding="utf-8",
        )
    elif failures_path.exists():
        failures_path.unlink()
    print(f"complete: {completed} written, {skipped} skipped, {len(failures)} failed")


if __name__ == "__main__":
    main()
