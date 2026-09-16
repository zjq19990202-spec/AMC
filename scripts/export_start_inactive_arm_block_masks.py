#!/usr/bin/env python3
"""Export per-dataset training masks from the reviewed primary-arm audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-path-ratio", type=float, default=3.0)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    for dataset in audit["datasets"]:
        records = [
            row for row in dataset["records"]
            if float(row["path_ratio"]) >= args.min_path_ratio
        ]
        payload = {
            "version": 1,
            "source_audit": str(args.audit.resolve()),
            "dataset_root": dataset["dataset_root"],
            "atomic_block_frames": 10,
            "min_primary_to_inactive_path_ratio": args.min_path_ratio,
            "application": "exclude horizon starts with frame_index < mask_end_frame_exclusive",
            "records": records,
        }
        name = Path(dataset["dataset_root"]).name
        output = args.output_root / name / "start_inactive_arm_block_mask.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        print(name, len(records), sum(float(row["mask_seconds"]) for row in records), output)


if __name__ == "__main__":
    main()
