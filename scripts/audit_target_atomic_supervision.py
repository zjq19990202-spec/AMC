#!/usr/bin/env python3
"""Fail-fast audit for target's bimanual 3 Hz atomic supervision contract."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from atomic_latent_vla.pi05.training_data import _TargetAnnotationSidecars


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()

    root = args.dataset_root
    sidecars = _TargetAnnotationSidecars(root)
    lengths: dict[int, int] = {}
    manifest = root / "meta" / "fk_horizon_3hz" / "manifest.jsonl"
    with manifest.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            lengths[int(row["episode_index"])] = int(row["length"])

    block_counts: Counter[tuple[str, str]] = Counter()
    frame_counts: Counter[tuple[str, str]] = Counter()
    representatives: dict[tuple[str, str], tuple[int, int, object]] = {}
    for episode_index, by_arm in sidecars.atomic_horizons.items():
        episode_length = lengths[episode_index]
        for arm in ("right", "left"):
            for block, annotation in by_arm.get(arm, {}).items():
                positive = np.flatnonzero(annotation.weights)
                if positive.tolist() == [12]:
                    kind = "stay"
                elif len(positive) == 1:
                    kind = "single"
                elif len(positive) == 2:
                    kind = "dual"
                else:
                    raise AssertionError(
                        f"invalid {arm} target at episode={episode_index} block={block}: "
                        f"positive={positive.tolist()}"
                    )
                block_counts[(arm, kind)] += 1
                frame_counts[(arm, kind)] += max(
                    0, min((block + 1) * 10, episode_length) - block * 10
                )
                representatives.setdefault(
                    (arm, kind), (episode_index, block, annotation)
                )

    for arm in ("right", "left"):
        for kind in ("single", "dual", "stay"):
            if block_counts[(arm, kind)] <= 0:
                raise AssertionError(f"missing {arm}/{kind} supervision")
            episode_index, block, expected = representatives[(arm, kind)]
            episode_length = lengths[episode_index]
            for frame in range(block * 10, min((block + 1) * 10, episode_length)):
                actual = sidecars.atomic_horizon(episode_index, frame, arm=arm)
                if actual is not expected:
                    raise AssertionError(
                        f"3 Hz coverage failed: {arm} episode={episode_index} "
                        f"block={block} frame={frame}"
                    )

    total_frames = sum(lengths.values())
    result = {
        "codebook_contract": "2 arms x 13 atoms = 26 distinct prototypes",
        "block_counts_3hz": {
            f"{arm}/{kind}": block_counts[(arm, kind)]
            for arm in ("right", "left")
            for kind in ("single", "dual", "stay")
        },
        "covered_start_frames_30hz": {
            f"{arm}/{kind}": frame_counts[(arm, kind)]
            for arm in ("right", "left")
            for kind in ("single", "dual", "stay")
        },
        "total_dataset_frames": total_frames,
        "supervised_arm_slot_fraction": (
            sum(frame_counts.values()) / (2.0 * total_frames)
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
