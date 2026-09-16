#!/usr/bin/env python3
"""Inject Qwen/FK atomic annotations into a completed original-shape LeRobot set.

Every source episode is retained.  Episodes without an annotation JSON receive an
explicit drop prompt and a zero 12-D atom vector, rather than disappearing from
the dataset.  Source timestamps are preserved verbatim, so Qwen boundaries in
seconds align with their original 30 Hz HDF frames.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from atomic_latent_vla.data.atomic_ratios import local_atomic_weight_rows

N_ATOMS = 12
DEFAULT_PROMPT = "No stable atomic motion is annotated for this frame."


def fixed(values: np.ndarray, width: int) -> pa.Array:
    return pa.FixedSizeListArray.from_arrays(
        pa.array(np.asarray(values, dtype=np.float32).reshape(-1), type=pa.float32()), width
    )


def load_annotations(directory: Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for path in sorted(directory.glob("*.json")):
        obj = json.loads(path.read_text())
        key = str(obj.get("episode_id") or path.stem)
        # The physical left/right augmentation annotations deliberately carry
        # ``_mirror_lr`` in their identity.  LeRobot's source map remains tied
        # to the original HDF filename, so normalize only that documented
        # suffix when matching; retain the JSON itself unchanged.
        if key.endswith("_mirror_lr"):
            key = key.removesuffix("_mirror_lr")
        if key in result:
            raise ValueError(f"Duplicate annotation identity after normalization: {key}")
        result[key] = obj
    if not result:
        raise FileNotFoundError(f"No annotation JSON found in {directory}")
    return result


def segment_targets(segment: dict) -> list[tuple[int, float]]:
    out = []
    for item in segment.get("atomic_targets", []):
        label = int(item["label"])
        # ``confidence`` is the intended soft target weight for one/two atoms.
        out.append((label, float(item.get("confidence", item.get("weight", 1.0)))))
    return out


def label_frames(times: np.ndarray, annotation: dict | None):
    n = len(times)
    segment_id = np.full(n, -1, dtype=np.int64)
    typ = np.zeros(n, dtype=np.int8)  # 0 = drop/audit, 1 = atomic supervision
    prompt = np.full(n, DEFAULT_PROMPT, dtype=object)
    atoms = np.zeros((n, N_ATOMS), dtype=np.float32)
    if annotation is None:
        return segment_id, typ, prompt, atoms

    for order, segment in enumerate(annotation.get("segments", [])):
        start, end = float(segment["start_s"]), float(segment["end_s"])
        hit = (times >= start - 1e-6) & (times < end - 1e-6)
        targets = segment_targets(segment)
        eligible = (
            bool(segment.get("training_eligible", False))
            and segment.get("gate_mode") in {"single", "dual"}
            and bool(targets)
        )
        segment_id[hit] = int(segment.get("segment_id", order))
        prompt[hit] = str(segment.get("low_level_instruction") or DEFAULT_PROMPT)
        if eligible:
            fallback = np.zeros(N_ATOMS, dtype=np.float32)
            typ[hit] = 1
            for label, weight in targets:
                if not 0 <= label < N_ATOMS:
                    raise ValueError(f"invalid atom label {label}")
                fallback[label] = weight
            atoms[hit] = local_atomic_weight_rows(times[hit], segment, fallback)
    return segment_id, typ, prompt, atoms


def only_parquet(root: Path) -> Path:
    paths = sorted(root.rglob("*.parquet"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected exactly one completed parquet under {root}, found {paths}")
    return paths[0]


def main(dataset: Path, annotations_dir: Path, default_task: str) -> None:
    source_map_path = dataset / "source_episode_map.json"
    source_map = json.loads(source_map_path.read_text())
    annotations = load_annotations(annotations_dir)
    data_path = only_parquet(dataset / "data")
    table = pq.read_table(data_path)
    values = table.to_pydict()
    episode = np.asarray(values["episode_index"], dtype=np.int64)
    timestamp = np.asarray(values["timestamp"], dtype=np.float64)

    segment_id = np.full(len(episode), -1, dtype=np.int64)
    typ = np.zeros(len(episode), dtype=np.int8)
    prompt = np.full(len(episode), DEFAULT_PROMPT, dtype=object)
    atoms = np.zeros((len(episode), N_ATOMS), dtype=np.float32)
    task_by_ep: dict[int, str] = {}
    annotated = 0
    for ep_str, filename in source_map.items():
        ep = int(ep_str)
        key = Path(filename).stem
        ann = annotations.get(key)
        mask = episode == ep
        if not mask.any():
            raise RuntimeError(f"episode {ep} ({filename}) is missing from data parquet")
        sid, kind, text, atom = label_frames(timestamp[mask], ann)
        segment_id[mask], typ[mask], prompt[mask], atoms[mask] = sid, kind, text, atom
        task_by_ep[ep] = str((ann or {}).get("global_instruction") or (ann or {}).get("task") or (ann or {}).get("global_description") or default_task)
        annotated += int(ann is not None)

    task_index: OrderedDict[str, int] = OrderedDict()
    for ep in sorted(task_by_ep):
        task_index.setdefault(task_by_ep[ep], len(task_index))
    columns = {name: table.column(name) for name in table.column_names
               if name not in {"task_index", "annotation.segment_id", "annotation.type", "annotation.prompt", "annotation.atoms"}}
    columns["task_index"] = pa.array([task_index[task_by_ep[int(ep)]] for ep in episode], type=pa.int64())
    columns["annotation.segment_id"] = pa.array(segment_id, type=pa.int64())
    columns["annotation.type"] = pa.array(typ, type=pa.int8())
    columns["annotation.prompt"] = pa.array(prompt, type=pa.string())
    columns["annotation.atoms"] = fixed(atoms, N_ATOMS)
    replacement = data_path.with_suffix(".annotating")
    pq.write_table(pa.table(columns), replacement, compression="zstd")
    os.replace(replacement, data_path)

    info_path = dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    for name, dtype, shape in (
        ("annotation.segment_id", "int64", [1]),
        ("annotation.type", "int8", [1]),
        ("annotation.prompt", "string", [1]),
        ("annotation.atoms", "float32", [N_ATOMS]),
    ):
        info["features"][name] = {"dtype": dtype, "shape": shape,
                                  "names": None if shape == [1] else [f"atom_{i}" for i in range(N_ATOMS)]}
    info["total_tasks"] = len(task_index)
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")
    pq.write_table(pa.table({"task_index": pa.array(list(task_index.values()), type=pa.int64()),
                             "task": pa.array(list(task_index.keys()), type=pa.string())}),
                   dataset / "meta" / "tasks.parquet", compression="zstd")

    episodes_path = only_parquet(dataset / "meta" / "episodes")
    episodes = pq.read_table(episodes_path).to_pylist()
    for row in episodes:
        ep = int(row["episode_index"])
        row["tasks"] = [task_by_ep[ep]]
    replacement = episodes_path.with_suffix(".annotating")
    pq.write_table(pa.Table.from_pylist(episodes), replacement, compression="zstd")
    os.replace(replacement, episodes_path)
    summary = {"dataset": str(dataset), "source_episodes": len(source_map), "annotated_episodes": annotated,
               "unannotated_drop_episodes": len(source_map) - annotated, "frames": int(len(episode)),
               "atomic_frames": int(typ.sum()), "tasks": len(task_index)}
    (dataset / "atomic_annotation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--default-task", default="recorded CR1 bimanual manipulation")
    args = parser.parse_args()
    main(args.dataset, args.annotations, args.default_task)
