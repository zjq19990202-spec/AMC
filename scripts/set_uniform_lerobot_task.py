#!/usr/bin/env python3
"""Replace every LeRobot episode task with one shared training global task."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def only_parquet(root: Path) -> Path:
    files = sorted(root.rglob("*.parquet"))
    if len(files) != 1:
        raise RuntimeError(f"expected one parquet under {root}, found {len(files)}")
    return files[0]


def replace(path: Path, table: pa.Table) -> None:
    temporary = path.with_suffix(".updating")
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, path)


def update(root: Path, task: str) -> None:
    data_path = only_parquet(root / "data")
    data = pq.read_table(data_path)
    cols = {name: data.column(name) for name in data.column_names if name != "task_index"}
    cols["task_index"] = pa.array([0] * data.num_rows, type=pa.int64())
    replace(data_path, pa.table(cols))

    pq.write_table(pa.table({"task_index": pa.array([0], type=pa.int64()), "task": pa.array([task])}),
                   root / "meta" / "tasks.parquet", compression="zstd")
    episodes_path = only_parquet(root / "meta" / "episodes")
    rows = pq.read_table(episodes_path).to_pylist()
    for row in rows:
        row["tasks"] = [task]
    replace(episodes_path, pa.Table.from_pylist(rows))
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["total_tasks"] = 1
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"dataset": str(root), "episodes": len(rows), "task": task}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("datasets", nargs="+", type=Path)
    args = parser.parse_args()
    for dataset in args.datasets:
        update(dataset, args.task)
