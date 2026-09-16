#!/usr/bin/env python3
"""Numerically verify CR1 full-vector left/right mirror consistency."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

SIGN = np.asarray([1, -1, -1, 1, -1, 1, -1, 1], dtype=np.float32)


def parquet(root: Path) -> Path:
    files = sorted((root / "data").rglob("*.parquet"))
    if len(files) != 1: raise RuntimeError(f"Expected one data parquet under {root}, got {files}")
    return files[0]


def expected(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2 or x.shape[1] != 16: raise ValueError(f"Expected [N,16], got {x.shape}")
    y = np.empty_like(x); y[:, :8] = x[:, 8:] * SIGN; y[:, 8:] = x[:, :8] * SIGN
    return y


def main(source: Path, mirror: Path) -> None:
    a = pq.read_table(parquet(source), columns=["episode_index", "frame_index", "observation.state", "action"])
    b = pq.read_table(parquet(mirror), columns=["episode_index", "frame_index", "observation.state", "action"])
    if a.num_rows != b.num_rows: raise AssertionError(f"row count differs: {a.num_rows} != {b.num_rows}")
    for key in ("episode_index", "frame_index"):
        if a.column(key).to_pylist() != b.column(key).to_pylist(): raise AssertionError(f"{key} changed")
    out = {}
    for key in ("observation.state", "action"):
        src = np.asarray(a.column(key).to_pylist(), dtype=np.float32); dst = np.asarray(b.column(key).to_pylist(), dtype=np.float32)
        err = float(np.max(np.abs(expected(src) - dst)))
        if err != 0.0: raise AssertionError(f"{key} mirror mismatch: max_abs_error={err}")
        out[key] = {"shape": list(dst.shape), "max_abs_error": err}
    print(json.dumps({"source": str(source), "mirror": str(mirror), **out}, ensure_ascii=False))


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--source",type=Path,required=True); p.add_argument("--mirror",type=Path,required=True)
    args=p.parse_args(); main(args.source,args.mirror)
