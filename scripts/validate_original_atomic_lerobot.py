#!/usr/bin/env python3
"""Structural verifier for original-shape prompt-aligned LeRobot outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("root", type=Path); a = p.parse_args()
    root=a.root; data_files=list((root/"data").rglob("*.parquet")); assert data_files, "missing data parquet"
    table=pq.ParquetFile(data_files[0]).read(columns=["observation.state","action","annotation.type","annotation.prompt","annotation.atoms","episode_index"])
    d=table.to_pydict(); assert d, "empty parquet"; assert len(d["observation.state"][0])==16; assert len(d["action"][0])==16; assert len(d["annotation.atoms"][0])==12
    assert all(x in (0,1) for x in d["annotation.type"]); assert all(bool(x) for x in d["annotation.prompt"])
    videos=list((root/"videos").rglob("*.mp4")); assert videos, "missing videos"
    info=json.loads((root/"meta"/"info.json").read_text()); features=info["features"]
    for key in ("observation.images.base_0_rgb","observation.images.left_wrist_0_rgb","observation.images.right_wrist_0_rgb","annotation.segment_id","annotation.type","annotation.prompt","annotation.atoms"):
        assert key in features, f"missing feature {key}"
    print(json.dumps({"root":str(root),"frames":len(d["episode_index"]),"episodes":len(set(d["episode_index"])),"videos":len(videos),"atomic_frames":sum(d["annotation.type"]),"state_dim":16,"action_dim":16},ensure_ascii=False))

if __name__ == "__main__": main()
