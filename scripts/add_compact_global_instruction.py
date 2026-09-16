#!/usr/bin/env python3
"""Use Qwen to add a short, training-facing global instruction to annotations."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from atomic_latent_vla.annotation.client import QwenVLPlusClient


def summarize(client: QwenVLPlusClient, data: dict, max_words: int) -> str:
    source = str(data.get("task") or data.get("global_description") or "")
    answer, _ = client.complete_json([
        {"role": "system", "content": (
            "You write LeRobot task instructions. Infer the overall physical robot task from the supplied Qwen audit summary. "
            "Return JSON only: {\"global_instruction\": \"...\"}. The value must be one concrete English sentence of at most "
            f"{max_words} words; preserve objects/goal if known; omit camera, narration, arm-side, uncertainty, procedural detail, "
            "all counts, grid dimensions (for example 3x3), coordinates, and numerical measurements."
        )},
        {"role": "user", "content": source},
    ], max_tokens=96)
    value = " ".join(str(answer.get("global_instruction", "")).split())
    if not value:
        raise ValueError("Qwen returned an empty global_instruction")
    words = value.split()
    for limit in (max_words, 24, 18):
        if value and len(value.split()) <= max_words:
            break
        answer, _ = client.complete_json([
            {"role": "system", "content": f"Write one concrete English robot task sentence using at most {limit} words. Preserve only the main goal and objects; omit counts, dimensions, coordinates, measurements, camera, arm side, and procedural detail. Return JSON only: {{\"global_instruction\": \"...\"}}."},
            {"role": "user", "content": value or source},
        ], max_tokens=64)
        value = " ".join(str(answer.get("global_instruction", "")).split())
    if not value or len(value.split()) > max_words:
        raise ValueError(f"Qwen still exceeded {max_words} words: {value!r}")
    return value


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("annotation_dirs", nargs="+", type=Path)
    p.add_argument("--max-words", type=int, default=32)
    p.add_argument("--model", default="qwen3-vl-plus")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--names", nargs="*", default=None, help="annotation filenames to process")
    p.add_argument("--copy-global-from", type=Path, default=None,
                   help="copy existing global_instruction by normalized source stem instead of calling Qwen")
    args = p.parse_args()
    copied: dict[str, str] = {}
    if args.copy_global_from is not None:
        for path in args.copy_global_from.glob("*.json"):
            data = json.loads(path.read_text())
            if data.get("global_instruction"):
                copied[path.stem.removesuffix("_mirror_lr")] = str(data["global_instruction"])
    client = None if copied else QwenVLPlusClient(model=args.model, api_key=os.environ.get("DASHSCOPE_API_KEY"), timeout_s=180)
    changed = skipped = 0
    for directory in args.annotation_dirs:
        paths = sorted(directory.glob("*.json"))
        if args.names is not None:
            names = set(args.names)
            paths = [path for path in paths if path.name in names]
        paths = paths[args.start:] if args.limit is None else paths[args.start:args.start + args.limit]
        for path in paths:
            data = json.loads(path.read_text())
            if data.get("global_instruction") and not args.overwrite:
                skipped += 1
                continue
            if copied:
                value = copied.get(path.stem.removesuffix("_mirror_lr"))
                if not value:
                    raise KeyError(f"no source global_instruction for {path.name}")
            else:
                assert client is not None
                value = summarize(client, data, args.max_words)
            if data.get("global_instruction") != value:
                data["global_instruction"] = value
                path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
                changed += 1
                print(f"UPDATED {path.name}", flush=True)
    print(json.dumps({"updated": changed, "skipped": skipped, "max_words": args.max_words}))


if __name__ == "__main__":
    main()
