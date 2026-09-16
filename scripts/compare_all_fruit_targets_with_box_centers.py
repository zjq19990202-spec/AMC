#!/usr/bin/env python3
"""Compare model endpoints to fruit contacts or estimated per-episode box centers."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


FRUITS = ("carrot", "orange", "green bitter melon", "green radish", "yellow pear", "banana", "red chili pepper")


def fruit_in(text: str) -> str | None:
    text = text.lower()
    if "orange pumpkin" in text:
        return None
    for fruit in sorted(FRUITS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(fruit)}\b", text):
            return fruit
    if "yellow banana" in text:
        return "banana"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--plain-json", type=Path, required=True)
    ap.add_argument("--afro-json", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    table = pq.read_table(args.dataset_root / "data/chunk-000/file-000.parquet", columns=["episode_index", "frame_index", "action", "index"])
    episodes = np.asarray(table["episode_index"])
    frames = np.asarray(table["frame_index"])
    actions = np.asarray(table["action"].to_pylist())
    indices = np.asarray(table["index"])
    tcp = np.load(args.dataset_root / "meta/tcp_pose_bimanual_base_tcp200.npy", mmap_mode="r")
    segments = {int(x["episode_index"]): x["semantic_segments"] for x in map(json.loads, (args.dataset_root / "meta/episode_subtasks.jsonl").open())}
    selection = json.loads(args.selection.read_text())["datasets"]["target2058"]
    plain = json.loads(args.plain_json.read_text())
    afro = json.loads(args.afro_json.read_text())
    plain_rows = {(int(r["episode"]), int(r["frame"])): r for r in plain["rows"]}
    afro_rows = {(int(r["episode"]), int(r["frame"])): r for r in afro["results"]}

    def row_at(ep: int, frame: int) -> int:
        found = np.flatnonzero((episodes == ep) & (frames == frame))
        if not len(found):
            raise KeyError((ep, frame))
        return int(found[0])

    def arm_for_interval(ep: int, start: int, end: int) -> str:
        chunk = actions[(episodes == ep) & (frames >= start) & (frames < end)]
        left = np.linalg.norm(chunk[-1, :7] - chunk[0, :7])
        right = np.linalg.norm(chunk[-1, 8:15] - chunk[0, 8:15])
        return "left" if left > right else "right"

    def scene_targets(ep: int, source_frame: int):
        segs = segments[ep]
        placements = []
        contacts = {}
        for i, seg in enumerate(segs):
            text = seg["current_subtask"].lower()
            fruit = fruit_in(text)
            box_match = re.search(r"\b(left|center|right) box\b", text)
            if not (text.startswith("lift ") or text.startswith("raise ")):
                continue
            start = int(seg["start_frame_30hz"])
            end = int(segs[i + 1]["start_frame_30hz"]) if i + 1 < len(segs) else int(frames[episodes == ep].max()) + 1
            arm = arm_for_interval(ep, start, end)
            offset = 0 if arm == "left" else 24
            # Every lift/raise segment provides a grasp-contact position at its
            # first frame, including objects that are later returned to the
            # tabletop rather than placed in a box.
            contact_row = row_at(ep, start)
            if fruit is not None:
                contacts[fruit] = {"xyz": np.asarray(tcp[int(indices[contact_row]), offset : offset + 3], dtype=float), "frame": start, "arm": arm}
            # A release endpoint can estimate a box center only when the
            # segment explicitly names a destination box.
            if box_match is None:
                continue
            interval_rows = np.flatnonzero((episodes == ep) & (frames >= start) & (frames < end))
            grip_column = 7 if arm == "left" else 15
            grip = actions[interval_rows, grip_column]
            # The placement endpoint is the object-release instant, not the
            # later retracted pose at the semantic boundary. Open is 1.0.
            release_local = int(np.argmax(np.diff(grip)) + 1) if len(grip) > 1 else len(grip) - 1
            release_row = int(interval_rows[release_local])
            endpoint = np.asarray(tcp[int(indices[release_row]), offset : offset + 3], dtype=float)
            placements.append({"fruit": fruit, "box": box_match.group(1), "start": start, "end": end, "arm": arm, "endpoint": endpoint})
        box_points = {}
        for box in ("left", "center", "right"):
            pts = [p["endpoint"] for p in placements if p["box"] == box]
            if pts:
                box_points[box] = {"xyz": np.mean(pts, axis=0), "n": len(pts), "spread_mm": float(np.sqrt(np.mean(np.sum((np.asarray(pts) - np.mean(pts, axis=0)) ** 2, axis=1))) * 1000)}
        targets = {}
        for fruit in FRUITS:
            earlier = [p for p in placements if p["fruit"] == fruit and p["start"] < source_frame]
            if earlier:
                placement = earlier[-1]
                estimate = box_points[placement["box"]]
                targets[fruit] = {"xyz": estimate["xyz"], "kind": "placed_box_center", "box": placement["box"], "box_samples": estimate["n"], "box_spread_mm": estimate["spread_mm"]}
            elif fruit in contacts and contacts[fruit]["frame"] >= source_frame:
                targets[fruit] = {"xyz": contacts[fruit]["xyz"], "kind": "future_grasp_contact", "contact_frame": contacts[fruit]["frame"]}
        return targets, box_points

    rows = []
    box_reports = {}
    for spec in selection:
        ep, frame, arm = int(spec["episode"]), int(spec["frame"]), spec["active_arm"]
        targets, boxes = scene_targets(ep, frame)
        box_reports[f"{ep}:{frame}"] = {name: {"xyz_m": value["xyz"].tolist(), "n": value["n"], "spread_mm": value["spread_mm"]} for name, value in boxes.items()}
        source = row_at(ep, frame)
        offset = 0 if arm == "left" else 24
        start = np.asarray(tcp[int(indices[source]), offset : offset + 3], dtype=float)
        pr, ar = plain_rows[(ep, frame)], afro_rows[(ep, frame)]
        plain_ends = dict(zip(FRUITS, np.asarray(pr[f"{arm}_endpoint_xyz_m"]), strict=True))
        afro_ends = {fruit: start + np.asarray(delta) / 1000 for fruit, delta in ar["endpoint_displacement_mm"].items()}
        for fruit, info in targets.items():
            target = info["xyz"]
            initial = float(np.linalg.norm(start - target) * 1000)
            pd = float(np.linalg.norm(plain_ends[fruit] - target) * 1000)
            ad = float(np.linalg.norm(afro_ends[fruit] - target) * 1000)
            rows.append({"episode": ep, "frame": frame, "active_arm": arm, "placed_fruits": spec["placed_fruits"], "fruit": fruit, "target_kind": info["kind"], "target_xyz_m": target.tolist(), "initial_distance_mm": initial, "plain_distance_mm": pd, "afro_distance_mm": ad, "plain_reduction_mm": initial - pd, "afro_reduction_mm": initial - ad, "plain_minus_afro_mm": pd - ad, **{k: v for k, v in info.items() if k != "xyz"}})

    def aggregate(selected):
        diff = np.asarray([r["plain_minus_afro_mm"] for r in selected])
        return {"n": len(selected), "plain_mean_mm": float(np.mean([r["plain_distance_mm"] for r in selected])), "afro_mean_mm": float(np.mean([r["afro_distance_mm"] for r in selected])), "afro_gain_mm": float(diff.mean()), "afro_wins": int(np.sum(diff > 0)), "plain_wins": int(np.sum(diff < 0)), "afro_wins_over_10mm": int(np.sum(diff > 10)), "plain_wins_over_10mm": int(np.sum(diff < -10)), "median_plain_minus_afro_mm": float(np.median(diff))}

    report = {"summary": {"all": aggregate(rows), "placed_box_center": aggregate([r for r in rows if r["target_kind"] == "placed_box_center"]), "future_grasp_contact": aggregate([r for r in rows if r["target_kind"] == "future_grasp_contact"])}, "box_estimates": box_reports, "rows": rows}
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
