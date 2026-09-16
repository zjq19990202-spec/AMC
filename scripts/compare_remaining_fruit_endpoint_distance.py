#!/usr/bin/env python3
"""Compare PI0.5 and AFRO endpoint distance to each still-unpicked fruit."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


FRUITS = ["carrot", "orange", "green bitter melon", "green radish", "yellow pear", "banana", "red chili pepper"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--plain-json", type=Path, required=True)
    ap.add_argument("--afro-json", type=Path, required=True)
    ap.add_argument("--film-json", type=Path)
    ap.add_argument("--film-label", default="FiLM all18 60K")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    plain = json.loads(args.plain_json.read_text()); afro = json.loads(args.afro_json.read_text())
    film = json.loads(args.film_json.read_text()) if args.film_json else None
    table = pq.read_table(args.dataset_root / "data/chunk-000/file-000.parquet", columns=["episode_index", "frame_index", "action", "index"])
    eps=np.asarray(table["episode_index"]); frames=np.asarray(table["frame_index"]); acts=np.asarray(table["action"].to_pylist()); indices=np.asarray(table["index"])
    tcp=np.load(args.dataset_root / "meta/tcp_pose_bimanual_base_tcp200.npy", mmap_mode="r")
    segs={}
    for line in (args.dataset_root / "meta/episode_subtasks.jsonl").open():
        item=json.loads(line); segs[int(item["episode_index"])]=item["semantic_segments"]

    def positions(ep: int, source_frame: int):
        out={}
        for i,s in enumerate(segs[ep]):
            text=s["current_subtask"].lower(); boundary=int(s["start_frame_30hz"])
            if boundary < source_frame or "box" not in text or not (text.startswith("lift ") or text.startswith("raise ")): continue
            fruit=next((x for x in FRUITS if re.search(rf"\b{re.escape(x)}\b", text)),None)
            if fruit is None: continue
            begin=int(segs[ep][max(0,i-1)]["start_frame_30hz"]); mask=(eps==ep)&(frames>=begin)&(frames<boundary); chunk=acts[mask]
            left=np.linalg.norm(chunk[-1,:7]-chunk[0,:7]); right=np.linalg.norm(chunk[-1,8:15]-chunk[0,8:15]); contact_arm="left" if left>right else "right"
            row=np.flatnonzero((eps==ep)&(frames==boundary))[0]; offset=0 if contact_arm=="left" else 24
            out[fruit]={"xyz_m":np.asarray(tcp[int(indices[row]),offset:offset+3],dtype=float),"contact_frame":boundary,"recorded_arm":contact_arm}
        return out

    prows={(r["episode"],r["frame"]):r for r in plain["rows"]}; arows={(r["episode"],r["frame"]):r for r in afro["results"]}; frows={(r["episode"],r["frame"]):r for r in film["results"]} if film else {}; results=[]
    for key,pr in prows.items():
        ep,frame=map(int,key); ar=arows[key]; arm=pr["active_arm"]; available=positions(ep,frame)
        source_row=np.flatnonzero((eps==ep)&(frames==frame))[0]; offset=0 if arm=="left" else 24; start=np.asarray(tcp[int(indices[source_row]),offset:offset+3],dtype=float)
        plain_end=dict(zip(FRUITS,np.asarray(pr[f"{arm}_endpoint_xyz_m"]),strict=True))
        afro_end={fruit:start+np.asarray(delta,dtype=float)/1000 for fruit,delta in ar["endpoint_displacement_mm"].items()}
        film_end={fruit:start+np.asarray(delta,dtype=float)/1000 for fruit,delta in frows[key]["endpoint_displacement_mm"].items()} if film else {}
        for fruit,target_info in available.items():
            target=target_info["xyz_m"]; pd=float(np.linalg.norm(plain_end[fruit]-target)*1000); ad=float(np.linalg.norm(afro_end[fruit]-target)*1000)
            distances={"Plain PI0.5 25K":pd,"AFRO 50K":ad}
            if film: distances[args.film_label]=float(np.linalg.norm(film_end[fruit]-target)*1000)
            ordered=sorted(distances.items(),key=lambda item:item[1])
            results.append({"episode":ep,"frame":frame,"active_arm":arm,"prompt_fruit":fruit,"fruit_position_xyz_m":target.tolist(),"fruit_contact_frame":target_info["contact_frame"],"plain_pi05_25k_distance_mm":pd,"afro_50k_distance_mm":ad,"film_all18_60k_distance_mm":distances.get(args.film_label),"winner":ordered[0][0],"winner_margin_mm":ordered[1][1]-ordered[0][1]})
    summary={"pairs":len(results),"afro_wins":sum(r["winner"]=="AFRO 50K" for r in results),"plain_wins":sum(r["winner"]=="Plain PI0.5 25K" for r in results),"film_wins":sum(r["winner"]==args.film_label for r in results),"film_label":args.film_label,"plain_mean_mm":float(np.mean([r["plain_pi05_25k_distance_mm"] for r in results])),"afro_mean_mm":float(np.mean([r["afro_50k_distance_mm"] for r in results])),"film_mean_mm":float(np.mean([r["film_all18_60k_distance_mm"] for r in results])) if film else None}
    (args.output_dir/"remaining_fruit_endpoint_distance.json").write_text(json.dumps({"summary":summary,"rows":results},indent=2))
    labels=[f"ep{r['episode']} f{r['frame']}\n{r['prompt_fruit']}" for r in results]; x=np.arange(len(results)); fig,ax=plt.subplots(figsize=(18,7)); width=.26 if film else .4; ax.bar(x-width,[r["plain_pi05_25k_distance_mm"] for r in results],width,label="Plain PI0.5 25K",color="#2563eb"); ax.bar(x,[r["afro_50k_distance_mm"] for r in results],width,label="AFRO 50K",color="#f97316");
    if film: ax.bar(x+width,[r["film_all18_60k_distance_mm"] for r in results],width,label=args.film_label,color="#16a34a")
    ax.set_xticks(x,labels,rotation=55,ha="right"); ax.set_ylabel("50-step endpoint distance to named fruit (mm)"); ax.set_title("Same observation, fruit-specific SUBtask; no GT trajectory"); ax.legend(); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(args.output_dir/"remaining_fruit_endpoint_distance.png",dpi=180)
    print(json.dumps({"summary":summary,"rows":[{k:r[k] for k in ("episode","frame","prompt_fruit","plain_pi05_25k_distance_mm","afro_50k_distance_mm","film_all18_60k_distance_mm","winner","winner_margin_mm")} for r in results]},indent=2))


if __name__ == "__main__": main()
