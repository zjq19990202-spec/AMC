#!/usr/bin/env python3
"""Three one-row AFRO panels: native SUBtask plus favorable steered SUBtasks."""

from __future__ import annotations

import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from evaluate_global_episode_chunks import _endpoint_tcp_from_actions

PANELS = (
    {"episode":96,"frame":120,"native":"carrot","targets":["carrot","banana","orange","yellow pear"]},
    {"episode":135,"frame":250,"native":"yellow pear","targets":["yellow pear","red chili pepper","banana","green radish"]},
    {"episode":124,"frame":770,"native":"green bitter melon","targets":["green bitter melon","yellow pear","red chili pepper","banana"]},
)
COLORS={"carrot":"#f97316","orange":"#ea580c","banana":"#eab308","green radish":"#22c55e","yellow pear":"#facc15","green bitter melon":"#15803d","red chili pepper":"#dc2626"}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--dataset-root",type=Path,required=True);ap.add_argument("--afro-npz",type=Path,required=True);ap.add_argument("--target-json",type=Path,required=True);ap.add_argument("--moving-json",type=Path,required=True);ap.add_argument("--output-dir",type=Path,required=True);args=ap.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    pred=np.load(args.afro_npz,allow_pickle=True);target_rows=json.loads(args.target_json.read_text())["rows"];targets={(r["episode"],r["frame"],r["fruit"]):np.asarray(r["target_xyz_m"]) for r in target_rows};moving={(r["episode"],r["frame"],r["fruit"]):r for r in json.loads(args.moving_json.read_text())["rows"]}
    t=pq.read_table(args.dataset_root/"data/chunk-000/file-000.parquet",columns=["episode_index","frame_index","index"]);eps=np.asarray(t["episode_index"]);frames=np.asarray(t["frame_index"]);indices=np.asarray(t["index"]);tcp=np.load(args.dataset_root/"meta/tcp_pose_bimanual_base_tcp200.npy",mmap_mode="r")
    fig,axes=plt.subplots(1,3,figsize=(22,7),constrained_layout=True);report=[]
    for ax,spec in zip(axes,PANELS,strict=True):
        ep,frame,native=spec["episode"],spec["frame"],spec["native"];prefix=f"episode_{ep:06d}_frame_{frame:06d}";names=list(pred[prefix+"_prompt_names"]);row=int(np.flatnonzero((eps==ep)&(frames==frame))[0]);case=[];allpts=[]
        for fruit in spec["targets"]:
            info=moving[(ep,frame,fruit)];arm=info["afro_moving_arm"];start=np.asarray(tcp[int(indices[row]),0:3] if arm=="left" else tcp[int(indices[row]),24:27]);actions=pred[prefix+"_prediction_actions"][names.index(fruit)];traj=np.concatenate([start[None],_endpoint_tcp_from_actions(None,0,actions,arm)]);target=targets[(ep,frame,fruit)];dist=np.linalg.norm(traj-target,axis=1)*1000;color=COLORS[fruit];is_native=fruit==native
            ax.plot(traj[:,0],traj[:,1],color=color,lw=3.4 if is_native else 2.1,ls="-" if is_native else "--",label=f"{'native' if is_native else 'steer'}: {fruit} ({arm}), {dist[0]:.0f}→{dist[-1]:.0f} mm")
            ax.scatter(traj[0,0],traj[0,1],color=color,marker="o",s=32);ax.scatter(traj[-1,0],traj[-1,1],color=color,marker="x",s=65);ax.scatter(target[0],target[1],color=color,marker="*",s=180,zorder=8);ax.annotate(fruit,(target[0],target[1]),xytext=(5,5),textcoords="offset points",fontsize=9,color=color,fontweight="bold")
            allpts.extend([*traj[:,0:2],target[0:2]]);case.append({"fruit":fruit,"role":"native" if is_native else "steer","arm":arm,"initial_mm":float(dist[0]),"final_mm":float(dist[-1]),"minimum_mm":float(dist.min()),"closest_step":int(dist.argmin())})
        pts=np.asarray(allpts);lo,hi=pts.min(0),pts.max(0);pad=np.maximum((hi-lo)*.12,.025);ax.set_xlim(lo[0]-pad[0],hi[0]+pad[0]);ax.set_ylim(lo[1]-pad[1],hi[1]+pad[1]);ax.set_aspect("equal",adjustable="box");ax.grid(alpha=.25);ax.set_xlabel("base x (m)");ax.set_ylabel("base y (m)");ax.set_title(f"episode {ep}, frame {frame}\noriginal SUBtask: grasp {native}",fontsize=12);ax.legend(fontsize=7.5,loc="best");report.append({**spec,"results":case})
    fig.suptitle("AFRO 50K + correct ZM: native and counterfactual fruit-SUBtask steering (TCP200, XY view)",fontsize=16);fig.savefig(args.output_dir/"afro_three_frames_native_plus_steers_row.png",dpi=220,bbox_inches="tight");(args.output_dir/"afro_three_frames_native_plus_steers.json").write_text(json.dumps(report,indent=2))
if __name__=="__main__":main()
