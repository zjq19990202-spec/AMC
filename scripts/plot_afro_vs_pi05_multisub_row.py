#!/usr/bin/env python3
"""One-row qualitative comparison of PI0.5 and AFRO across multiple SUBtasks."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from atomic_latent_vla.pi05.training_data import atomic_collate, batch_to_observation, build_atomic_dataset
from evaluate_fruit_target_switch import _find_row
from evaluate_global_episode_chunks import _endpoint_tcp_from_actions

PANELS=(
 {"episode":100,"frame":890,"native":"yellow pear","targets":["yellow pear","carrot","banana"]},
 {"episode":114,"frame":490,"native":"yellow pear","targets":["yellow pear","green bitter melon"]},
 {"episode":124,"frame":770,"native":"green bitter melon","targets":["green bitter melon","green radish","yellow pear"]},
)
COLORS={
 "yellow pear":"#7B61FF",
 "carrot":"#D55E00",
 "banana":"#E6AB02",
 "green bitter melon":"#009E73",
 "green radish":"#0072B2",
}

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--dataset-root",type=Path,required=True);ap.add_argument("--plain-json",type=Path,required=True);ap.add_argument("--afro-npz",type=Path,required=True);ap.add_argument("--target-json",type=Path,required=True);ap.add_argument("--paired-json",type=Path,required=True);ap.add_argument("--output-dir",type=Path,required=True);args=ap.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
 plain=json.loads(args.plain_json.read_text());prows={(r["episode"],r["frame"]):r for r in plain["rows"]};afro=np.load(args.afro_npz,allow_pickle=True);targets={(r["episode"],r["frame"],r["fruit"]):np.asarray(r["target_xyz_m"]) for r in json.loads(args.target_json.read_text())["rows"]};paired={(r["episode"],r["frame"],r["fruit"]):r for r in json.loads(args.paired_json.read_text())["rows"]}
 dataset=build_atomic_dataset((args.dataset_root,),norm_assets_dir="/mnt/cunchu/zjq/target",norm_asset_id="openpi_norm_union2375_allframes_v1",action_horizon=50,max_token_len=192,include_fast=False,atomic_composition_sidecar="fk_horizon_3hz_gate_top5_stay_v2",pad_subtask_horizon=True)
 fig=plt.figure(figsize=(21,12.5),constrained_layout=True);grid=fig.add_gridspec(2,3,height_ratios=(.82,1.18));axes=[];report=[]
 for column,spec in enumerate(PANELS):
  image_ax=fig.add_subplot(grid[0,column]);ax=fig.add_subplot(grid[1,column]);axes.append(ax)
  ep,frame,native=spec["episode"],spec["frame"],spec["native"];pr=prows[(ep,frame)];prefix=f"episode_{ep:06d}_frame_{frame:06d}";names=list(afro[prefix+"_prompt_names"]);panel=[];pts=[]
  dataset_index,*_=_find_row(dataset,ep,frame);obs,_=batch_to_observation(atomic_collate([dataset[dataset_index]]));image=np.clip((np.asarray(obs.images["base_0_rgb"])[0]+1.0)*127.5,0,255).astype(np.uint8);image_ax.imshow(image);image_ax.axis("off");image_ax.set_title(f"episode {ep}, frame {frame}\noriginal: {pr['native_subtask']}",fontsize=11)
  for fruit in spec["targets"]:
   info=paired[(ep,frame,fruit)];assert info["plain_arm"]==info["afro_moving_arm"];arm=info["afro_moving_arm"];pi=next(i for i,x in enumerate(pr["prompts"]) if fruit in x.lower());ptraj=np.asarray(pr[f"{arm}_tcp_trajectories_m"])[pi];atraj=_endpoint_tcp_from_actions(None,0,afro[prefix+"_prediction_actions"][names.index(fruit)],arm);target=targets[(ep,frame,fruit)];color=COLORS[fruit];native_width=3.4 if fruit==native else 2.25
   ax.plot(-ptraj[:,1],ptraj[:,0],color=color,ls="--",lw=native_width,alpha=.70)
   ax.plot(-atraj[:,1],atraj[:,0],color=color,ls="-",lw=native_width)
   ax.scatter(-ptraj[-1,1],ptraj[-1,0],color=color,marker="x",s=55,alpha=.75);ax.scatter(-atraj[-1,1],atraj[-1,0],facecolor="white",edgecolor=color,marker="o",s=55,lw=2,zorder=7);ax.scatter(-target[1],target[0],color=color,marker="*",s=190,zorder=8)
   role="native" if fruit==native else "steer";ax.annotate(f"{role}: {fruit}\nPI {info['plain_final_mm']:.0f} / AFRO {info['final_mm']:.0f} mm",(-target[1],target[0]),xytext=(6,6),textcoords="offset points",fontsize=8.5,color=color,fontweight="bold");pts.extend([*np.column_stack([-ptraj[:,1],ptraj[:,0]]),*np.column_stack([-atraj[:,1],atraj[:,0]]),[-target[1],target[0]]]);panel.append({"fruit":fruit,"role":role,"arm":arm,"pi05_final_mm":info["plain_final_mm"],"afro_final_mm":info["final_mm"],"afro_advantage_mm":info["afro_advantage_final_mm"]})
  p=np.asarray(pts);lo,hi=p.min(0),p.max(0);pad=np.maximum((hi-lo)*.14,.025);ax.set_xlim(lo[0]-pad[0],hi[0]+pad[0]);ax.set_ylim(lo[1]-pad[1],hi[1]+pad[1]);ax.set_aspect("equal",adjustable="box");ax.grid(alpha=.22);ax.set_xlabel("−base y (m)  → image right");ax.set_ylabel("base x (m)  → image up");ax.set_title("TCP200 projected to base-camera-like top view",fontsize=12)
  ax.legend(handles=[Line2D([0],[0],color="#374151",ls="--",lw=2.5,label="PI0.5 25K"),Line2D([0],[0],color="#374151",ls="-",lw=2.5,label="AFRO 50K + correct ZM"),Line2D([0],[0],color="#374151",marker="*",ls="",markersize=11,label="fruit position"),Line2D([0],[0],color="#374151",marker="x",ls="",label="PI0.5 endpoint"),Line2D([0],[0],marker="o",markerfacecolor="white",markeredgecolor="#374151",ls="",label="AFRO endpoint")],fontsize=8,loc="best");report.append({**spec,"results":panel})
 fig.suptitle("Same observation and SUBtask: AFRO trajectories finish closer to the named fruit",fontsize=16);fig.savefig(args.output_dir/"afro_vs_pi05_three_frames_multisub_row.png",dpi=220,bbox_inches="tight");(args.output_dir/"afro_vs_pi05_three_frames_multisub.json").write_text(json.dumps(report,indent=2))
if __name__=="__main__":main()
