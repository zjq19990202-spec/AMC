#!/usr/bin/env python3
"""Clear bimanual XY/XZ plots for selected AFRO fruit-steering cases."""

from __future__ import annotations

import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from atomic_latent_vla.pi05.training_data import atomic_collate, batch_to_observation, build_atomic_dataset
from evaluate_fruit_target_switch import _find_row
from evaluate_global_episode_chunks import _endpoint_tcp_from_actions

CASES = ((124, 770, "green bitter melon"), (142, 0, "orange"), (107, 570, "yellow pear"))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset-root",type=Path,required=True); ap.add_argument("--plain-json",type=Path,required=True); ap.add_argument("--afro-npz",type=Path,required=True); ap.add_argument("--target-json",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True); args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    plain=json.loads(args.plain_json.read_text()); prows={(r["episode"],r["frame"]):r for r in plain["rows"]}; afro=np.load(args.afro_npz,allow_pickle=True)
    target_rows=json.loads(args.target_json.read_text())["rows"]; scenes={}
    for r in target_rows: scenes.setdefault((r["episode"],r["frame"]),[]).append(r)
    table=pq.read_table(args.dataset_root/"data/chunk-000/file-000.parquet",columns=["episode_index","frame_index","index"]); eps=np.asarray(table["episode_index"]); frames=np.asarray(table["frame_index"]); indices=np.asarray(table["index"]); tcp=np.load(args.dataset_root/"meta/tcp_pose_bimanual_base_tcp200.npy",mmap_mode="r")
    dataset=build_atomic_dataset((args.dataset_root,),norm_assets_dir="/mnt/cunchu/zjq/target",norm_asset_id="openpi_norm_union2375_allframes_v1",action_horizon=50,max_token_len=192,include_fast=False,atomic_composition_sidecar="fk_horizon_3hz_gate_top5_stay_v2",pad_subtask_horizon=True)
    fig=plt.figure(figsize=(20,15),constrained_layout=True); grid=fig.add_gridspec(3,3,width_ratios=(1.0,1.15,1.15)); report=[]
    model_style={"PI0.5 25K":("#2563eb",prows),"AFRO 50K + correct ZM":("#ef4444",None)}
    for ri,(ep,frame,fruit) in enumerate(CASES):
        di,*_=_find_row(dataset,ep,frame); obs,_=batch_to_observation(atomic_collate([dataset[di]])); image=np.clip((np.asarray(obs.images["base_0_rgb"])[0]+1)*127.5,0,255).astype(np.uint8)
        axim=fig.add_subplot(grid[ri,0]);axim.imshow(image);axim.axis("off");axim.set_title(f"episode {ep}, frame {frame}\nMove toward the {fruit} and grasp it",fontsize=11)
        source_row=int(np.flatnonzero((eps==ep)&(frames==frame))[0]); starts={"left":np.asarray(tcp[int(indices[source_row]),0:3]),"right":np.asarray(tcp[int(indices[source_row]),24:27])}
        pr=prows[(ep,frame)];pi=next(i for i,x in enumerate(pr["prompts"]) if fruit in x.lower());prefix=f"episode_{ep:06d}_frame_{frame:06d}";names=list(afro[prefix+"_prompt_names"]);ai=names.index(fruit);actions=afro[prefix+"_prediction_actions"][ai]
        traj={}
        for arm in ("left","right"):
            traj[("PI0.5 25K",arm)]=np.concatenate([starts[arm][None],np.asarray(pr[f"{arm}_tcp_trajectories_m"])[pi]])
            traj[("AFRO 50K + correct ZM",arm)]=np.concatenate([starts[arm][None],_endpoint_tcp_from_actions(None,0,actions,arm)])
        requested=next(r for r in scenes[(ep,frame)] if r["fruit"]==fruit);target=np.asarray(requested["target_xyz_m"])
        axes=[fig.add_subplot(grid[ri,1]),fig.add_subplot(grid[ri,2])]
        for ax,(u,v,title) in zip(axes,((0,1,"XY top view"),(0,2,"XZ side view")),strict=True):
            for (model,arm),xyz in traj.items():
                color=model_style[model][0]; ls="-" if arm=="left" else "--"; ax.plot(xyz[:,u],xyz[:,v],color=color,ls=ls,lw=2.3,label=f"{model}, {arm}");ax.scatter(xyz[0,u],xyz[0,v],color=color,marker="o",s=28);ax.scatter(xyz[-1,u],xyz[-1,v],color=color,marker="x",s=55)
            for item in scenes[(ep,frame)]:
                xyz=np.asarray(item["target_xyz_m"]); active=item["fruit"]==fruit;color="#16a34a" if active else "#f59e0b";ax.scatter(xyz[u],xyz[v],color=color,marker="*" if active else "o",s=165 if active else 32,zorder=8);ax.annotate(item["fruit"]+(" (box)" if item["target_kind"]=="placed_box_center" else ""),(xyz[u],xyz[v]),xytext=(4,4),textcoords="offset points",fontsize=7,color=color)
            ax.set_xlabel("base x (m)");ax.set_ylabel(f"base {'y' if v==1 else 'z'} (m)");ax.set_title(title);ax.set_aspect("equal",adjustable="datalim");ax.grid(alpha=.25)
        axes[0].legend(fontsize=7,loc="best")
        metrics={}
        for model in model_style:
            distances=np.stack([np.linalg.norm(traj[(model,arm)]-target,axis=1)*1000 for arm in ("left","right")]);metrics[model]={"initial_nearest_mm":float(distances[:,0].min()),"final_nearest_mm":float(distances[:,-1].min()),"minimum_over_chunk_mm":float(distances.min()),"closest_arm":["left","right"][int(np.unravel_index(distances.argmin(),distances.shape)[0])],"closest_step":int(np.unravel_index(distances.argmin(),distances.shape)[1])}
        axes[1].text(.02,.98,f"nearest TCP distance\nPI0.5: {metrics['PI0.5 25K']['initial_nearest_mm']:.0f}→{metrics['PI0.5 25K']['final_nearest_mm']:.0f} mm\nAFRO: {metrics['AFRO 50K + correct ZM']['initial_nearest_mm']:.0f}→{metrics['AFRO 50K + correct ZM']['final_nearest_mm']:.0f} mm",transform=axes[1].transAxes,va="top",fontsize=9,bbox=dict(facecolor="white",alpha=.8,edgecolor="none"))
        report.append({"episode":ep,"frame":frame,"fruit":fruit,"metrics":metrics})
    fig.suptitle("AFRO qualitative steering advantages — both arms shown, TCP200",fontsize=16);fig.savefig(args.output_dir/"afro_advantage_three_cases_bimanual.png",dpi=200,bbox_inches="tight");(args.output_dir/"bimanual_cases.json").write_text(json.dumps(report,indent=2))

if __name__=="__main__":main()
