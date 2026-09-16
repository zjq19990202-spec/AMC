#!/usr/bin/env python3
"""Paired B2 action ablation for zM, slow zF, and fast force/state routes."""
from __future__ import annotations

import argparse, json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from atomic_latent_vla.pi05.force_training_data import (
    ForceNormalization, batch_to_force_inputs, build_force_dataset, force_collate,
)
from atomic_latent_vla.pi05.weights import AtomicPi05CheckpointLoader
from evaluate_force_b2_episode50_ablation import _candidate_indices, _choose_episode, _config
from evaluate_force_rtc_offsets import _force_metadata_at_offset, _prepare_force_context, _sample_force_update

OFFSETS = (0, 10, 20, 30, 40)
METHODS = ("full", "no_zm", "no_zf", "no_fast")

def rms(x):
    x=np.asarray(x,dtype=np.float64); return float(np.sqrt(np.mean(x*x)))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--dataset-root",type=Path,required=True)
    p.add_argument("--norm-assets-dir",type=Path,required=True); p.add_argument("--norm-asset-id",required=True)
    p.add_argument("--force-norm",type=Path,required=True); p.add_argument("--episode",default="auto")
    p.add_argument("--batch-size",type=int,default=4); p.add_argument("--max-windows",type=int,default=8)
    p.add_argument("--seed",type=int,default=20260831); p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); a.output.parent.mkdir(parents=True,exist_ok=True)
    cfg=_config()
    ds=build_force_dataset((a.dataset_root,),norm_assets_dir=a.norm_assets_dir,norm_asset_id=a.norm_asset_id,
        force_norm_path=a.force_norm,max_token_len=cfg.max_token_len,force_update_offsets=OFFSETS,
        load_future_force_targets=False,seed=0)
    cand=_candidate_indices(ds); ep=_choose_episode(cand,a.episode); raw=ds._raw
    sel=sorted(cand[ep],key=lambda i:int(raw.base._frame_index[int(raw.anchors[i])]))[:a.max_windows]
    init=cfg.create(jax.random.key(0)); _,state=nnx.split(init)
    params=AtomicPi05CheckpointLoader(str(a.checkpoint/"params")).load(state.to_pure_dict())
    model=cfg.load(params,remove_extra_params=False); model.eval(); norm=ForceNormalization.load(a.force_norm)
    errors={m:{o:[] for o in OFFSETS} for m in METHODS}; changes={m:{o:[] for o in OFFSETS} for m in METHODS[1:]}
    dz={m:{o:[] for o in OFFSETS} for m in METHODS}
    for start in range(0,len(sel),a.batch_size):
        ids=sel[start:start+a.batch_size]; n=len(ids); ids += [ids[-1]]*(a.batch_size-n)
        batch=force_collate([ds[i] for i in ids]); obs_np,gt_np,initial=batch_to_force_inputs(batch)
        obs=jax.tree.map(jnp.asarray,obs_np); gt=np.asarray(gt_np)[...,:16]
        ctx=_prepare_force_context(model,obs,jnp.asarray(initial["slow_force_history"]),
            jnp.asarray(initial["slow_state_history"]),jnp.asarray(initial["slow_history_mask"]))
        contexts={"full":ctx,"no_zm":ctx.replace(
                z_model=jnp.zeros_like(ctx.z_model),
                layerwise_arm_latents=jnp.zeros_like(ctx.layerwise_arm_latents),
            ),
            "no_zf":ctx.replace(force_latent=jnp.zeros_like(ctx.force_latent)),"no_fast":ctx}
        noise=jax.random.normal(jax.random.fold_in(jax.random.key(a.seed),start),(a.batch_size,cfg.action_horizon,cfg.action_dim))
        for off in OFFSETS:
            f=_force_metadata_at_offset(raw,norm,ids,off); mask=jnp.asarray(f["current_history_mask"])
            preds={}
            for m in METHODS:
                use_mask=jnp.zeros_like(mask) if m=="no_fast" else mask
                pred,mod=_sample_force_update(model,contexts[m],jnp.asarray(f["current_force_history"]),
                    jnp.asarray(f["current_state_history"]),use_mask,jnp.asarray(f["update_offset"]),noise,jnp.asarray(gt))
                pred=np.asarray(jax.device_get(pred))[:n]; preds[m]=pred
                d=np.asarray(jax.device_get(mod.delta_z))[:n]; dz[m][off].extend(np.linalg.norm(d,axis=-1).mean(-1).tolist())
                errors[m][off].extend(np.sqrt(np.mean((pred[:,off:]-gt[:n,off:])**2,axis=(1,2))).tolist())
            for m in METHODS[1:]:
                changes[m][off].extend(np.sqrt(np.mean((preds[m][:,off:]-preds["full"][:,off:])**2,axis=(1,2))).tolist())
    summary={"episode":ep,"windows":len(sel),"checkpoint":str(a.checkpoint),"dataset":str(a.dataset_root),"offsets":{}}
    for off in OFFSETS:
        row={m:{"action_rmse":rms(errors[m][off]),"delta_z_norm":float(np.mean(dz[m][off]))} for m in METHODS}
        for m in METHODS[1:]:
            row[m]["prediction_change_rmse"]=rms(changes[m][off]); row[m]["gt_rmse_change_pct"]=100*(row[m]["action_rmse"]/row["full"]["action_rmse"]-1)
        total=sum(row[m]["prediction_change_rmse"] for m in METHODS[1:])
        for m in METHODS[1:]: row[m]["change_share_pct"]=100*row[m]["prediction_change_rmse"]/max(total,1e-12)
        summary["offsets"][str(off)]=row
    a.output.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n"); print(json.dumps(summary,indent=2,ensure_ascii=False))
if __name__=="__main__": main()
