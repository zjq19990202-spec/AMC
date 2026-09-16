#!/usr/bin/env python3
"""Keep original CR1 LeRobot shape and inject prompt-aligned atom labels.

``native`` retains the three original cameras and full 16-D bimanual state/action.
``mirror`` makes a physically consistent left/right augmentation:
  base -> hflip(base), left wrist -> hflip(right wrist), right wrist -> hflip(left wrist)
  [L7,Lg,R7,Rg] -> [S*R7,Rg,S*L7,Lg]
where S is the verified CR1 left/right joint-sign map.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from atomic_latent_vla.data.atomic_ratios import local_atomic_weight_rows

N_ATOMS = 12
ARM_SIGN_8 = np.asarray([1, -1, -1, 1, -1, 1, -1, 1], dtype=np.float32)


def episode_id(path: Path) -> int:
    return int(path.stem.split("_")[1])


def annotations(directory: Path) -> dict[int, dict]:
    result = {episode_id(p): json.loads(p.read_text()) for p in sorted(directory.glob("episode_*.json"))}
    if not result:
        raise FileNotFoundError(f"No annotations in {directory}")
    return result


def targets(segment: dict) -> list[tuple[int, float]]:
    if segment.get("atomic_targets"):
        return [
            (
                int(item["label"]),
                float(item.get("confidence", item.get("weight", 1.0))),
            )
            for item in segment["atomic_targets"]
        ]
    return [(int(x["label"]), float(segment.get("gate_weights", [1.0] * 12)[i]))
            for i, x in enumerate(segment.get("gate_labels", []))]


def annotate_times(times: np.ndarray, annotation: dict):
    n = len(times); sid = np.full(n, -1, np.int64); typ = np.zeros(n, np.int8)
    prompt = np.asarray(["No stable atomic motion is annotated for this frame."] * n, dtype=object)
    atoms = np.zeros((n, N_ATOMS), np.float32)
    for order, seg in enumerate(annotation.get("segments", [])):
        begin, end = float(seg["start_s"]), float(seg["end_s"])
        labels = targets(seg)
        valid = bool(seg.get("training_eligible", bool(labels))) and bool(labels)
        if "gate_mode" in seg:
            valid = seg["gate_mode"] in {"single", "dual"} and bool(labels)
        hit = (times >= begin - 1e-6) & (times < end - 1e-6)
        sid[hit] = int(seg.get("segment_id", order)); typ[hit] = int(valid)
        prompt[hit] = str(seg.get("low_level_instruction") or prompt[0])
        if valid:
            fallback = np.zeros(N_ATOMS, dtype=np.float32)
            for label, weight in labels:
                if not 0 <= label < N_ATOMS: raise ValueError(f"bad atom {label}")
                fallback[label] = weight
            atoms[hit] = local_atomic_weight_rows(times[hit], seg, fallback)
    return sid, typ, prompt, atoms


def fixed(values: np.ndarray, width: int) -> pa.Array:
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1), type=pa.float32()), width)


def mirror_bimanual(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape[-1] != 16: raise ValueError(f"expected full [..,16], got {values.shape}")
    out = np.empty_like(values)
    out[..., :8] = values[..., 8:] * ARM_SIGN_8
    out[..., 8:] = values[..., :8] * ARM_SIGN_8
    return out


def flip_videos(source: Path, output: Path) -> None:
    encoders = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"], stderr=subprocess.DEVNULL, text=True)
    codec = "h264_nvenc" if "h264_nvenc" in encoders else "libx264"
    pairs = [("observation.images.base_0_rgb", "observation.images.base_0_rgb"),
             ("observation.images.left_wrist_0_rgb", "observation.images.right_wrist_0_rgb"),
             ("observation.images.right_wrist_0_rgb", "observation.images.left_wrist_0_rgb")]
    for source_key, destination_key in pairs:
        root = source / "videos" / source_key
        for video in sorted(root.rglob("*.mp4")):
            dst = output / "videos" / destination_key / video.relative_to(root)
            dst.parent.mkdir(parents=True, exist_ok=True)
            cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(video), "-vf", "hflip", "-an", "-c:v", codec]
            cmd += (["-preset", "p4", "-cq", "23"] if codec == "h264_nvenc" else ["-preset", "medium", "-crf", "20"])
            subprocess.run(cmd + ["-pix_fmt", "yuv420p", str(dst)], check=True)


def rewrite_metadata(source: Path, output: Path, task_by_episode: dict[int, str], mirror: bool, total: int) -> None:
    info = json.loads((output / "meta" / "info.json").read_text())
    info["total_episodes"] = len(task_by_episode); info["total_frames"] = total; info["total_tasks"] = len(set(task_by_episode.values()))
    for name, width, dtype in (("annotation.segment_id", 1, "int64"), ("annotation.type", 1, "int8"), ("annotation.prompt", 1, "string"), ("annotation.atoms", 12, "float32")):
        info["features"][name] = {"dtype": dtype, "shape": [width], "names": None if width == 1 else [f"atom_{i}" for i in range(12)]}
    (output / "meta" / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")
    mapping: OrderedDict[str, int] = OrderedDict()
    for task in task_by_episode.values(): mapping.setdefault(task, len(mapping))
    pq.write_table(pa.table({"task_index": pa.array(list(mapping.values()), type=pa.int64()), "task": pa.array(list(mapping), type=pa.string())}), output / "meta" / "tasks.parquet", compression="zstd")
    ep_file = next((source / "meta" / "episodes").rglob("*.parquet")); rows = pq.ParquetFile(ep_file).read().to_pylist(); cleaned=[]
    for row in rows:
        ep=int(row["episode_index"])
        if ep not in task_by_episode: continue
        if mirror:
            for tail in ("chunk_index", "file_index", "from_timestamp", "to_timestamp"):
                a=f"videos/observation.images.left_wrist_0_rgb/{tail}"; b=f"videos/observation.images.right_wrist_0_rgb/{tail}"; row[a],row[b]=row[b],row[a]
        row={k:v for k,v in row.items() if not k.startswith("stats/")}; row["tasks"]=[task_by_episode[ep]]; cleaned.append(row)
    episode_target=output/"meta"/"episodes"/"chunk-000"/"file-000.parquet"; episode_target.parent.mkdir(parents=True,exist_ok=True); pq.write_table(pa.Table.from_pylist(cleaned),episode_target,compression="zstd")
    (output/"meta"/"stats.json").unlink(missing_ok=True)


def build(source: Path, annotation_dir: Path, output: Path, mode: str) -> None:
    if output.exists(): raise FileExistsError(output)
    ann=annotations(annotation_dir); output.mkdir(parents=True)
    if mode == "native": shutil.copytree(source/"videos", output/"videos")
    else: flip_videos(source, output)
    shutil.copytree(source/"meta", output/"meta"); (output/"data"/"chunk-000").mkdir(parents=True)
    data_file=next((source/"data").rglob("*.parquet")); data=pq.ParquetFile(data_file).read().to_pydict()
    eps=np.asarray(data["episode_index"],np.int64); keep=np.isin(eps,np.asarray(list(ann),np.int64)); times=np.asarray(data["timestamp"],np.float64)
    sid=np.full(len(eps),-1,np.int64); typ=np.zeros(len(eps),np.int8); prompt=np.asarray(["No stable atomic motion is annotated for this frame."]*len(eps),object); atoms=np.zeros((len(eps),12),np.float32)
    for ep,a in ann.items():
        mask=eps==ep; x=annotate_times(times[mask],a); sid[mask],typ[mask],prompt[mask],atoms[mask]=x
    task_by_ep={ep:str(a.get("task") or a.get("global_description") or "") for ep,a in ann.items()}; task_map=OrderedDict(); ep_task={ep:task_map.setdefault(t,len(task_map)) for ep,t in task_by_ep.items()}
    cols={}
    for key,value in data.items():
        if key in {"observation.state","action","task_index"}: continue
        cols[key]=pa.array(np.asarray(value)[keep])
    state=np.asarray(data["observation.state"],np.float32); action=np.asarray(data["action"],np.float32)
    if mode == "mirror": state,action=mirror_bimanual(state),mirror_bimanual(action)
    cols["observation.state"]=fixed(state[keep],16); cols["action"]=fixed(action[keep],16); cols["task_index"]=pa.array(np.asarray([ep_task[int(e)] for e in eps[keep]],np.int64))
    cols["annotation.segment_id"]=pa.array(sid[keep]); cols["annotation.type"]=pa.array(typ[keep]); cols["annotation.prompt"]=pa.array(prompt[keep],type=pa.string()); cols["annotation.atoms"]=fixed(atoms[keep],12)
    pq.write_table(pa.table(cols),output/"data"/"chunk-000"/"file-000.parquet",compression="zstd")
    rewrite_metadata(source,output,task_by_ep,mode=="mirror",int(keep.sum()))
    summary={"source":str(source),"annotations":str(annotation_dir),"mode":mode,"episodes":len(ann),"frames":int(keep.sum()),"atomic_frames":int(typ[keep].sum()),"cameras":3,"state_action_dim":16}
    (output/"conversion_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n"); print(json.dumps(summary,ensure_ascii=False))


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--source",type=Path,required=True); p.add_argument("--annotations",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--mode",choices=("native","mirror"),required=True); a=p.parse_args(); build(a.source,a.annotations,a.output,a.mode)
