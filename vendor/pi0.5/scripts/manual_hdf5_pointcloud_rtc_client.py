#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import h5py
import numpy as np

from openpi_client.websocket_client_policy import WebsocketClientPolicy

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap
except ModuleNotFoundError:
    plt = None


DEFAULT_IMAGE_KEYS = (
    "/observations/images/cam_high",
    "/observations/images/cam_left_wrist",
    "/observations/images/cam_right_wrist",
)
CUTOFFS = ("t0.0", "t0.3", "t0.7")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=12000)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--episode-index", type=int, default=0)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--num-iters", type=int, default=30)
    p.add_argument("--inference-delay", type=int, default=7)
    p.add_argument("--execution-horizon", type=int, default=20)
    p.add_argument("--warmup-infers", type=int, default=1)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--prompt", default="perform the cabinet manipulation task")
    p.add_argument("--state-key", default="/observations/qpos")
    p.add_argument("--actions-key", default="/actions")
    p.add_argument("--depth-key", default="/observations/depths/cam_d435")
    p.add_argument("--image-keys", nargs=3, default=DEFAULT_IMAGE_KEYS)
    p.add_argument("--fallback-image-key", default="/observations/images/cam_d435")
    p.add_argument("--global-sample-offset", type=int, default=0)
    p.add_argument("--noise-plot", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def find_episode_file(root: Path, episode_index: int) -> Path:
    candidates = [
        root / f"episode_{episode_index}.hdf5",
        root / f"episode_{episode_index:06d}.hdf5",
        root / f"episode_{episode_index}.h5",
        root / f"episode_{episode_index:06d}.h5",
    ]
    for path in candidates:
        if path.exists():
            return path
    patterns = [
        f"**/episode_{episode_index}.hdf5",
        f"**/episode_{episode_index}_*.hdf5",
        f"**/episode_{episode_index:06d}.hdf5",
        f"**/episode_{episode_index:06d}_*.hdf5",
        f"**/episode_{episode_index}.h5",
        f"**/episode_{episode_index}_*.h5",
        f"**/episode_{episode_index:06d}.h5",
        f"**/episode_{episode_index:06d}_*.h5",
    ]
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find episode {episode_index} under {root}")


def dataset_exists(h5: h5py.File, key: str) -> bool:
    return key in h5


def decode_image_row(h5: h5py.File, key: str, frame: int) -> np.ndarray:
    ds = h5[key]
    row = ds[frame]
    if row.ndim == 3 and row.shape[-1] == 3:
        return np.asarray(row, dtype=np.uint8)

    len_key = f"{key}_len"
    encoded_len = int(h5[len_key][frame]) if len_key in h5 else int(np.asarray(row).size)
    image = cv2.imdecode(np.asarray(row[:encoded_len], dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to decode image key={key} frame={frame}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_policy_images(h5: h5py.File, frame: int, image_keys: tuple[str, str, str], fallback_key: str) -> dict[str, np.ndarray]:
    resolved = []
    for key in image_keys:
        resolved.append(key if dataset_exists(h5, key) else fallback_key)
    if not dataset_exists(h5, fallback_key) and any(not dataset_exists(h5, key) for key in image_keys):
        missing = [key for key in image_keys if not dataset_exists(h5, key)]
        raise KeyError(f"Missing image keys {missing} and fallback {fallback_key} is not present.")
    base, left, right = [decode_image_row(h5, key, frame) for key in resolved]
    return {
        "cam_high": np.transpose(base, (2, 0, 1)),
        "cam_left_wrist": np.transpose(left, (2, 0, 1)),
        "cam_right_wrist": np.transpose(right, (2, 0, 1)),
    }


def extract_cutoff(result: dict, cutoff: str) -> np.ndarray | None:
    if cutoff == "t0.0":
        return np.asarray(result["actions"], dtype=np.float32)
    cutoff_actions = result.get("noise_cutoff_actions") or {}
    if cutoff in cutoff_actions and cutoff_actions[cutoff] is not None:
        return np.asarray(cutoff_actions[cutoff], dtype=np.float32)
    return None


def _iter_cmap(num_iters: int):
    cm = get_cmap("plasma")
    return [cm(0.15 + 0.75 * i / max(num_iters - 1, 1)) for i in range(num_iters)]


def _plot_dims(action_dim: int) -> list[int]:
    return list(range(action_dim))


def plot_action_trunk(output_dir: Path, iters: list[dict], execution_horizon: int) -> None:
    if plt is None or not iters:
        return
    chunk_len = iters[0]["t0.0"].shape[0]
    action_dim = iters[0]["t0.0"].shape[1]
    dims = _plot_dims(action_dim)
    colors = _iter_cmap(len(iters))
    total_x = (len(iters) - 1) * execution_horizon + chunk_len

    fig, axes = plt.subplots(len(dims), 1, figsize=(14, 2.4 * len(dims)), sharex=True)
    if len(dims) == 1:
        axes = [axes]
    for ax, dim in zip(axes, dims, strict=False):
        first_dim = dim == dims[0]
        for k in range(len(iters)):
            ax.axvline(k * execution_horizon, color="dimgray", linewidth=0.8, linestyle="--", alpha=0.35)
        for k, it in enumerate(iters):
            x = k * execution_horizon + np.arange(chunk_len)
            a00 = it["t0.0"]
            a07 = it.get("t0.7")
            if a07 is not None:
                ax.fill_between(x, np.minimum(a00[:, dim], a07[:, dim]), np.maximum(a00[:, dim], a07[:, dim]),
                                color=colors[k], alpha=0.12)
                ax.plot(x, a07[:, dim], color=colors[k], linewidth=1.0, alpha=0.30, linestyle=":")
            a03 = it.get("t0.3")
            if a03 is not None:
                ax.plot(x, a03[:, dim], color=colors[k], linewidth=1.2, alpha=0.55, linestyle="--")
            ax.plot(x, a00[:, dim], color=colors[k], linewidth=2.0, alpha=0.95,
                    label=f"iter {k + 1}" if first_dim else None)
        ax.set_ylabel(f"dim {dim}")
        ax.set_xlim(0, total_x)
        ax.grid(alpha=0.22)
    axes[0].legend(fontsize=8, loc="upper right", ncol=max(1, min(len(iters), 6)))
    axes[0].set_title("Action chunks: solid=t0.0 dashed=t0.3 dotted=t0.7 band=spread")
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    out = output_dir / "action_trunk.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"saved {out}")


def plot_gt_branches(output_dir: Path, iters: list[dict], gt_actions: np.ndarray, gt_states: np.ndarray, execution_horizon: int) -> None:
    if plt is None or not iters or gt_actions.size == 0:
        return
    chunk_len = iters[0]["t0.0"].shape[0]
    action_dim = iters[0]["t0.0"].shape[1]
    dims = _plot_dims(action_dim)
    colors = _iter_cmap(len(iters))
    total_x = (len(iters) - 1) * execution_horizon + chunk_len
    gt = gt_actions if gt_actions.shape[1] == action_dim else gt_states[:, :action_dim]

    fig, axes = plt.subplots(len(dims), 1, figsize=(14, 2.4 * len(dims)), sharex=True)
    if len(dims) == 1:
        axes = [axes]
    for ax, dim in zip(axes, dims, strict=False):
        first_dim = dim == dims[0]
        ax.plot(np.arange(len(gt)), gt[:, dim], color="black", linewidth=2.4, alpha=0.9,
                label="GT action" if first_dim and gt is gt_actions else ("GT state" if first_dim else None))
        for k in range(len(iters)):
            ax.axvline(k * execution_horizon, color="dimgray", linewidth=0.8, linestyle="--", alpha=0.35)
        for k, it in enumerate(iters):
            x = k * execution_horizon + np.arange(chunk_len)
            for cutoff, lw, alpha, ls in (("t0.0", 2.0, 0.95, "-"), ("t0.3", 1.2, 0.55, "--"), ("t0.7", 1.0, 0.30, ":")):
                arr = it.get(cutoff)
                if arr is None:
                    continue
                ax.plot(x, arr[:, dim], color=colors[k], linewidth=lw, alpha=alpha, linestyle=ls,
                        label=f"iter {k + 1}" if first_dim and cutoff == "t0.0" else None)
        ax.set_ylabel(f"dim {dim}")
        ax.set_xlim(0, total_x)
        ax.grid(alpha=0.22)
    axes[0].legend(fontsize=8, loc="upper right", ncol=max(1, min(len(iters) + 1, 6)))
    axes[0].set_title("GT trunk plus predicted branches")
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    out = output_dir / "gt_branches.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"saved {out}")


def build_request(
    h5: h5py.File,
    frame: int,
    args: argparse.Namespace,
    *,
    clear_prefix: bool,
) -> dict:
    state = np.asarray(h5[args.state_key][frame], dtype=np.float32)
    depth_raw = np.asarray(h5[args.depth_key][frame])
    payload = {
        "state": state,
        "images": read_policy_images(h5, frame, tuple(args.image_keys), args.fallback_image_key),
        "prompt": args.prompt,
        "depth": {"depth_raw": depth_raw},
        "depth_sample_index": int(args.global_sample_offset + frame),
        "frame_index": int(frame),
        "inference_delay": int(args.inference_delay),
        "execution_horizon": int(args.execution_horizon),
        "noise_plot": bool(args.noise_plot),
    }
    if clear_prefix:
        payload["clear_prefix"] = True
    return payload


def main() -> None:
    args = parse_args()
    episode_path = find_episode_file(args.dataset_root, args.episode_index)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy = WebsocketClientPolicy(host=args.host, port=args.port)
    print(f"server: {policy.get_server_metadata()}")
    print(f"episode_path={episode_path}")

    records = []
    iters = []
    infer_states = []
    with h5py.File(episode_path, "r") as h5:
        length = len(h5[args.state_key])
        if args.start_frame >= length:
            raise ValueError(f"start-frame {args.start_frame} >= episode length {length}")

        for w in range(args.warmup_infers):
            req = build_request(h5, args.start_frame, args, clear_prefix=w == args.warmup_infers - 1)
            t0 = time.perf_counter()
            _ = policy.infer(req)
            print(f"warmup {w + 1}/{args.warmup_infers}: {(time.perf_counter() - t0) * 1000:.1f} ms")

        for i in range(args.num_iters):
            frame = min(args.start_frame + i * args.execution_horizon, length - 1)
            req = build_request(h5, frame, args, clear_prefix=False)
            t0 = time.perf_counter()
            result = policy.infer(req)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            actions = np.asarray(result["actions"], dtype=np.float32)
            timing = result.get("server_timing", {})
            counts = result.get("depth_counts", {})
            entry = {"client_ms": elapsed_ms, "server_timing": timing, "depth_counts": counts}
            for cutoff in CUTOFFS:
                entry[cutoff] = extract_cutoff(result, cutoff)
            iters.append(entry)
            infer_states.append(np.asarray(h5[args.state_key][frame], dtype=np.float32))
            print(
                f"iter {i + 1:02d}/{args.num_iters} frame={frame} "
                f"client_ms={elapsed_ms:.1f} action_shape={actions.shape} "
                f"server={timing} counts={counts}"
            )
            records.append(
                {
                    "iter": i,
                    "frame": int(frame),
                    "client_ms": float(elapsed_ms),
                    "server_timing": timing,
                    "depth_counts": counts,
                }
            )
            np.save(args.output_dir / f"actions_iter_{i:03d}.npy", actions)

            if i == 0:
                with (args.output_dir / "response_keys.json").open("w") as f:
                    json.dump(sorted(result.keys()), f, indent=2)

    with (args.output_dir / "timing.json").open("w") as f:
        json.dump(records, f, indent=2)
    if iters:
        chunk_len = iters[0]["t0.0"].shape[0]
        gt_start = args.start_frame
        gt_end = min(length, gt_start + (args.num_iters - 1) * args.execution_horizon + chunk_len + 1)
        with h5py.File(episode_path, "r") as h5:
            gt_actions = np.asarray(h5[args.actions_key][gt_start:gt_end], dtype=np.float32)
            gt_states = np.asarray(h5[args.state_key][gt_start:gt_end], dtype=np.float32)
        np.savez_compressed(
            args.output_dir / "refinement_data.npz",
            gt_actions=gt_actions,
            gt_states=gt_states,
            infer_states=np.stack(infer_states),
            latencies_ms=np.array([it["client_ms"] for it in iters], dtype=np.float32),
            **{
                f"iters_{cutoff}": np.stack([it[cutoff] for it in iters if it[cutoff] is not None])
                for cutoff in CUTOFFS
                if any(it[cutoff] is not None for it in iters)
            },
        )
        print(f"saved {args.output_dir / 'refinement_data.npz'}")
        plot_action_trunk(args.output_dir, iters, args.execution_horizon)
        plot_gt_branches(args.output_dir, iters, gt_actions, gt_states, args.execution_horizon)
    print(f"output -> {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
