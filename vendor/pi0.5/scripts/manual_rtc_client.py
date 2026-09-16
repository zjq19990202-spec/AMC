"""Sequential denoising-refinement RTC client.

Runs --num-iters inferences on the same observation. Each inference's fully-denoised
output (t0.0) becomes the server's prefix for the next call. Every inference also
returns partially-denoised snapshots (t0.3, t0.7) via noise_plot.

Produces combined plots including:
  action_trunk.png            : returned policy action chunks overlaid by iteration
  gt_branches.png             : GT state trunk + predicted branches in absolute action space
  action_minus_obs.png        : predicted branches minus the input observation state
"""

import argparse
import json
import os
import time
from pathlib import Path

import av

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import numpy as np
import pandas as pd

from openpi_client.websocket_client_policy import WebsocketClientPolicy

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap
except ModuleNotFoundError:
    matplotlib = None
    plt = None

VIDEO_KEYS = (
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
    "observation.images.right_wrist_0_rgb",
)

CUTOFFS = ["t0.0", "t0.3", "t0.7"]


# ── args ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=12000)
    p.add_argument("--dataset-root", type=Path,
                   default=Path("/home/admin123/zjq/pickv3_legacy_ep0/plugbottle"))
    p.add_argument("--episode-index", type=int, default=0)
    p.add_argument("--start-frame", type=int, default=0,
                   help="Dataset row index used as the observation for all inferences.")
    p.add_argument("--num-iters", type=int, default=15,
                   help="Number of sequential denoising inferences (each output becomes next prefix).")
    p.add_argument("--inference-delay", type=int, default=7,
                   help="Steps until policy result arrives. Hard part = delay, soft part = chunk_len - delay.")
    p.add_argument("--execution-horizon", type=int, default=15,
                   help="Chunk length returned by the policy.")
    p.add_argument("--warmup-infers", type=int, default=2,
                   help="Warmup inferences (discarded) before measured iterations.")
    p.add_argument("--output-dir", type=Path, default=Path("rtc_manual_logs"))
    return p.parse_args()


# ── dataset helpers ────────────────────────────────────────────────────────────

def load_task_prompt(dataset_root: Path) -> str:
    with (dataset_root / "meta" / "tasks.jsonl").open() as f:
        return str(json.loads(next(f))["task"])

def load_episode_dataframe(dataset_root: Path, episode_index: int, chunks_size: int) -> pd.DataFrame:
    chunk = episode_index // chunks_size
    path = dataset_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
    return pd.read_parquet(path)

class EpisodeVideoReader:
    def __init__(self, path: Path) -> None:
        self.frames: list[np.ndarray] = []
        with av.open(str(path)) as container:
            for frame in container.decode(container.streams.video[0]):
                self.frames.append(frame.to_ndarray(format="rgb24"))
        if not self.frames:
            raise RuntimeError(f"No frames in {path}")

    def read_frame(self, idx: int) -> np.ndarray:
        return self.frames[idx]

    def close(self) -> None:
        self.frames.clear()

def load_video_readers(dataset_root: Path, episode_index: int, chunks_size: int) -> dict[str, EpisodeVideoReader]:
    chunk = episode_index // chunks_size
    readers = {}
    for key in VIDEO_KEYS:
        path = dataset_root / "videos" / f"chunk-{chunk:03d}" / key / f"episode_{episode_index:06d}.mp4"
        readers[key] = EpisodeVideoReader(path)
    return readers

def build_request(row: pd.Series, prompt: str, readers: dict, delay: int, horizon: int,
                  clear_prefix: bool) -> dict:
    fi = int(row["frame_index"])
    base  = readers["observation.images.base_0_rgb"].read_frame(fi)
    left  = readers["observation.images.left_wrist_0_rgb"].read_frame(fi)
    right = readers["observation.images.right_wrist_0_rgb"].read_frame(fi)
    payload = {
        "state": np.asarray(row["observation.state"], dtype=np.float32),
        "images": {
            "cam_high":        np.transpose(base,  (2, 0, 1)),
            "cam_left_wrist":  np.transpose(left,  (2, 0, 1)),
            "cam_right_wrist": np.transpose(right, (2, 0, 1)),
        },
        "prompt": prompt,
        "inference_delay": delay,
        "execution_horizon": horizon,
        "noise_plot": True,          # always request all cutoff variants
    }
    if clear_prefix:
        payload["clear_prefix"] = True
    return payload

def extract_cutoff(result: dict, cutoff: str) -> np.ndarray | None:
    """Pull the right action array out of a server response. Returns None if not available."""
    if cutoff == "t0.0":
        return np.asarray(result["actions"], dtype=np.float32)
    ca = result.get("noise_cutoff_actions") or {}
    if cutoff in ca and ca[cutoff] is not None:
        return np.asarray(ca[cutoff], dtype=np.float32)
    return None  # missing — do not fall back silently


# ── plot helpers ───────────────────────────────────────────────────────────────

def _plot_dims(action_dim: int) -> list[int]:
    return list(range(action_dim))

def _iter_cmap(num_iters: int):
    """Return a list of colors from light→dark for iteration 0…num_iters-1."""
    cm = get_cmap("plasma")
    return [cm(0.15 + 0.75 * i / max(num_iters - 1, 1)) for i in range(num_iters)]


def compute_overlap_stats(
    iters: list[dict],
    delay: int,
    infer_states: np.ndarray,
) -> dict:
    """Compare consecutive chunks on the overlapping absolute-time region."""
    pair_stats: list[dict] = []
    summary: dict[str, dict[str, float | int]] = {}

    for cutoff in CUTOFFS:
        diffs: list[np.ndarray] = []
        rel_diffs: list[np.ndarray] = []

        for k in range(len(iters) - 1):
            prev_arr = iters[k].get(cutoff)
            next_arr = iters[k + 1].get(cutoff)
            if prev_arr is None or next_arr is None:
                continue
            if delay >= len(prev_arr) or delay <= 0:
                continue

            overlap = min(len(prev_arr) - delay, len(next_arr))
            if overlap <= 0:
                continue

            prev_overlap = prev_arr[delay:delay + overlap]
            next_overlap = next_arr[:overlap]
            diff = next_overlap - prev_overlap
            diffs.append(diff)

            prev_rel = prev_overlap - infer_states[k]
            next_rel = next_overlap - infer_states[k + 1]
            rel_diff = next_rel - prev_rel
            rel_diffs.append(rel_diff)

            pair_stats.append(
                {
                    "cutoff": cutoff,
                    "iter_prev": k,
                    "iter_next": k + 1,
                    "overlap_steps": int(overlap),
                    "mae": float(np.mean(np.abs(diff))),
                    "rmse": float(np.sqrt(np.mean(diff ** 2))),
                    "max_abs": float(np.max(np.abs(diff))),
                    "relative_mae": float(np.mean(np.abs(rel_diff))),
                    "relative_rmse": float(np.sqrt(np.mean(rel_diff ** 2))),
                    "relative_max_abs": float(np.max(np.abs(rel_diff))),
                }
            )

        if diffs:
            all_diff = np.concatenate(diffs, axis=0)
            all_rel = np.concatenate(rel_diffs, axis=0)
            summary[cutoff] = {
                "pairs": len(diffs),
                "mae": float(np.mean(np.abs(all_diff))),
                "rmse": float(np.sqrt(np.mean(all_diff ** 2))),
                "max_abs": float(np.max(np.abs(all_diff))),
                "relative_mae": float(np.mean(np.abs(all_rel))),
                "relative_rmse": float(np.sqrt(np.mean(all_rel ** 2))),
                "relative_max_abs": float(np.max(np.abs(all_rel))),
            }

    return {"pair_stats": pair_stats, "summary": summary}


# ── Plot A: action trunk + branches ───────────────────────────────────────────

def _draw_iter_cutoffs(ax, dim, k, it, x0, color, first_dim):
    """Draw t0.0 (solid), t0.3 (semi), t0.7 (faint) for one iter on one ax.
    Fills the band between t0.0 and t0.7 to show the denoising spread.
    Returns True if any data was drawn."""
    a00 = it["t0.0"]
    a03 = it["t0.3"]
    a07 = it["t0.7"]
    if a00 is None:
        return False

    x = x0 + np.arange(len(a00))

    # filled band t0.7 → t0.0 (spread of denoising)
    if a07 is not None:
        lo = np.minimum(a00[:, dim], a07[:, dim])
        hi = np.maximum(a00[:, dim], a07[:, dim])
        ax.fill_between(x, lo, hi, color=color, alpha=0.12, zorder=1)

    # t0.7 — most noisy, most transparent
    if a07 is not None:
        ax.plot(x, a07[:, dim], color=color, linewidth=1.0, alpha=0.30, linestyle=":")

    # t0.3 — mid
    if a03 is not None:
        ax.plot(x, a03[:, dim], color=color, linewidth=1.2, alpha=0.55, linestyle="--")

    # t0.0 — fully denoised, solid
    label = f"iter {k+1}" if first_dim else None
    ax.plot(x, a00[:, dim], color=color, linewidth=2.0, alpha=0.95, linestyle="-", label=label)

    return True


def plot_action_trunk(
    output_dir: Path,
    iters: list[dict],
    delay: int,
    execution_horizon: int,
) -> None:
    """All cutoffs combined. iter k starts at x=k*execution_horizon.
    Same color per iter: t0.0 solid, t0.3 dashed, t0.7 dotted, band fill between t0.0 and t0.7."""
    if plt is None or not iters:
        return

    chunk_len = iters[0]["t0.0"].shape[0]
    action_dim = iters[0]["t0.0"].shape[1]
    dims = _plot_dims(action_dim)
    if not dims:
        return

    num_iters = len(iters)
    colors = _iter_cmap(num_iters)
    total_x = (num_iters - 1) * execution_horizon + chunk_len

    fig, axes = plt.subplots(len(dims), 1, figsize=(14, 2.4 * len(dims)), sharex=True)
    if len(dims) == 1:
        axes = [axes]

    for ax, dim in zip(axes, dims, strict=False):
        first_dim = (dim == dims[0])
        # soft overlap bands
        for k in range(num_iters - 1):
            xs = (k + 1) * execution_horizon
            xe = k * execution_horizon + chunk_len
            if xe > xs:
                ax.axvspan(xs, xe, color="lightyellow", alpha=0.5, zorder=0)
        # iter boundary lines
        for k in range(num_iters):
            ax.axvline(k * execution_horizon, color="dimgray", linewidth=0.8, linestyle="--", alpha=0.35)

        for k, it in enumerate(iters):
            _draw_iter_cutoffs(ax, dim, k, it, k * execution_horizon, colors[k], first_dim)

        ax.set_ylabel(f"dim {dim}")
        ax.set_xlim(0, total_x)
        ax.grid(alpha=0.22)

    # legend: iter colors + linestyle guide
    from matplotlib.lines import Line2D
    handles = axes[0].get_legend_handles_labels()[0]
    handles += [
        Line2D([0], [0], color="gray", lw=2.0, linestyle="-",  label="t0.0 (full denoise)"),
        Line2D([0], [0], color="gray", lw=1.2, linestyle="--", label="t0.3"),
        Line2D([0], [0], color="gray", lw=1.0, linestyle=":",  label="t0.7 (noisy)"),
    ]
    axes[0].legend(handles=handles, fontsize=8, loc="upper right", ncol=num_iters + 3)
    axes[0].set_title(
        f"Action chunks — all cutoffs overlaid  (iter k @ x=k×horizon={execution_horizon})\n"
        f"solid=t0.0  dashed=t0.3  dotted=t0.7  band=spread  yellow=soft overlap"
    )
    axes[-1].set_xlabel("step (relative to obs)")
    fig.tight_layout()
    out = output_dir / "action_trunk.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"saved {out}")


def plot_gt_branches(
    output_dir: Path,
    iters: list[dict],
    gt_actions: np.ndarray,
    gt_states: np.ndarray,
    delay: int,
    execution_horizon: int,
) -> None:
    """GT action trunk (black) + all cutoffs overlaid per iter.
    Policy outputs are already absolute, so predicted branches are plotted directly.
    iter k is anchored at x=k*execution_horizon."""
    if plt is None or not iters or gt_actions.size == 0:
        return

    chunk_len = iters[0]["t0.0"].shape[0]
    action_dim = iters[0]["t0.0"].shape[1]
    dims = _plot_dims(action_dim)
    if not dims:
        return

    num_iters = len(iters)
    colors = _iter_cmap(num_iters)
    total_x = (num_iters - 1) * execution_horizon + chunk_len
    gt_steps = np.arange(len(gt_actions))

    fig, axes = plt.subplots(len(dims), 1, figsize=(14, 2.4 * len(dims)), sharex=True)
    if len(dims) == 1:
        axes = [axes]

    for ax, dim in zip(axes, dims, strict=False):
        first_dim = (dim == dims[0])

        # GT trunk — use absolute states (gt_actions are deltas, gt_states are absolute)
        gt_state_steps = np.arange(len(gt_states))
        ax.plot(gt_state_steps, gt_states[:, dim], color="black", linewidth=2.4, alpha=0.9,
                label="GT" if first_dim else None, zorder=5)

        for k in range(num_iters):
            ax.axvline(k * execution_horizon, color="dimgray", linewidth=0.8, linestyle="--", alpha=0.35)

        for k, it in enumerate(iters):
            x0 = k * execution_horizon

            # Policy outputs are already absolute actions. Plot them directly.
            for cutoff, lw, alpha, ls in [("t0.0", 2.0, 0.95, "-"),
                                           ("t0.3", 1.2, 0.55, "--"),
                                           ("t0.7", 1.0, 0.30, ":")]:
                arr = it[cutoff]
                if arr is None:
                    continue
                bx = x0 + np.arange(len(arr))
                by = arr[:, dim]
                label = f"iter {k+1}" if (first_dim and cutoff == "t0.0") else None
                ax.plot(bx, by, color=colors[k], linewidth=lw, alpha=alpha, linestyle=ls, label=label)

            # fill band t0.7 → t0.0
            a00 = it["t0.0"]
            a07 = it["t0.7"]
            if a00 is not None and a07 is not None:
                bx = x0 + np.arange(len(a00))
                by00 = a00[:, dim]
                by07 = a07[:, dim]
                ax.fill_between(bx, np.minimum(by00, by07), np.maximum(by00, by07),
                                color=colors[k], alpha=0.10, zorder=1)

        ax.set_ylabel(f"dim {dim}")
        ax.set_xlim(0, total_x)
        ax.grid(alpha=0.22)

    from matplotlib.lines import Line2D
    handles = axes[0].get_legend_handles_labels()[0]
    handles += [
        Line2D([0], [0], color="gray", lw=2.0, linestyle="-",  label="t0.0"),
        Line2D([0], [0], color="gray", lw=1.2, linestyle="--", label="t0.3"),
        Line2D([0], [0], color="gray", lw=1.0, linestyle=":",  label="t0.7"),
    ]
    axes[0].legend(handles=handles, fontsize=8, loc="upper right", ncol=num_iters + 4)
    axes[0].set_title(
        f"GT trunk (black) + predicted branches — all cutoffs overlaid  (iter k @ x=k×horizon={execution_horizon})\n"
        f"solid=t0.0  dashed=t0.3  dotted=t0.7  band=spread"
    )
    axes[-1].set_xlabel("step (relative to obs)")
    fig.tight_layout()
    out = output_dir / "gt_branches.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"saved {out}")


    print(f"saved {out}")




# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    with (args.dataset_root / "meta" / "info.json").open() as f:
        chunks_size = int(json.load(f)["chunks_size"])

    prompt       = load_task_prompt(args.dataset_root)
    episode_df   = load_episode_dataframe(args.dataset_root, args.episode_index, chunks_size)
    readers      = load_video_readers(args.dataset_root, args.episode_index, chunks_size)

    if args.start_frame >= len(episode_df):
        raise ValueError(f"--start-frame {args.start_frame} >= episode length {len(episode_df)}")

    obs_row   = episode_df.iloc[args.start_frame]
    obs_frame = int(obs_row["frame_index"])
    d         = args.inference_delay
    h         = args.execution_horizon

    policy = WebsocketClientPolicy(host=args.host, port=args.port)
    print(f"server: {policy.get_server_metadata()}")
    print(
        f"episode={args.episode_index}  obs_frame={obs_frame}  "
        f"num_iters={args.num_iters}  delay={d}  horizon={h}  "
        f"rtc_warmup={args.warmup_infers}  hard_part={d}  soft_part={h-d}"
    )

    try:
        # ── warmup ────────────────────────────────────────────────────────────
        if args.warmup_infers > 0:
            req = build_request(obs_row, prompt, readers, d, h, clear_prefix=False)
            t0 = time.perf_counter()
            _ = policy.infer(req)
            ms = (time.perf_counter() - t0) * 1000.0
            print(f"  warmup init/1  {ms:.0f} ms  discarded")

        for w in range(args.warmup_infers):
            # Warm the RTC path and clear the prefix afterwards so the first measured
            # iteration starts from an empty prefix.
            req = build_request(obs_row, prompt, readers, d, h, clear_prefix=True)
            t0 = time.perf_counter()
            _ = policy.infer(req)
            ms = (time.perf_counter() - t0) * 1000.0
            print(f"  warmup rtc {w+1}/{args.warmup_infers}  {ms:.0f} ms  discarded")

        # ── sequential inferences ─────────────────────────────────────────────
        # iters[k] = {"t0.0": np.array, "t0.3": np.array, "t0.7": np.array, "ms": float}
        iters: list[dict] = []
        infer_states: list[np.ndarray] = []

        for k in range(args.num_iters):
            # never clear — warmup already seeded the prefix; all iters inherit and refine it
            iter_row = episode_df.iloc[min(args.start_frame + k * h, len(episode_df) - 1)]
            infer_states.append(np.asarray(iter_row["observation.state"], dtype=np.float32).copy())
            req = build_request(iter_row, prompt, readers, d, h, clear_prefix=False)
            t0 = time.perf_counter()
            result = policy.infer(req)
            ms = (time.perf_counter() - t0) * 1000.0

            # debug: show raw response keys on iter 1
            if k == 0:
                ca = result.get("noise_cutoff_actions") or {}
                print(f"  [debug iter1] response keys={list(result.keys())}  noise_cutoff_actions keys={list(ca.keys())}")

            entry = {"ms": ms}
            for cutoff in CUTOFFS:
                entry[cutoff] = extract_cutoff(result, cutoff)

            # Print diff between consecutive outputs for the hard-weight region.
            # With prefix roll aligned to execution_horizon, old[h:h+d] is the hard target for curr[:d].
            if k > 0 and d > 0:
                joint_dims = list(range(7)) + list(range(8, 15))
                prev = iters[-1]["t0.0"][h:h+d, joint_dims]
                curr = entry["t0.0"][:d, joint_dims]
                hard_diff = curr - prev
                print(
                    f"  [hard14_diff] mean={hard_diff.mean():+.4f} "
                    f"abs_mean={np.abs(hard_diff).mean():.4f}"
                )

            iters.append(entry)
            shape = entry["t0.0"].shape
            missing = [c for c in CUTOFFS if entry[c] is None]
            print(
                f"  iter {k+1:02d}/{args.num_iters}  {ms:.0f} ms  shape={shape}"
                + (f"  missing={missing}" if missing else "")
            )

        # ── ground-truth context ──────────────────────────────────────────────
        # GT covers [start_frame, start_frame + (num_iters-1)*execution_horizon + chunk_len)
        chunk_len = iters[0]["t0.0"].shape[0]
        total_steps = (args.num_iters - 1) * h + chunk_len + 1
        gt_end      = min(len(episode_df), args.start_frame + total_steps)
        gt_actions = np.stack([
            np.asarray(episode_df.iloc[i]["action"], dtype=np.float32)
            for i in range(args.start_frame, gt_end)
        ])
        gt_states = np.stack([
            np.asarray(episode_df.iloc[i]["observation.state"], dtype=np.float32)
            for i in range(args.start_frame, gt_end)
        ])
        infer_states_arr = np.stack(infer_states)
        overlap_stats = compute_overlap_stats(iters, h, infer_states_arr)

        # ── save raw data ─────────────────────────────────────────────────────
        args.output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output_dir / "refinement_data.npz",
            gt_actions   = gt_actions,
            gt_states    = gt_states,
            infer_states = infer_states_arr,
            obs_frame    = np.array(obs_frame),
            latencies_ms = np.array([it["ms"] for it in iters]),
            **{
                f"iters_{c}": np.stack([it[c] for it in iters if it[c] is not None])
                for c in CUTOFFS
                if any(it[c] is not None for it in iters)
            },
        )
        print(f"saved refinement_data.npz")
        with (args.output_dir / "overlap_stats.json").open("w") as f:
            json.dump(overlap_stats, f, indent=2)
        print(f"saved {args.output_dir / 'overlap_stats.json'}")
        if overlap_stats["summary"]:
            print("overlap stats (same absolute time, consecutive chunks):")
            for cutoff in CUTOFFS:
                stats = overlap_stats["summary"].get(cutoff)
                if stats is None:
                    continue
                print(
                    f"  {cutoff}: pairs={stats['pairs']}  "
                    f"mae={stats['mae']:.6f}  rmse={stats['rmse']:.6f}  max={stats['max_abs']:.6f}  "
                    f"rel_mae={stats['relative_mae']:.6f}  rel_rmse={stats['relative_rmse']:.6f}  "
                    f"rel_max={stats['relative_max_abs']:.6f}"
                )

        # ── combined plots ────────────────────────────────────────────────────
        plot_action_trunk(args.output_dir, iters, d, h)
        plot_gt_branches(args.output_dir, iters, gt_actions, gt_states, d, h)

        print(f"output → {args.output_dir.resolve()}")

    finally:
        for r in readers.values():
            r.close()


if __name__ == "__main__":
    main()
