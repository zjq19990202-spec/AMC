"""Inspect DCT bandwidth of π0.5-style right-arm delta action chunks.

For a chunk starting at t, OpenPI's DeltaActions transform uses
``action[t:t+H] - state[t]`` for joint dimensions.  This script evaluates that
exact chunk-level representation rather than framewise ``action[t]-state[t]``.
It reads the right seven CR1 joints (columns 8:15), forms 50-step chunks, and
reports how much DCT energy remains for several low-pass cutoffs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from scipy.fft import dct, idct


RIGHT_JOINTS = slice(8, 15)


def fixed_list_matrix(column) -> np.ndarray:
    array = column.combine_chunks()
    # Native and compacted LeRobot exports respectively use FixedSizeList and
    # List. Both expose a contiguous ``values`` array; infer the common width
    # rather than assuming the former representation.
    values = np.asarray(array.values)
    if len(values) % len(array):
        raise ValueError(f"ragged list column cannot form an action matrix: {array.type}")
    return values.reshape(len(array), len(values) // len(array)).astype(np.float32)


def read_dataset(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {root / 'data'}")
    table = pq.read_table(files, columns=["action", "observation.state", "episode_index"])
    return (
        fixed_list_matrix(table["action"]),
        fixed_list_matrix(table["observation.state"]),
        np.asarray(table["episode_index"].combine_chunks()),
    )


def make_delta_chunks(
    action: np.ndarray,
    state: np.ndarray,
    episode: np.ndarray,
    *,
    horizon: int,
    stride: int,
) -> np.ndarray:
    """Make [windows,H,7] chunks matching OpenPI DeltaActions semantics."""

    chunks: list[np.ndarray] = []
    starts = np.flatnonzero(np.r_[True, episode[1:] != episode[:-1]])
    ends = np.r_[starts[1:], len(episode)]
    for start, end in zip(starts, ends, strict=True):
        for index in range(start, end - horizon + 1, stride):
            # action targets across the horizon are all relative to the state
            # at the *first* frame of this training sample.
            chunks.append(action[index : index + horizon, RIGHT_JOINTS] - state[index, RIGHT_JOINTS])
    if not chunks:
        raise ValueError("no complete action chunks found")
    return np.stack(chunks)


def retained_energy(
    coefficients: np.ndarray, cutoffs: list[int], *, exclude_dc: bool = False
) -> dict[int, np.ndarray]:
    start = 1 if exclude_dc else 0
    total = np.sum(coefficients[:, start:] ** 2, axis=(1, 2))
    total = np.maximum(total, 1e-12)
    return {
        cutoff: np.sum(coefficients[:, start:cutoff] ** 2, axis=(1, 2)) / total
        for cutoff in cutoffs
    }


def retained_energy_per_joint(coefficients: np.ndarray, cutoff: int) -> np.ndarray:
    """Per-window/per-joint fraction, so large-range joints cannot dominate."""

    numerator = np.sum(coefficients[:, :cutoff] ** 2, axis=1)
    denominator = np.maximum(np.sum(coefficients**2, axis=1), 1e-12)
    return numerator / denominator


def rfft_retained_energy_at_hz(chunks: np.ndarray, *, fps: float, cutoff_hz: float) -> np.ndarray:
    """Energy ratio for a real FFT low-pass with correct conjugate weighting."""

    horizon = chunks.shape[1]
    spectrum = np.fft.rfft(chunks, axis=1, norm="ortho")
    energy = np.abs(spectrum) ** 2
    # rFFT stores only non-negative frequencies. Interior coefficients stand
    # for a positive/negative conjugate pair; DC (and Nyquist for even N) do
    # not, hence their weights stay one.
    weights = np.ones(energy.shape[1], dtype=energy.dtype)
    if horizon % 2 == 0:
        weights[1:-1] = 2
    else:
        weights[1:] = 2
    frequency = np.fft.rfftfreq(horizon, d=1 / fps)
    keep = frequency < cutoff_hz
    total = np.sum(energy * weights[None, :, None], axis=(1, 2))
    retained = np.sum(energy[:, keep] * weights[None, keep, None], axis=(1, 2))
    return retained / np.maximum(total, 1e-12)


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
    }


def plot_energy(coefficients: np.ndarray, output: Path, fps: float) -> None:
    energy = np.mean(coefficients**2, axis=(0, 2))
    energy /= np.maximum(energy.sum(), 1e-12)
    frequency = np.arange(len(energy)) * fps / (2 * len(energy))
    fig, axis = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    axis.bar(frequency, energy, width=frequency[1] - frequency[0], color="#2E6FBB")
    axis.axvline(8 * fps / (2 * len(energy)), color="#D34A4A", linestyle="--", label="K=8 cutoff")
    axis.set(xlabel="DCT frequency (Hz)", ylabel="mean relative energy", title="Right-arm delta-action DCT spectrum")
    axis.set_yscale("log")
    axis.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_reconstructions(chunks: np.ndarray, coefficients: np.ndarray, output: Path, cutoffs: list[int]) -> None:
    # Pick the median-energy chunk, avoiding a hand-selected unusually active example.
    total_energy = np.sum(coefficients**2, axis=(1, 2))
    index = int(np.argsort(total_energy)[len(total_energy) // 2])
    target = chunks[index]
    fig, axes = plt.subplots(7, 1, figsize=(11, 12), sharex=True, constrained_layout=True)
    colors = {1: "#D34A4A", 2: "#F28E2B", 4: "#E07A5F", 6: "#F2B134", 8: "#3D9970", 10: "#3A86FF", 12: "#8338EC"}
    for joint, axis in enumerate(axes):
        axis.plot(target[:, joint], color="black", linewidth=1.7, label="delta target" if joint == 0 else None)
        for cutoff in cutoffs:
            kept = coefficients[index].copy()
            kept[cutoff:] = 0
            reconstruction = idct(kept, type=2, axis=0, norm="ortho")
            axis.plot(reconstruction[:, joint], color=colors[cutoff], linewidth=1.1, label=f"K={cutoff}" if joint == 0 else None)
        axis.set_ylabel(f"q{joint + 1}\n(rad)")
        axis.grid(alpha=0.2)
    axes[0].legend(ncol=5, loc="upper right")
    axes[-1].set_xlabel("step in 50-step chunk (30 Hz)")
    fig.suptitle("Median-energy π0.5 delta-action chunk and DCT low-pass reconstructions")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[4, 6, 8, 10, 12, 16])
    args = parser.parse_args()
    if args.horizon <= 1 or args.stride <= 0:
        raise ValueError("horizon must exceed one and stride must be positive")
    if any(cutoff <= 0 or cutoff > args.horizon for cutoff in args.cutoffs):
        raise ValueError("all cutoffs must lie in [1, horizon]")

    all_chunks = []
    source_windows: dict[str, int] = {}
    for root in args.dataset:
        action, state, episode = read_dataset(root)
        chunks = make_delta_chunks(action, state, episode, horizon=args.horizon, stride=args.stride)
        all_chunks.append(chunks)
        source_windows[str(root)] = int(len(chunks))
    chunks = np.concatenate(all_chunks)
    coefficients = dct(chunks, type=2, axis=1, norm="ortho")
    retained = retained_energy(coefficients, args.cutoffs)
    retained_dynamic = retained_energy(coefficients, args.cutoffs, exclude_dc=True)
    full_energy = np.sum(coefficients**2, axis=(1, 2))
    non_dc_energy = np.sum(coefficients[:, 1:] ** 2, axis=(1, 2))
    # A perfectly stationary/constant delta chunk has essentially no shape
    # energy after its DC term. It is valid training data, but cannot tell us
    # which bandwidth is needed for curved trajectories.
    shape_mask = non_dc_energy / np.maximum(full_energy, 1e-12) >= 0.01
    # A DCT K-bin target retains frequencies strictly below K*fps/(2H).
    fft_at_dct_cutoff = {
        cutoff: rfft_retained_energy_at_hz(
            chunks, fps=args.fps, cutoff_hz=cutoff * args.fps / (2 * args.horizon)
        )
        for cutoff in args.cutoffs
    }

    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "representation": "right joint target action[t:t+H,8:15] - state[t,8:15]",
        "horizon": args.horizon,
        "fps": args.fps,
        "window_stride": args.stride,
        "windows": int(len(chunks)),
        "sources": source_windows,
        "retained_dct_energy": {str(cutoff): summarize(values) for cutoff, values in retained.items()},
        "retained_dct_energy_per_joint": {
            str(cutoff): [summarize(retained_energy_per_joint(coefficients, cutoff)[:, joint]) for joint in range(7)]
            for cutoff in args.cutoffs
        },
        "retained_non_dc_dct_energy": {
            str(cutoff): summarize(values) for cutoff, values in retained_dynamic.items()
        },
        "non_dc_shape_windows": int(shape_mask.sum()),
        "retained_non_dc_dct_energy_when_shape_energy_ge_1pct": {
            str(cutoff): summarize(values[shape_mask])
            for cutoff, values in retained_dynamic.items()
        },
        "rfft_retained_energy_at_same_physical_dct_cutoff": {
            str(cutoff): summarize(values) for cutoff, values in fft_at_dct_cutoff.items()
        },
        "frequency_hz": {str(cutoff): cutoff * args.fps / (2 * args.horizon) for cutoff in args.cutoffs},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_energy(coefficients, args.output / "mean_dct_energy.png", args.fps)
    plot_reconstructions(chunks, coefficients, args.output / "reconstructions.png", [4, 6, 8, 10, 12])
    plot_reconstructions(chunks, coefficients, args.output / "reconstruction_k1.png", [1, 2, 4, 8])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
