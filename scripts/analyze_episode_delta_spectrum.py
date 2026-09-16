"""Measure physical-frequency DCT energy on complete π0.5 delta episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dct

from analyze_delta_action_spectrum import RIGHT_JOINTS, read_dataset, summarize


def episode_slices(episode: np.ndarray):
    starts = np.flatnonzero(np.r_[True, episode[1:] != episode[:-1]])
    ends = np.r_[starts[1:], len(episode)]
    return zip(starts, ends, strict=True)


def retain_at_hz(values: np.ndarray, fps: float, cutoff_hz: float) -> tuple[float, float, int]:
    """Return full/non-DC retained energy and number of retained DCT bins."""

    coefficients = dct(values, type=2, axis=0, norm="ortho")
    frequency = np.arange(len(values)) * fps / (2 * len(values))
    keep = frequency < cutoff_hz
    full = np.sum(coefficients**2)
    non_dc = np.sum(coefficients[1:] ** 2)
    return (
        float(np.sum(coefficients[keep] ** 2) / max(full, 1e-12)),
        float(np.sum(coefficients[keep & (np.arange(len(values)) > 0)] ** 2) / max(non_dc, 1e-12)),
        int(keep.sum()),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--cutoff-hz", type=float, nargs="+", default=[1.0, 2.0, 3.0, 5.0])
    args = parser.parse_args()

    full: dict[float, list[float]] = {cutoff: [] for cutoff in args.cutoff_hz}
    non_dc: dict[float, list[float]] = {cutoff: [] for cutoff in args.cutoff_hz}
    bins: dict[float, list[int]] = {cutoff: [] for cutoff in args.cutoff_hz}
    sources: dict[str, int] = {}
    lengths: list[int] = []
    for root in args.dataset:
        action, state, episode = read_dataset(root)
        count = 0
        for start, end in episode_slices(episode):
            if end - start < 2:
                continue
            # Exact episode-scale analogue of OpenPI's chunk delta: all future
            # targets are relative to the state at this episode's first frame.
            delta = action[start:end, RIGHT_JOINTS] - state[start, RIGHT_JOINTS]
            lengths.append(len(delta))
            for cutoff in args.cutoff_hz:
                retained, retained_non_dc, kept = retain_at_hz(delta, args.fps, cutoff)
                full[cutoff].append(retained)
                non_dc[cutoff].append(retained_non_dc)
                bins[cutoff].append(kept)
            count += 1
        sources[str(root)] = count

    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "representation": "right action[episode,8:15] - state[episode_start,8:15]",
        "fps": args.fps,
        "episodes": len(lengths),
        "sources": sources,
        "episode_length_frames": summarize(np.asarray(lengths)),
        "retained_dct_energy_at_physical_cutoff_hz": {
            str(cutoff): summarize(np.asarray(values)) for cutoff, values in full.items()
        },
        "retained_non_dc_dct_energy_at_physical_cutoff_hz": {
            str(cutoff): summarize(np.asarray(values)) for cutoff, values in non_dc.items()
        },
        "kept_bins_per_episode": {str(cutoff): summarize(np.asarray(values)) for cutoff, values in bins.items()},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    positions = np.arange(len(args.cutoff_hz))
    axes[0].boxplot([full[x] for x in args.cutoff_hz], tick_labels=args.cutoff_hz, showfliers=False)
    axes[0].set(title="Complete episode: total DCT energy", xlabel="physical low-pass cutoff (Hz)", ylabel="retained energy")
    axes[1].boxplot([non_dc[x] for x in args.cutoff_hz], tick_labels=args.cutoff_hz, showfliers=False)
    axes[1].set(title="Complete episode: trajectory-shape energy", xlabel="physical low-pass cutoff (Hz)", ylabel="retained non-DC energy")
    for axis in axes:
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.2, axis="y")
    fig.savefig(args.output / "episode_retained_energy.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
