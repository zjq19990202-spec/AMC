#!/usr/bin/env python3
"""Render the reviewed three-frame fruit steering comparison for publication."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

from evaluate_global_episode_chunks import _endpoint_tcp_from_actions


PANELS = (
    {"episode": 100, "frame": 890, "arm": "right", "native": "yellow pear", "targets": ("yellow pear", "carrot", "banana")},
    {"episode": 95, "frame": 280, "arm": "left", "native": "orange", "targets": ("orange", "banana", "green bitter melon")},
    {"episode": 2, "frame": 520, "arm": "right", "native": "carrot", "targets": ("carrot", "orange", "red chili pepper")},
)

# Fruit identity is encoded by hue; model identity is encoded by line style.
# The palette remains separable under common red-green color-vision deficiencies.
COLORS = {
    "yellow pear": "#377EB8",
    "carrot": "#7B2CBF",
    "banana": "#D9A400",
    "green bitter melon": "#009E73",
    "orange": "#E66100",
    "red chili pepper": "#D62728",
}

LABEL_OFFSETS = {
    (100, 890, "yellow pear"): (-45, -17),
    (100, 890, "carrot"): (6, 5),
    (100, 890, "banana"): (6, 5),
    (95, 280, "orange"): (6, 5),
    (95, 280, "banana"): (6, 5),
    (95, 280, "green bitter melon"): (6, 5),
    (2, 520, "carrot"): (6, -24),
    (2, 520, "orange"): (6, 5),
    (2, 520, "red chili pepper"): (6, 5),
}

# Fixed camera-plane limits make displacement magnitudes directly comparable
# across panels and keep all three lower axes physically identical in size.
COMMON_XLIM = (-0.16, 0.27)
COMMON_YLIM = (0.34, 0.56)


def _trim_black_border(image: np.ndarray) -> np.ndarray:
    """Remove only nearly black outer rows/columns; preserve the camera FOV."""
    bright = image.astype(np.float32).mean(axis=2) > 8.0
    rows = np.flatnonzero(bright.mean(axis=1) > 0.08)
    cols = np.flatnonzero(bright.mean(axis=0) > 0.08)
    if not len(rows) or not len(cols):
        return image
    return image[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain-json", type=Path, required=True)
    ap.add_argument("--afro-npz", type=Path, required=True)
    ap.add_argument("--target-json", type=Path, required=True)
    ap.add_argument("--paired-json", type=Path, required=True)
    ap.add_argument("--source-image-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plain = json.loads(args.plain_json.read_text())
    plain_rows = {(r["episode"], r["frame"]): r for r in plain["rows"]}
    afro = np.load(args.afro_npz, allow_pickle=True)
    targets = {
        (r["episode"], r["frame"], r["fruit"]): np.asarray(r["target_xyz_m"])
        for r in json.loads(args.target_json.read_text())["rows"]
    }
    paired = {
        (r["episode"], r["frame"], r["fruit"]): r
        for r in json.loads(args.paired_json.read_text())["rows"]
    }

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(13.8, 6.55), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(0.86, 1.14), hspace=0.03)
    report = []

    for column, spec in enumerate(PANELS):
        ep, frame, arm = spec["episode"], spec["frame"], spec["arm"]
        image_ax = fig.add_subplot(grid[0, column])
        ax = fig.add_subplot(grid[1, column])
        row = plain_rows[(ep, frame)]
        prefix = f"episode_{ep:06d}_frame_{frame:06d}"
        names = [str(x) for x in afro[prefix + "_prompt_names"]]

        image_path = args.source_image_dir / f"ep{ep:03d}_f{frame:03d}.png"
        image = _trim_black_border(np.asarray(Image.open(image_path).convert("RGB")))
        image_ax.imshow(image, interpolation="lanczos")
        image_ax.set_axis_off()
        image_ax.set_title(
            f"({chr(97 + column)}) Episode {ep}, frame {frame} · {arm.capitalize()} arm\n"
            f"Native: {row['native_subtask']}",
            pad=5,
        )

        points = []
        panel_report = []
        for fruit in spec["targets"]:
            info = paired[(ep, frame, fruit)]
            if info["plain_arm"] != arm or info["afro_moving_arm"] != arm:
                raise ValueError(f"arm mismatch for episode={ep}, frame={frame}, fruit={fruit}")
            prompt_index = next(i for i, text in enumerate(row["prompts"]) if fruit in text.lower())
            plain_traj = np.asarray(row[f"{arm}_tcp_trajectories_m"])[prompt_index]
            atomic_traj = _endpoint_tcp_from_actions(
                None, 0, afro[prefix + "_prediction_actions"][names.index(fruit)], arm
            )
            target = targets[(ep, frame, fruit)]
            color = COLORS[fruit]
            width = 2.5 if fruit == spec["native"] else 2.0
            # Camera-like top view: horizontal=-base y, vertical=base x.
            px, py = -plain_traj[:, 1], plain_traj[:, 0]
            ox, oy = -atomic_traj[:, 1], atomic_traj[:, 0]
            tx, ty = -target[1], target[0]
            ax.plot(px, py, color=color, ls=(0, (4, 2.3)), lw=width, alpha=0.72, zorder=2)
            ax.plot(ox, oy, color=color, ls="-", lw=width + 0.25, zorder=3)
            ax.scatter(px[-1], py[-1], color=color, marker="x", s=38, lw=1.6, zorder=5)
            ax.scatter(ox[-1], oy[-1], facecolor="white", edgecolor=color, marker="o", s=39, lw=1.7, zorder=6)
            ax.scatter(tx, ty, facecolor=color, edgecolor="white", marker="*", s=160, lw=0.8, zorder=7)
            role = "Native" if fruit == spec["native"] else "Steer"
            ax.annotate(
                f"{role}: {fruit}\nPI $\pi_{{0.5}}$ {info['plain_final_mm']:.0f} / Ours {info['final_mm']:.0f} mm",
                (tx, ty), xytext=LABEL_OFFSETS[(ep, frame, fruit)], textcoords="offset points",
                fontsize=7.4, color=color, fontweight="semibold", linespacing=1.05,
            )
            points.extend(np.column_stack((px, py)))
            points.extend(np.column_stack((ox, oy)))
            points.append((tx, ty))
            panel_report.append({
                "fruit": fruit,
                "role": role.lower(),
                "arm": arm,
                "pi05_final_mm": info["plain_final_mm"],
                "ours_atomic_final_mm": info["final_mm"],
                "ours_advantage_mm": info["afro_advantage_final_mm"],
            })

        ax.set_xlim(*COMMON_XLIM)
        ax.set_ylim(*COMMON_YLIM)
        ax.set_aspect("equal", adjustable="box")
        ax.set_facecolor("#FAFAFA")
        ax.grid(color="#D6D9DE", lw=0.55, alpha=0.75)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Image horizontal  ($-y_{base}$, m)")
        ax.set_ylabel("Image vertical  ($x_{base}$, m)")
        report.append({**spec, "native_prompt": row["native_subtask"], "results": panel_report})

    legend = [
        Line2D([0], [0], color="#333333", ls=(0, (4, 2.3)), lw=2.2, label=r"PI $\pi_{0.5}$"),
        Line2D([0], [0], color="#333333", ls="-", lw=2.4, label="Ours (Atomic)"),
        Line2D([0], [0], marker="*", markerfacecolor="#555555", markeredgecolor="white", ls="", markersize=11, label="Fruit target"),
        Line2D([0], [0], marker="x", color="#555555", ls="", label=r"PI $\pi_{0.5}$ endpoint"),
        Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor="#555555", ls="", label="Ours endpoint"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.012))
    fig.suptitle("Target-conditioned fruit steering from the same observation", fontsize=13, fontweight="semibold")

    stem = args.output_dir / "fruit_target_steering_ours_atomic"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.035)
    fig.savefig(stem.with_suffix(".png"), dpi=450, bbox_inches="tight", pad_inches=0.035)
    (stem.with_suffix(".json")).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
