# AMC: Atomic Motion Coordinate

Minimal research release for **Atomic Motion Coordinate (AMC)**: a
language-steerable, force-responsive extension of the π0.5 action expert.
The repository intentionally contains only the training and inference source
needed for the released spherical model. It does not include robot recordings,
normalization statistics, checkpoint parameters, calibration data, or internal
experiment/analysis scripts.

Project page: https://zjq19990202-spec.github.io/AMC/

## Contents

```
src/atomic_latent_vla/pi05/  AMC model, atomic coordinate, force route and loaders
scripts/train_atomic_zt_fast.py      vision-free atomic grounding (z_T)
scripts/train_atomic_pi05.py         full-observation atomic continuation (z_M)
scripts/train_force_encoder_b1.py    future-wrench encoder pretraining (B1)
scripts/train_force_stage_b2.py      RTC force adaptation (B2)
scripts/serve_force_pi05_keyboard_spherical_b2_5x10.py  spherical 5×10 serving
```

The remaining serving modules are small dependencies of the spherical serving
entry point. They provide the three-camera websocket policy interface and
keyboard subtask selection.

## Dependencies

AMC builds on [OpenPI](https://github.com/Physical-Intelligence/openpi) and
expects its JAX/Flax π0.5 training environment. Install OpenPI separately,
then install this package and expose both source trees:

```bash
git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi && uv sync && cd ..
python -m pip install -e AMC
export PYTHONPATH="$PWD/AMC/src:$PWD/openpi/src:$PWD/openpi/packages/openpi-client/src:${PYTHONPATH:-}"
```

The released scripts require a π0.5 base checkpoint plus task-matched
normalization assets. Force training additionally requires calibrated 120 Hz
wrench/state histories and subtask sidecars. These data and trained parameters
are not redistributed here.

## Training

The stages are run sequentially:

```bash
# 1. Vision-free language/state grounding and atomic codebook learning.
python scripts/train_atomic_zt_fast.py --help

# 2. Full-observation continuation with the frozen codebook.
python scripts/train_atomic_pi05.py --help

# 3. Optional force route: B1 future-wrench prediction, then B2 RTC adaptation.
python scripts/train_force_encoder_b1.py --help
python scripts/train_force_stage_b2.py --help
```

Every command requires explicit dataset, normalization and checkpoint paths;
use `--help` to see the complete contract. No machine-specific data paths or
launcher scripts are included in this release.

## Inference

The spherical force policy serves a WebSocket endpoint for a policy bridge:

```bash
python scripts/serve_force_pi05_keyboard_spherical_b2_5x10.py \
  --checkpoint /path/to/checkpoint \
  --dataset-root /path/to/lerobot_dataset \
  --norm-assets-dir /path/to/norm_assets \
  --norm-asset-id YOUR_NORM_ID \
  --force-norm /path/to/force_norm.json \
  --prompt-file /path/to/subtasks.json
```

It uses a 50-action horizon and regenerates the remaining suffix at offsets
0, 10, 20, 30 and 40. Camera/state/force transport is deployment-specific and
must supply the observation contract expected by the server.

## License and attribution

AMC code is provided for research use. π0.5/OpenPI remains subject to its own
upstream license and terms. Please cite the accompanying AMC paper and OpenPI
when using this release.
