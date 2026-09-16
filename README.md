# AMC: Atomic Motion Coordinate

Source code for spherical AMC: language-steerable and force-responsive dual-arm
manipulation. [Project page and video](https://zjq19990202-spec.github.io/AMC/).

This branch contains source code, configuration and tests. Website assets live
on `gh-pages`. Datasets, normalization assets, checkpoints and experiment outputs
must be supplied separately.

## Spherical implementation

- `src/atomic_latent_vla/pi05/model.py`: atomic queries, spherical motion latent and policy.
- `src/atomic_latent_vla/pi05/config.py`: model options and training filters.
- `src/atomic_latent_vla/pi05/force.py`: force encoding and conditioning.
- `src/atomic_latent_vla/pi05/force_training_data.py`: force training inputs.
- `scripts/train_atomic_zt_fast.py`: vision-free grounding.
- `scripts/train_atomic_pi05.py`: full-observation training.
- `scripts/train_force_encoder_b1.py`: force encoder pretraining.
- `scripts/train_force_stage_b2.py`: force adaptation and RTC.
- `scripts/serve_force_pi05_keyboard_spherical_b2_5x10.py`: spherical B2 serving.

Shared modules retain earlier experimental variants. Select
`spherical_visual_latent=True` and, for force adaptation,
`spherical_force_update=True`. The dedicated spherical serving entry point
automatically selects `spherical-b2-final`: 512-D latents, a two-head full-token
force adapter, a 45-degree spherical cap and a 20-degree force penalty-free cone.
RTC uses offsets 0, 10, 20, 30 and 40 of a 50-action horizon.

## Environment

Use Python 3.11 with the JAX/OpenPI dependencies specified in
`vendor/pi0.5/pyproject.toml`, then install this package:

```bash
python -m pip install -e '.[dev,data]'
export PYTHONPATH="$PWD/src:$PWD/vendor/pi0.5/src:$PWD/vendor/pi0.5/packages/openpi-client/src:${PYTHONPATH:-}"
python scripts/serve_force_pi05_keyboard_spherical_b2_5x10.py --help
```

Checkpoints and action/state/force normalization must match the selected recipe.

## Training recipes

`configs/*spherical*.contract.txt` and `scripts/*spherical*.sh` record spherical
experiments. Launchers contain original machine and dataset paths, which must
be adapted before use. They are not portable one-command installers.

`scripts/launch_force_spherical_b2_unfreeze_ae_zf_5k_8ka_20260907.sh` records the
later 5k RTC continuation with action-path and zF-path training enabled.
Earlier adapter-only configurations remain available for reference.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q
```

Some tests need model dependencies or external robot/data assets. Vendored
OpenPI code retains its upstream license files.
