#!/usr/bin/env python3
"""Train a stock PI0.5 flow policy on the reviewed CR1 union.

The entrypoint deliberately uses OpenPI's unmodified ``Pi0Config(pi05=True)``
and flow-matching loss.  ``--vision-mask-probability 0.5`` implements the
mixed LA/VLA intervention from LA4VLA while keeping architecture, data,
normalization, optimizer, and update budget identical to the plain baseline.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx
from flax.training import common_utils

from openpi.models import model as _model
from openpi.models.pi0_config import Pi0Config
from openpi.shared import array_typing as at
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from openpi.training import sharding
from openpi.training import utils as training_utils
from openpi.training import weight_loaders as _weight_loaders

from atomic_latent_vla.pi05.training_data import batch_to_observation, build_atomic_loader


DEFAULT_ROOTS = (
    "/mnt/cunchu/zjq/target/lerobot_2058_domain_split_20260822/cabinet",
    "/mnt/cunchu/zjq/target/lerobot_2058_domain_split_20260822/drawer",
    "/mnt/cunchu/zjq/target/lerobot_2058_domain_split_20260822/fruit",
    "/mnt/cunchu/zjq/target/lerobot_2058_domain_split_20260822/mixed_rest",
    "/mnt/cunchu/zjq/target/lerobot_2058_screw5_subtask_atomicfix_20260825",
    "/mnt/cunchu/zjq/target/lerobot_plug_vase_split_20260822/plug",
    "/mnt/cunchu/zjq/target/lerobot_vase167_manual_atomicfix_20260825",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", action="append", default=[])
    parser.add_argument("--norm-assets-dir", required=True)
    parser.add_argument("--norm-asset-id", required=True)
    parser.add_argument(
        "--base-params",
        default="/mnt/cunchu/zjq/.cache/openpi/openpi-assets/checkpoints/pi05_base/params",
    )
    parser.add_argument("--checkpoint-base-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--devices", type=int, default=8)
    parser.add_argument("--fsdp-devices", type=int, default=1)
    parser.add_argument("--steps", type=int, default=25_000)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--peak-lr", type=float, default=2.5e-5)
    parser.add_argument("--decay-lr", type=float, default=2.5e-6)
    parser.add_argument("--lr-decay-steps", type=int, default=30_000)
    parser.add_argument("--max-token-len", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--save-interval", type=int, default=1_000)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--vision-mask-probability",
        type=float,
        default=0.0,
        help="Per-sample probability of masking every image token; 0.5 is LA4VLA mixed LA/VLA.",
    )
    parser.add_argument(
        "--prompt-route",
        choices=("subtask_only", "atomic_if_valid_else_subtask"),
        default="subtask_only",
        help=(
            "Select the training text per row. The hybrid route uses the reviewed "
            "arm-specific atomic prompt whenever either arm has strict atomic "
            "supervision, and otherwise falls back to the native SUBtask."
        ),
    )
    parser.add_argument(
        "--restore-full-state",
        type=Path,
        default=None,
        help=(
            "Optional numeric Orbax step directory from which to restore params, "
            "optimizer state, and global step into a new run directory."
        ),
    )
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--skip-checkpoint", action="store_true")
    return parser.parse_args()


def _base_train_module():
    root = "/mnt/cunchu/yc/pi05/scripts"
    if root not in sys.path:
        sys.path.insert(0, root)
    import train as base_train  # noqa: PLC0415

    return base_train


def _select_training_prompt(
    batch_np: dict,
    prompt_route: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return prompt tokens/mask and the row-level atomic routing decision."""

    if prompt_route == "atomic_if_valid_else_subtask":
        use_atomic_prompt = np.any(
            np.asarray(batch_np["atomic_supervision_mask"], dtype=np.bool_), axis=-1
        )
        prompt_tokens = np.where(
            use_atomic_prompt[:, None],
            np.asarray(batch_np["atomic_prompt_tokens"]),
            np.asarray(batch_np["subtask_prompt_tokens"]),
        )
        prompt_mask = np.where(
            use_atomic_prompt[:, None],
            np.asarray(batch_np["atomic_prompt_mask"]),
            np.asarray(batch_np["subtask_prompt_mask"]),
        )
    elif prompt_route == "subtask_only":
        prompt_tokens = np.asarray(batch_np["subtask_prompt_tokens"])
        prompt_mask = np.asarray(batch_np["subtask_prompt_mask"])
        use_atomic_prompt = np.zeros(prompt_tokens.shape[0], dtype=np.bool_)
    else:
        raise ValueError(f"unsupported prompt route: {prompt_route}")
    return prompt_tokens, prompt_mask, use_atomic_prompt


def _to_jax_batch(
    batch_np: dict,
    data_sharding: jax.sharding.NamedSharding,
    prompt_route: str,
):
    observation, actions = batch_to_observation(batch_np)
    prompt_tokens, prompt_mask, use_atomic_prompt = _select_training_prompt(
        batch_np, prompt_route
    )

    # Global task text is never passed to either baseline. The loader guarantees
    # that the fallback SUBtask is populated for every reviewed row.
    observation = observation.replace(
        tokenized_prompt=prompt_tokens,
        tokenized_prompt_mask=prompt_mask,
    )
    observation = jax.tree.map(
        lambda x: jax.make_array_from_process_local_data(data_sharding, x), observation
    )
    actions = jax.make_array_from_process_local_data(data_sharding, actions)
    use_atomic_prompt = jax.make_array_from_process_local_data(
        data_sharding, use_atomic_prompt
    )
    return observation, actions, use_atomic_prompt


def _restore_state_to_requested_sharding(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    state_sharding: training_utils.TrainState,
    *,
    step: int,
) -> training_utils.TrainState:
    """Restore a full PI0.5 train state directly into the current DDP mesh."""

    with at.disable_typechecking():
        train_state, params = _checkpoints._split_params(state)  # noqa: SLF001
        train_state_sharding, params_sharding = _checkpoints._split_params(  # noqa: SLF001
            state_sharding
        )
    restored = checkpoint_manager.restore(
        step,
        args=ocp.args.Composite(
            train_state=ocp.args.PyTreeRestore(
                item=train_state,
                restore_args=ocp.checkpoint_utils.construct_restore_args(
                    train_state, train_state_sharding
                ),
            ),
            params=ocp.args.PyTreeRestore(
                item={"params": params},
                restore_args=ocp.checkpoint_utils.construct_restore_args(
                    {"params": params}, {"params": params_sharding}
                ),
            ),
        ),
    )
    with at.disable_typechecking():
        return _checkpoints._merge_params(  # noqa: SLF001
            restored["train_state"], restored["params"]
        )


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    vision_mask_probability: float,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions, at.Array],
):
    model = nnx.merge(state.model_def, state.params)
    model.train()
    train_rng = jax.random.fold_in(rng, state.step)
    mask_rng, loss_rng = jax.random.split(train_rng)
    observation, actions, use_atomic_prompt = batch
    if vision_mask_probability > 0.0:
        batch_size = actions.shape[0]
        keep_vision = jax.random.uniform(mask_rng, (batch_size,)) >= vision_mask_probability
        observation = observation.replace(
            image_masks={
                key: value & keep_vision for key, value in observation.image_masks.items()
            }
        )
        masked_fraction = 1.0 - jnp.mean(keep_vision.astype(jnp.float32))
    else:
        masked_fraction = jnp.asarray(0.0, dtype=jnp.float32)

    def loss_fn(diff_model: _model.BaseModel):
        flow = diff_model.compute_loss(loss_rng, observation, actions, train=True)
        return jnp.mean(flow)

    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model)
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(model, new_params)
    new_state = dataclasses.replace(
        state,
        step=state.step + 1,
        params=nnx.state(model),
        opt_state=new_opt_state,
    )
    return new_state, {
        "loss": loss,
        "flow_loss": loss,
        "grad_norm": optax.global_norm(grads),
        "vision_masked_fraction": masked_fraction,
        "atomic_prompt_fraction": jnp.mean(use_atomic_prompt.astype(jnp.float32)),
    }


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.vision_mask_probability <= 1.0:
        raise ValueError("vision-mask-probability must be in [0,1]")
    if args.batch_size % args.devices:
        raise ValueError("batch-size must be divisible by devices")
    if args.devices % args.fsdp_devices:
        raise ValueError("fsdp-devices must divide devices")
    if args.max_token_len != 200:
        raise ValueError("stock PI0.5 comparison requires max-token-len=200")
    restore_step = None
    if args.restore_full_state is not None:
        if not args.restore_full_state.is_dir() or not args.restore_full_state.name.isdigit():
            raise FileNotFoundError(
                "restore-full-state must be an existing numeric Orbax step directory: "
                f"{args.restore_full_state}"
            )
        restore_step = int(args.restore_full_state.name)
        if restore_step >= (args.smoke_steps or args.steps):
            raise ValueError(
                f"restore step {restore_step} must be below final step "
                f"{args.smoke_steps or args.steps}"
            )
    roots = tuple(args.dataset_root) if args.dataset_root else DEFAULT_ROOTS
    for root in roots:
        root_path = Path(root)
        for required in (
            "meta/info.json",
            "meta/episode_subtasks.jsonl",
            "meta/start_inactive_arm_block_mask.json",
        ):
            if not (root_path / required).is_file():
                raise FileNotFoundError(root_path / required)

    base_config = _config.get_config("pi05_hdf5_dscrew_v3_jax")
    model_config = Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=50,
        max_token_len=args.max_token_len,
    )
    lr_schedule = dataclasses.replace(
        base_config.lr_schedule,
        warmup_steps=args.warmup_steps,
        peak_lr=args.peak_lr,
        decay_steps=args.lr_decay_steps,
        decay_lr=args.decay_lr,
    )
    config = dataclasses.replace(
        base_config,
        name=args.run_name,
        exp_name=args.run_name,
        model=model_config,
        weight_loader=_weight_loaders.CheckpointWeightLoader(args.base_params),
        lr_schedule=lr_schedule,
        freeze_filter=model_config.get_freeze_filter(),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_train_steps=args.smoke_steps or args.steps,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        checkpoint_base_dir=args.checkpoint_base_dir,
        wandb_enabled=False,
        ema_decay=None,
        fsdp_devices=args.fsdp_devices,
        overwrite=False,
        resume=False,
    )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    logging.info("config=%s", config)
    logging.info(
        "recipe=%s batch=%d (%d/GPU) devices=%d mesh=[%d,%d] prompt=%s",
        "la4vla_mixed" if args.vision_mask_probability else "plain_pi05",
        args.batch_size,
        args.batch_size // args.devices,
        args.devices,
        args.devices // args.fsdp_devices,
        args.fsdp_devices,
        args.prompt_route,
    )

    loader = build_atomic_loader(
        roots,
        norm_assets_dir=args.norm_assets_dir,
        norm_asset_id=args.norm_asset_id,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        action_horizon=50,
        max_token_len=args.max_token_len,
        include_fast=False,
        pad_subtask_horizon=True,
    )
    data_iter = iter(loader)
    first_np = next(data_iter)
    logging.info(
        "first batch actions=%s state=%s prompt_tokens=%s atomic_prompt_rows=%.3f "
        "atomic_arm_labels=%.3f",
        first_np["actions"].shape,
        first_np["state"].shape,
        first_np["subtask_prompt_tokens"].shape,
        float(np.any(first_np["atomic_supervision_mask"], axis=-1).mean()),
        float(first_np["atomic_supervision_mask"].mean()),
    )
    if args.devices > jax.device_count():
        raise ValueError(f"requested {args.devices} devices, found {jax.device_count()}")

    base_train = _base_train_module()
    rng = jax.random.key(args.seed)
    train_rng, init_rng = jax.random.split(rng)
    mesh = sharding.make_mesh(args.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)
    )
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    if resuming:
        raise RuntimeError("baseline comparison must start from PI0.5 base, not resume")
    state, state_sharding = base_train.init_train_state(
        config, init_rng, mesh, resume=restore_step is not None
    )
    checkpoint_loader_view = SimpleNamespace(
        data_config=lambda: SimpleNamespace(norm_stats=None, asset_id=None)
    )
    if restore_step is not None:
        assert args.restore_full_state is not None
        source_manager, source_available = _checkpoints.initialize_checkpoint_dir(
            args.restore_full_state.parent,
            keep_period=None,
            overwrite=False,
            resume=True,
        )
        if not source_available or restore_step not in source_manager.all_steps():
            raise FileNotFoundError(
                f"checkpoint manager cannot see step {restore_step} under "
                f"{args.restore_full_state.parent}"
            )
        state = _restore_state_to_requested_sharding(
            source_manager,
            state,
            state_sharding,
            step=restore_step,
        )
        logging.info(
            "restored complete external train_state step=%d from %s",
            restore_step,
            args.restore_full_state,
        )
    else:
        logging.info("restored stock PI0.5 base params from %s", args.base_params)
    jax.block_until_ready(state)
    ptrain = jax.jit(
        functools.partial(train_step, config, args.vision_mask_probability),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated),
        donate_argnums=(1,),
    )
    loader_view = checkpoint_loader_view
    current_np = first_np
    infos = []
    final_step = config.num_train_steps
    while int(state.step) < final_step:
        batch = _to_jax_batch(current_np, data_sharding, args.prompt_route)
        with sharding.set_mesh(mesh):
            state, info = ptrain(train_rng, state, batch)
        infos.append(info)
        step = int(state.step)
        if step % args.log_interval == 0 or step == 1:
            stacked = common_utils.stack_forest(infos)
            reduced = jax.device_get(jax.tree.map(jnp.mean, stacked))
            logging.info(
                "step=%d %s",
                step,
                " ".join(f"{key}={float(value):.6f}" for key, value in reduced.items()),
            )
            infos = []
        if not args.skip_checkpoint and (step % args.save_interval == 0 or step == final_step):
            _checkpoints.save_state(checkpoint_manager, state, loader_view, step)
            logging.info("submitted checkpoint step=%d", step)
        try:
            current_np = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            current_np = next(data_iter)
    if not args.skip_checkpoint:
        checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main()
