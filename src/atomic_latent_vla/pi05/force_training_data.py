"""Training data path for the optional force-conditioned PI0.5 stage."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.training import config as _training_config
from openpi.training.data_loader import TransformedDataset
from openpi.training.lerobot_v3_dataset import LeRobotV3Dataset

from .training_data import _NormalizeWithoutQuantileClipping, _stack_tree


_FORCE_BATCH_KEYS = (
    "slow_force_history",
    "slow_state_history",
    "slow_history_mask",
    "current_force_history",
    "current_state_history",
    "current_history_mask",
    "future_force",
    "future_force_mask",
    "update_offset",
)


def _load_subtask_sidecar(path: str | Path) -> dict[int, tuple[tuple[int, int, str], ...]]:
    """Load strict, per-frame subtask supervision for a force dataset.

    A missing episode is intentionally *not* assigned the dataset task prompt;
    its anchors are excluded by ``_ForceAnchorView``.  This prevents a mixed
    training contract in which a few episodes silently fall back to a global
    prompt.
    """

    source = Path(path)
    result: dict[int, tuple[tuple[int, int, str], ...]] = {}
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            episode = int(row["episode_index"])
            if episode in result:
                raise ValueError(f"duplicate episode {episode} in {source}:{line_number}")
            segments: list[tuple[int, int, str]] = []
            expected_start = 0
            for segment in row.get("semantic_segments", ()):
                start = int(segment["start_frame_30hz"])
                end = int(segment["end_frame_30hz_exclusive"])
                text = str(segment.get("current_subtask", "")).strip()
                if not text:
                    raise ValueError(
                        f"empty subtask for episode {episode} in {source}:{line_number}"
                    )
                if start != expected_start or end <= start:
                    raise ValueError(
                        f"non-contiguous subtask coverage for episode {episode}: "
                        f"expected start {expected_start}, got [{start}, {end})"
                    )
                segments.append((start, end, text))
                expected_start = end
            if not segments:
                raise ValueError(f"episode {episode} has no subtasks in {source}:{line_number}")
            result[episode] = tuple(segments)
    if not result:
        raise ValueError(f"subtask sidecar is empty: {source}")
    return result


@dataclass(frozen=True)
class ForceNormalization:
    """Independent per-channel normalization for force and force-side state."""

    force_q01: np.ndarray
    force_q99: np.ndarray
    state_q01: np.ndarray
    state_q99: np.ndarray
    constant_range_threshold: float = 1.0e-6

    @classmethod
    def load(cls, path: str | Path) -> "ForceNormalization":
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("force_contract", {}).get("arm_identity") != "pooled; no left/right label":
            raise ValueError(f"force norm does not use the shared-arm contract: {source}")
        result = cls(
            force_q01=np.asarray(payload["force"]["q01"], dtype=np.float32),
            force_q99=np.asarray(payload["force"]["q99"], dtype=np.float32),
            state_q01=np.asarray(payload["state"]["q01"], dtype=np.float32),
            state_q99=np.asarray(payload["state"]["q99"], dtype=np.float32),
            constant_range_threshold=float(payload.get("constant_range_threshold", 1.0e-6)),
        )
        if result.force_q01.shape != (6,) or result.force_q99.shape != (6,):
            raise ValueError("force q01/q99 must have shape [6]")
        if result.state_q01.shape != (16,) or result.state_q99.shape != (16,):
            raise ValueError("force-state q01/q99 must have shape [16]")
        return result

    def _normalize(self, values: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        scale = q99 - q01
        normalized = 2.0 * (values - q01) / (scale + 1.0e-6) - 1.0
        # Constant sensors carry no information.  Zero is neutral after the
        # encoder's bias and avoids turning a constant coordinate into -1.
        normalized = np.where(
            scale < self.constant_range_threshold,
            np.zeros_like(normalized),
            normalized,
        )
        return normalized.astype(np.float32, copy=False)

    def normalize_force(self, force: np.ndarray) -> np.ndarray:
        return self._normalize(force, self.force_q01, self.force_q99)

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        return self._normalize(state, self.state_q01, self.state_q99)


def adapt_force_state_to_pi(state: np.ndarray) -> np.ndarray:
    """Vectorized Marvin ``_decode_state(..., adapt_to_pi=True)``."""

    state = np.asarray(state, dtype=np.float32).copy()
    if state.shape[-1] != 16:
        raise ValueError("120 Hz force-side state must end in 16 coordinates")
    flip = np.asarray(
        [1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1],
        dtype=np.float32,
    )
    state *= flip
    linear = np.clip(state[..., [7, 15]] / 2.0, 0.00001, 0.035)
    cosine = (0.018**2 + linear**2 - 0.018**2) / (2.0 * 0.018 * linear)
    angular = np.arccos(np.clip(cosine, -1.0, 1.0))
    state[..., [7, 15]] = (angular - 0.00001) / (1.0 - 0.00001)
    return state.astype(np.float32, copy=False)


def _flatten_rows(values: np.ndarray, start: int, stop: int, width: int) -> np.ndarray:
    selected = np.asarray(values[start:stop], dtype=np.float32)
    expected = (stop - start, 4, width)
    if selected.shape != expected:
        raise ValueError(f"expected force-clock rows {expected}, got {selected.shape}")
    return selected.reshape((stop - start) * 4, width)


class _ForceAnchorView(Dataset[dict[str, Any]]):
    """Valid anchors retaining both wrenches for the shared-arm encoder."""

    # AtomicPi05 arm latents are ordered [right, left].  The order only keeps
    # each sensor associated with the corresponding latent; normalization and
    # every force encoder/projection parameter remain shared across the axis.
    _FORCE_KEYS = (
        "observation.force.right_120hz",
        "observation.force.left_120hz",
    )
    _STATE_KEY = "observation.state_120hz"

    def __init__(
        self,
        root: Path,
        *,
        subtask_sidecar: str | Path | None,
        pad_subtask_horizon: bool,
        truncate_to_sidecar_end: bool,
        action_horizon: int,
        history_rows: int,
        future_rows: int,
        load_future_force_targets: bool,
        update_action_steps: int,
        update_offsets: Sequence[int] | None,
        seed: int,
    ):
        info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
        if int(info["fps"]) != 30:
            raise ValueError(f"force training requires a 30 Hz LeRobot clock: {root}")
        features = info.get("features", {})
        required = {*self._FORCE_KEYS, self._STATE_KEY}
        if missing := sorted(required - features.keys()):
            raise ValueError(f"force dataset is missing {missing}: {root}")
        self.base = LeRobotV3Dataset(
            root,
            delta_timestamps={"action": [step / 30 for step in range(action_horizon)]},
        )
        self.subtasks = (
            _load_subtask_sidecar(subtask_sidecar) if subtask_sidecar is not None else None
        )
        self.pad_subtask_horizon = bool(pad_subtask_horizon)
        self.truncate_to_sidecar_end = bool(truncate_to_sidecar_end)
        self.action_horizon = int(action_horizon)
        self.history_rows = history_rows
        self.future_rows = future_rows
        self.load_future_force_targets = bool(load_future_force_targets)
        self.update_action_steps = update_action_steps
        if update_offsets is None:
            update_offsets = tuple(range(0, future_rows, update_action_steps))
        self.update_offsets = tuple(int(offset) for offset in update_offsets)
        if not self.update_offsets:
            raise ValueError("force RTC update offsets cannot be empty")
        if tuple(sorted(set(self.update_offsets))) != self.update_offsets:
            raise ValueError("force RTC update offsets must be unique and sorted")
        if self.update_offsets[0] < 0 or self.update_offsets[-1] >= future_rows:
            raise ValueError("force RTC update offsets must lie in [0, action_horizon)")
        self.seed = int(seed)
        anchors: list[int] = []
        anchor_episodes: list[int] = []
        for episode in self.base.episode_indices:
            if self.subtasks is not None and int(episode) not in self.subtasks:
                continue
            start, end = self.base._get_episode_bounds(episode)  # noqa: SLF001
            if self.subtasks is not None:
                episode_length = int(end - start)
                sidecar_end = self.subtasks[int(episode)][-1][1]
                if sidecar_end > episode_length:
                    raise ValueError(
                        f"subtask sidecar for episode {episode} ends at {sidecar_end}, "
                        f"beyond dataset length {episode_length}"
                    )
                if sidecar_end != episode_length and not self.truncate_to_sidecar_end:
                    raise ValueError(
                        f"subtask sidecar for episode {episode} ends at {sidecar_end}, "
                        f"but dataset contains {episode_length} frames"
                    )
                if self.truncate_to_sidecar_end:
                    end = start + sidecar_end
            episode_anchors = list(range(start + history_rows - 1, end - future_rows))
            anchors.extend(episode_anchors)
            anchor_episodes.extend([int(episode)] * len(episode_anchors))
        self.anchors = np.asarray(anchors, dtype=np.int64)
        self.anchor_episodes = np.asarray(anchor_episodes, dtype=np.int64)
        if not len(self.anchors):
            raise ValueError(f"force dataset has no complete training windows: {root}")

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, index: int) -> dict[str, Any]:
        anchor = int(self.anchors[int(index)])
        item = dict(self.base[anchor])
        if self.subtasks is not None:
            episode = int(self.base._episode_index[anchor])  # noqa: SLF001
            frame = int(self.base._frame_index[anchor])  # noqa: SLF001
            active_subtask = next(
                (
                    (start, end, text)
                    for start, end, text in self.subtasks[episode]
                    if start <= frame < end
                ),
                None,
            )
            if active_subtask is None:
                raise ValueError(
                    f"no subtask covers episode={episode}, frame={frame}"
                )
            _, subtask_end, prompt = active_subtask
            item["prompt"] = prompt
            if self.pad_subtask_horizon:
                # Match Atomic ZM's ``--pad-subtask-horizon`` contract exactly:
                # the prompt remains the anchor's current subtask and any
                # target steps beyond that semantic boundary hold the final
                # absolute action from the current segment.  Never train one
                # subtask prompt against actions belonging to the next one.
                valid_steps = min(self.action_horizon, subtask_end - frame)
                if valid_steps <= 0:
                    raise ValueError(
                        f"subtask at episode={episode}, frame={frame} has no in-segment action"
                    )
                actions = np.asarray(item["action"]).copy()
                if actions.shape[0] != self.action_horizon:
                    raise ValueError(
                        f"expected {self.action_horizon} action steps, got {actions.shape}"
                    )
                if valid_steps < self.action_horizon:
                    actions[valid_steps:] = actions[valid_steps - 1]
                item["action"] = actions
        return item

    def force_metadata(self, index: int, norm: ForceNormalization) -> dict[str, np.ndarray]:
        anchor = int(self.anchors[int(index)])
        force_values = [
            self.base._extra_numeric[key]  # noqa: SLF001
            for key in self._FORCE_KEYS
        ]
        state_values = self.base._extra_numeric[self._STATE_KEY]  # noqa: SLF001
        slow_start = anchor - self.history_rows + 1
        slow_force = np.stack(
            [_flatten_rows(values, slow_start, anchor + 1, 6) for values in force_values],
            axis=0,
        )
        slow_state = _flatten_rows(state_values, slow_start, anchor + 1, 16)
        if getattr(self, "load_future_force_targets", True):
            future_force = np.stack(
                [
                    _flatten_rows(values, anchor + 1, anchor + 1 + self.future_rows, 6)
                    for values in force_values
                ],
                axis=0,
            )
            normalized_future_force = norm.normalize_force(future_force)
            future_force_mask = np.ones(future_force.shape[:2], dtype=np.bool_)
        else:
            # Keep a stable JAX batch/tree contract while avoiding the 200-point
            # target slice, normalization, collation, and transfer in B2.  The
            # model's disabled future-loss branch must not consume this masked
            # neutral placeholder.
            normalized_future_force = np.zeros(
                (len(self._FORCE_KEYS), 1, 6), dtype=np.float32
            )
            future_force_mask = np.zeros(
                normalized_future_force.shape[:2], dtype=np.bool_
            )
        update_offset = self.update_offsets[
            ((anchor * 0x85EBCA6B) ^ self.seed) % len(self.update_offsets)
        ]
        current_anchor = anchor + update_offset
        current_start = current_anchor - self.history_rows + 1
        current_force = np.stack(
            [
                _flatten_rows(values, current_start, current_anchor + 1, 6)
                for values in force_values
            ],
            axis=0,
        )
        current_state = _flatten_rows(state_values, current_start, current_anchor + 1, 16)
        # Fast conditioning is causal relative to the slow anchor. At offset 0
        # no new measurements exist, so only z_F may be read. Later offsets
        # expose exactly the samples acquired after the anchor; pre-anchor
        # samples already summarized by z_F must not be counted a second time.
        acquired_samples = min(update_offset * 4, current_force.shape[1])
        current_mask = np.zeros(current_force.shape[:2], dtype=np.bool_)
        if acquired_samples:
            current_mask[:, -acquired_samples:] = True
        slow_state = adapt_force_state_to_pi(slow_state)
        current_state = adapt_force_state_to_pi(current_state)
        return {
            "slow_force_history": norm.normalize_force(slow_force),
            "slow_state_history": norm.normalize_state(slow_state),
            "slow_history_mask": np.ones(slow_force.shape[:2], dtype=np.bool_),
            "current_force_history": norm.normalize_force(current_force),
            "current_state_history": norm.normalize_state(current_state),
            "current_history_mask": current_mask,
            "future_force": normalized_future_force,
            "future_force_mask": future_force_mask,
            "update_offset": np.asarray(update_offset, dtype=np.int32),
        }


class _ForceProcessedDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        raw: _ForceAnchorView,
        transforms: Sequence[_transforms.DataTransformFn],
        norm: ForceNormalization,
    ):
        prompted = (
            raw
            if raw.subtasks is not None
            else TransformedDataset(raw, [_transforms.PromptFromLeRobotTask(raw.base.tasks)])
        )
        self._core = TransformedDataset(prompted, transforms)
        self._raw = raw
        self._norm = norm

    def __len__(self) -> int:
        return len(self._raw)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {**self._core[int(index)], **self._raw.force_metadata(int(index), self._norm)}


def force_collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot collate an empty force batch")
    return {key: _stack_tree(rows, key) for key in rows[0]}


def _worker_init_fn(_: int) -> None:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


def build_force_dataset(
    roots: Sequence[str | Path],
    *,
    subtask_sidecars: Sequence[str | Path | None] | None = None,
    pad_subtask_horizon: bool = False,
    truncate_to_sidecar_end: bool = False,
    norm_assets_dir: str | Path,
    norm_asset_id: str,
    force_norm_path: str | Path,
    action_horizon: int = 50,
    max_token_len: int = 144,
    force_history_samples: int = 120,
    force_future_samples: int = 200,
    force_temporal_stride: int = 4,
    force_update_action_steps: int = 10,
    force_update_offsets: Sequence[int] | None = None,
    load_future_force_targets: bool = True,
    seed: int = 0,
) -> Dataset[dict[str, Any]]:
    root_paths = tuple(Path(root) for root in roots)
    if not root_paths:
        raise ValueError("at least one force dataset root is required")
    if subtask_sidecars is None:
        subtask_sidecars = (None,) * len(root_paths)
    else:
        subtask_sidecars = tuple(subtask_sidecars)
        if len(subtask_sidecars) != len(root_paths):
            raise ValueError("one --subtask-sidecar is required for each force dataset root")
    if force_history_samples % force_temporal_stride:
        raise ValueError("force history must be divisible by temporal stride")
    if force_future_samples % force_temporal_stride:
        raise ValueError("future force must be divisible by temporal stride")
    history_rows = force_history_samples // force_temporal_stride
    future_rows = force_future_samples // force_temporal_stride
    if future_rows != action_horizon:
        raise ValueError("future force rows must equal the action horizon")

    bridge_config = pi0_config.Pi0Config(pi05=True, max_token_len=max_token_len)
    data_factory = _training_config.LeRobotMarvinDataConfig(
        repo_id=str(root_paths[0]),
        prompt_from_task=True,
        adapt_to_pi=True,
        assets=_training_config.AssetsConfig(
            assets_dir=str(norm_assets_dir), asset_id=norm_asset_id
        ),
        repack_transforms=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.images.base_0_rgb",
                            "cam_left_wrist": "observation.images.left_wrist_0_rgb",
                            "cam_right_wrist": "observation.images.right_wrist_0_rgb",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        ),
    )
    data_config = data_factory.create(Path(norm_assets_dir), bridge_config)
    transforms = (
        [*data_config.repack_transforms.inputs]
        + [*data_config.data_transforms.inputs]
        + [
            _NormalizeWithoutQuantileClipping(
                data_config.norm_stats,
                use_quantiles=data_config.use_quantile_norm,
            )
        ]
        + [*data_config.model_transforms.inputs]
    )
    force_norm = ForceNormalization.load(force_norm_path)
    datasets = [
        _ForceProcessedDataset(
            _ForceAnchorView(
                root,
                subtask_sidecar=subtask_sidecar,
                pad_subtask_horizon=pad_subtask_horizon,
                truncate_to_sidecar_end=truncate_to_sidecar_end,
                action_horizon=action_horizon,
                history_rows=history_rows,
                future_rows=future_rows,
                load_future_force_targets=load_future_force_targets,
                update_action_steps=force_update_action_steps,
                update_offsets=force_update_offsets,
                seed=seed + root_index * 104729,
            ),
            transforms,
            force_norm,
        )
        for root_index, (root, subtask_sidecar) in enumerate(
            zip(root_paths, subtask_sidecars, strict=True)
        )
    ]
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def build_force_loader(
    roots: Sequence[str | Path],
    *,
    subtask_sidecars: Sequence[str | Path | None] | None = None,
    pad_subtask_horizon: bool = False,
    truncate_to_sidecar_end: bool = False,
    norm_assets_dir: str | Path,
    norm_asset_id: str,
    force_norm_path: str | Path,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 0,
    action_horizon: int = 50,
    max_token_len: int = 144,
    force_history_samples: int = 120,
    force_future_samples: int = 200,
    force_temporal_stride: int = 4,
    force_update_action_steps: int = 10,
    force_update_offsets: Sequence[int] | None = None,
    load_future_force_targets: bool = True,
) -> DataLoader:
    dataset = build_force_dataset(
        roots,
        subtask_sidecars=subtask_sidecars,
        pad_subtask_horizon=pad_subtask_horizon,
        truncate_to_sidecar_end=truncate_to_sidecar_end,
        norm_assets_dir=norm_assets_dir,
        norm_asset_id=norm_asset_id,
        force_norm_path=force_norm_path,
        action_horizon=action_horizon,
        max_token_len=max_token_len,
        force_history_samples=force_history_samples,
        force_future_samples=force_future_samples,
        force_temporal_stride=force_temporal_stride,
        force_update_action_steps=force_update_action_steps,
        force_update_offsets=force_update_offsets,
        load_future_force_targets=load_future_force_targets,
        seed=seed,
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        drop_last=True,
        collate_fn=force_collate,
        worker_init_fn=_worker_init_fn,
        generator=generator,
    )


def _episode_split_indices(
    dataset: Dataset[dict[str, Any]],
    *,
    validation_modulus: int,
    validation_remainder: int,
) -> tuple[list[int], list[int]]:
    """Split anchors by episode, including concatenated multi-root datasets."""

    if validation_modulus <= 1:
        raise ValueError("validation_modulus must be greater than one")
    if not 0 <= validation_remainder < validation_modulus:
        raise ValueError("validation_remainder must be in [0, validation_modulus)")
    children = dataset.datasets if isinstance(dataset, ConcatDataset) else (dataset,)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    offset = 0
    for child in children:
        if not isinstance(child, _ForceProcessedDataset):
            raise TypeError("force episode split requires _ForceProcessedDataset children")
        is_validation = (
            child._raw.anchor_episodes % validation_modulus  # noqa: SLF001
            == validation_remainder
        )
        local_indices = np.arange(len(child), dtype=np.int64) + offset
        train_indices.extend(local_indices[~is_validation].tolist())
        validation_indices.extend(local_indices[is_validation].tolist())
        offset += len(child)
    if not train_indices or not validation_indices:
        raise ValueError("episode split produced an empty training or validation subset")
    return train_indices, validation_indices


def build_force_train_validation_loaders(
    roots: Sequence[str | Path],
    *,
    subtask_sidecars: Sequence[str | Path | None] | None = None,
    pad_subtask_horizon: bool = False,
    truncate_to_sidecar_end: bool = False,
    norm_assets_dir: str | Path,
    norm_asset_id: str,
    force_norm_path: str | Path,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 0,
    validation_modulus: int = 10,
    validation_remainder: int = 0,
    action_horizon: int = 50,
    max_token_len: int = 144,
    force_history_samples: int = 120,
    force_future_samples: int = 200,
    force_temporal_stride: int = 4,
    force_update_action_steps: int = 10,
    force_update_offsets: Sequence[int] | None = None,
    load_future_force_targets: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Build one shared dataset with a deterministic episode-level holdout."""

    dataset = build_force_dataset(
        roots,
        subtask_sidecars=subtask_sidecars,
        pad_subtask_horizon=pad_subtask_horizon,
        truncate_to_sidecar_end=truncate_to_sidecar_end,
        norm_assets_dir=norm_assets_dir,
        norm_asset_id=norm_asset_id,
        force_norm_path=force_norm_path,
        action_horizon=action_horizon,
        max_token_len=max_token_len,
        force_history_samples=force_history_samples,
        force_future_samples=force_future_samples,
        force_temporal_stride=force_temporal_stride,
        force_update_action_steps=force_update_action_steps,
        force_update_offsets=force_update_offsets,
        load_future_force_targets=load_future_force_targets,
        seed=seed,
    )
    train_indices, validation_indices = _episode_split_indices(
        dataset,
        validation_modulus=validation_modulus,
        validation_remainder=validation_remainder,
    )
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
        "pin_memory": True,
        "drop_last": True,
        "collate_fn": force_collate,
        "worker_init_fn": _worker_init_fn,
    }
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        **common,
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices),
        shuffle=False,
        generator=torch.Generator().manual_seed(seed),
        **common,
    )
    return train_loader, validation_loader


def batch_to_force_inputs(
    batch: dict[str, Any],
) -> tuple[_model.Observation, np.ndarray, dict[str, np.ndarray]]:
    core = {key: value for key, value in batch.items() if key not in _FORCE_BATCH_KEYS}
    actions = core.pop("actions")
    force = {key: np.asarray(batch[key]) for key in _FORCE_BATCH_KEYS}
    return _model.Observation.from_dict(core), np.asarray(actions), force
