"""Atomic π0.5 LeRobot loader that preserves annotation sidecars.

OpenPI's standard v3 loader intentionally discards string parquet columns and
its generic collate function returns only ``Observation, actions``.  The
atomic model needs one extra text sequence, atomic labels, and raw joint
positions for its coefficient target, so this module keeps those fields out
of the frozen π0.5 data path and attaches them after the normal transforms.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import tokenizer as _tokenizer
from openpi.policies import marvin_policy as _marvin_policy
from openpi.training import config as _training_config
from openpi.training.data_loader import TransformedDataset
from openpi.training import lerobot_v3_dataset as _lerobot_v3
from openpi.training.lerobot_v3_dataset import LeRobotV3Dataset
from openpi import transforms as _transforms

from atomic_latent_vla.annotation.motion import CartesianPose, base_frame_pose_delta
from atomic_latent_vla.atomic import ATOMIC_NAMES, NUM_ATOMIC_SKILLS, STAY_ATOMIC_ID
from atomic_latent_vla.data.atomic_ratios import aggregate_horizon_atomic_weights
from atomic_latent_vla.data.gating import topk_gate_composition
from atomic_latent_vla.tcp import (
    BIMANUAL_TCP_LOCAL_Z_OFFSET_M,
    BIMANUAL_TCP_POSE_METADATA,
    BIMANUAL_TCP_POSE_SIDECAR,
    TCP_POSE_SIDECAR,
)


_RAW_EXTRA_KEYS = {
    "atomic_prompt",
    "global_prompt",
    "subtask_prompt",
    "subtask_supervision_mask",
    "atomic_weights",
    "atomic_supervision_mask",
    "atomic_composition_weights",
    "atomic_composition_confidence",
    "atomic_composition_mask",
    "tcp_twist_delta",
    "raw_state",
    "raw_actions",
    "data_index",
    "episode_index",
    "frame_index",
    "zt_teacher_directions",
    "zt_teacher_valid",
}


_TCP_POSE_SIDECAR = TCP_POSE_SIDECAR
_BIMANUAL_TCP_POSE_SIDECAR = BIMANUAL_TCP_POSE_SIDECAR


@lru_cache(maxsize=None)
def _paligemma_tokenizer(max_token_len: int) -> _tokenizer.PaligemmaTokenizer:
    """One read-only tokenizer instance per DataLoader process and length."""

    return _tokenizer.PaligemmaTokenizer(max_token_len)


@lru_cache(maxsize=None)
def _fast_tokenizer(max_token_len: int) -> _tokenizer.FASTTokenizer:
    """Avoid loading the HuggingFace FAST processor once per dataset root."""

    return _tokenizer.FASTTokenizer(max_token_len)


@dataclass(frozen=True)
class _NormalizeWithoutQuantileClipping:
    """Match the original PI0.5 q01/q99 affine normalization exactly.

    The pinned ``ycchen112/pi05`` fork clips every quantile-normalized leaf to
    ``[-1, 1]``.  The reference PI0.5 checkout uses q01/q99 only as an affine
    scale, retaining both state inputs and action targets outside that range.
    Override the two native CR1 leaves here so training is not dependent on
    that fork-specific behavior.
    """

    norm_stats: Any
    use_quantiles: bool

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        raw_values = {
            key: np.asarray(data[key]).copy()
            for key in ("state", "actions")
            if key in data
        }
        normalized = _transforms.Normalize(
            self.norm_stats, use_quantiles=self.use_quantiles
        )(data)
        if not self.use_quantiles or self.norm_stats is None:
            return normalized

        result = {**normalized}
        for key, raw in raw_values.items():
            stats = self.norm_stats.get(key)
            if stats is None:
                continue
            if stats.q01 is None or stats.q99 is None:
                raise ValueError(f"quantile {key} normalization requires q01/q99")
            q01 = np.asarray(stats.q01)
            q99 = np.asarray(stats.q99)
            dim = min(q01.shape[-1], raw.shape[-1])
            head = (
                (raw[..., :dim] - q01[..., :dim])
                / (q99[..., :dim] - q01[..., :dim] + 1e-6)
                * 2.0
                - 1.0
            ).astype(raw.dtype, copy=False)
            result[key] = (
                head
                if dim == raw.shape[-1]
                else np.concatenate([head, raw[..., dim:]], axis=-1)
            )
        return result


def _adapt_marvin_values_to_pi(
    raw_state: np.ndarray,
    raw_actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply admin123 PI0.5's Marvin ``adapt_to_pi=True`` value contract.

    The public Marvin transform also converts images, which the text-only zT
    loader deliberately never decodes.  Reuse its exact value helpers here so
    zT and image-conditioned zM see identical joint signs and gripper units.
    """

    state = _marvin_policy._decode_state(  # noqa: SLF001
        np.asarray(raw_state, dtype=np.float32).copy(),
        adapt_to_pi=True,
    )
    actions = _marvin_policy._encode_actions_inv(  # noqa: SLF001
        np.asarray(raw_actions, dtype=np.float32).copy(),
        adapt_to_pi=True,
    )
    return state.astype(np.float32, copy=False), actions.astype(np.float32, copy=False)


def _tokenize_subtask_target(text: str, max_token_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Tokenize an AR text target without π0.5's task/state wrapper."""

    tokenizer = _paligemma_tokenizer(max_token_len)._tokenizer
    cleaned = text.strip().replace("_", " ").replace("\n", " ")
    tokens = tokenizer.encode(cleaned, add_bos=False, add_eos=True)[:max_token_len]
    mask = np.zeros(max_token_len, dtype=np.bool_)
    mask[: len(tokens)] = True
    padded = np.zeros(max_token_len, dtype=np.int32)
    padded[: len(tokens)] = np.asarray(tokens, dtype=np.int32)
    return padded, mask


def compose_horizon_subtasks(texts: Sequence[str]) -> str:
    """Join the ordered, de-duplicated subtasks touched by one action horizon."""

    cleaned: list[str] = []
    for value in texts:
        text = " ".join(str(value).strip().split()).rstrip(" .;")
        if text and (not cleaned or text != cleaned[-1]):
            cleaned.append(text)
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    return "; then ".join(cleaned)


def compose_arm_atomic_prompts(present: Sequence[tuple[str, str]]) -> str:
    """Compose atomic text with an explicit arm identity for every labeled arm."""

    return " ".join(f"{arm}: {prompt}" for arm, prompt in present)


@dataclass(frozen=True)
class _AtomicHorizonAnnotation:
    prompt: str
    weights: np.ndarray


@dataclass(frozen=True)
class _AtomicCompositionAnnotation:
    """Top-k-normalized gate composition for one complete 50-frame horizon."""

    weights: np.ndarray
    confidence: float


def _topk_gate_composition(
    gate_probabilities: np.ndarray,
    *,
    top_k: int = 5,
) -> tuple[np.ndarray, float]:
    """Compatibility wrapper for the lightweight shared gate helper."""

    return topk_gate_composition(gate_probabilities, top_k=top_k)


class _TargetAnnotationSidecars:
    """Compact index for target's episode/global/subtask/atomic-horizon JSONLs.

    Each 3 Hz block labels the complete one-third-second interval represented
    by that block.  At 30 Hz, all ten action-window starts in the interval use
    the same reviewed 50-step FK-horizon label. A grouped ``start..end`` record
    means that every integer block in that run shares the prompt/labels.
    """

    def __init__(
        self,
        root: Path,
        atomic_composition_sidecar: str | Path | None = None,
    ):
        meta = root / "meta"
        self.global_prompts: dict[int, str] = {}
        self.subtasks: dict[int, tuple[tuple[int, int, str], ...]] = {}
        self.atomic_horizons: dict[int, dict[str, dict[int, _AtomicHorizonAnnotation]]] = {}
        self.atomic_compositions: dict[
            int, dict[str, dict[int, _AtomicCompositionAnnotation]]
        ] = {}

        global_path = meta / "global_episode_prompts.jsonl"
        subtask_path = meta / "episode_subtasks.jsonl"
        atomic_path = meta / "atomic_horizon_prompts_3hz.jsonl"
        if not (global_path.is_file() and subtask_path.is_file() and atomic_path.is_file()):
            raise FileNotFoundError("target annotation JSONLs are incomplete")

        with global_path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                self.global_prompts[int(row["episode_index"])] = str(row["global_prompt"]).strip()

        with subtask_path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                segments: list[tuple[int, int, str]] = []
                for segment in row.get("semantic_segments", []):
                    text = str(segment.get("current_subtask", "")).strip()
                    if not text:
                        continue
                    segments.append(
                        (
                            int(segment["start_frame_30hz"]),
                            int(segment["end_frame_30hz_exclusive"]),
                            text,
                        )
                    )
                self.subtasks[int(row["episode_index"])] = tuple(segments)

        name_to_id = {name: index for index, name in enumerate(ATOMIC_NAMES)}
        with atomic_path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                arm = str(row.get("arm"))
                if arm not in {"right", "left"}:
                    continue
                labels = [str(label) for label in row.get("fk_atomic_labels", [])]
                if not 1 <= len(labels) <= 2 or any(label not in name_to_id for label in labels):
                    continue
                weights = np.zeros(NUM_ATOMIC_SKILLS, dtype=np.float32)
                for label in labels:
                    weights[name_to_id[label]] = 1.0 / len(labels)
                annotation = _AtomicHorizonAnnotation(str(row["prompt"]).strip(), weights)
                by_anchor = self.atomic_horizons.setdefault(
                    int(row["episode_index"]), {}
                ).setdefault(arm, {})
                for block in range(int(row["block_start_id"]), int(row["block_end_id"]) + 1):
                    if block in by_anchor:
                        raise ValueError(
                            f"overlapping {arm} atomic horizons for episode "
                            f"{row['episode_index']} block {block}"
                        )
                    by_anchor[block] = annotation


        # Qwen reviews only motion prompts. Restore the thirteenth `stay` atom
        # from the strict FK gate, which declares idle only when the complete
        # five-block / 50-step horizon is inactive. Use recomputed tcp200 FK on
        # both arms so the label contract matches the reconstruction target.
        configured_fk_root = (
            Path(atomic_composition_sidecar)
            if atomic_composition_sidecar is not None
            else Path("fk_horizon_3hz_gate_top5_v1")
        )
        versioned_fk_root = (
            configured_fk_root
            if configured_fk_root.is_absolute()
            else meta / configured_fk_root
        )
        fk_root = (
            versioned_fk_root
            if versioned_fk_root.is_dir()
            else meta / "fk_horizon_3hz"
        )
        arm_directories = {
            "right": (
                fk_root / "right_recomputed_0p20m"
                if (fk_root / "right_recomputed_0p20m").is_dir()
                else fk_root / "right"
            ),
            "left": fk_root / "left",
        }
        stay_prompts = {
            "right": "Stay stationary.",
            "left": "Stay stationary.",
        }
        for arm, directory in arm_directories.items():
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("episode_*.json")):
                payload = json.loads(path.read_text(encoding="utf-8"))
                episode_index = int(payload.get("episode_index", path.stem.split("_")[-1]))
                by_anchor = self.atomic_horizons.setdefault(episode_index, {}).setdefault(
                    arm, {}
                )
                by_composition = self.atomic_compositions.setdefault(
                    episode_index, {}
                ).setdefault(arm, {})
                for segment in payload.get("segments", []):
                    block = int(
                        segment.get(
                            "segment_id", round(float(segment["start_s"]) * 3.0)
                        )
                    )
                    mode = str(segment.get("gate_mode", ""))
                    reason = str(segment.get("gate_reason", ""))
                    # Tail blocks do not contain the complete five-block /
                    # 50-frame target and therefore must not receive a soft
                    # composition loss.
                    has_complete_horizon = "insufficient future coverage" not in reason
                    if has_complete_horizon and mode in {"dual", "drop"}:
                        # Reuse the materialized Drop Top-5 target. For Dual,
                        # take Top-2 from the already-saved gate vector. The
                        # trainer routes Dual composition only to zT and Drop
                        # composition only to zM; neither path recomputes FK or
                        # the gate itself.
                        stored_mask = bool(
                            segment.get("atomic_composition_supervision_mask", False)
                        )
                        stored = segment.get("atomic_composition_target")
                        if stored_mask:
                            composition = np.asarray(stored, dtype=np.float32)
                            if (
                                composition.shape != (NUM_ATOMIC_SKILLS,)
                                or not np.all(np.isfinite(composition))
                                or np.any(composition < 0.0)
                                or not np.isclose(float(composition.sum()), 1.0, atol=1e-5)
                                or np.count_nonzero(composition) > 5
                            ):
                                raise ValueError(
                                    f"{path}: segment {block} has an invalid stored Top-5 target"
                                )
                            confidence = float(
                                segment.get("atomic_composition_confidence", 1.0)
                            )
                            by_composition[block] = _AtomicCompositionAnnotation(
                                composition,
                                confidence,
                            )
                        elif mode == "dual":
                            values = segment.get("gate_probabilities")
                            if values is None:
                                raise ValueError(
                                    f"{path}: dual segment {block} is missing gate probabilities"
                                )
                            composition, confidence = _topk_gate_composition(
                                np.asarray(values, dtype=np.float32), top_k=2
                            )
                            by_composition[block] = _AtomicCompositionAnnotation(
                                composition,
                                confidence,
                            )
                    if mode == "idle":
                        if block in by_anchor:
                            raise ValueError(
                                f"motion/stay conflict for {arm} episode {episode_index} "
                                f"block {block}"
                            )
                        weights = np.zeros(NUM_ATOMIC_SKILLS, dtype=np.float32)
                        weights[STAY_ATOMIC_ID] = 1.0
                        by_anchor[block] = _AtomicHorizonAnnotation(
                            stay_prompts[arm], weights
                        )

    def global_prompt(self, episode_index: int, fallback: str) -> str:
        return self.global_prompts.get(episode_index, fallback)

    def subtask_prompt(self, episode_index: int, start_frame: int, end_frame: int) -> str:
        texts = [
            text
            for start, end, text in self.subtasks.get(episode_index, ())
            if start < end_frame and end > start_frame
        ]
        return compose_horizon_subtasks(texts)

    def subtask_at(self, episode_index: int, frame_index: int) -> tuple[str, int] | None:
        """Return the single reviewed subtask active at ``frame_index`` and its end."""

        for start, end, text in self.subtasks.get(episode_index, ()):
            if start <= frame_index < end:
                return text, end
        return None

    def atomic_horizon(
        self,
        episode_index: int,
        frame_index: int,
        *,
        arm: str = "right",
        source_fps: int = 30,
        anchor_fps: int = 3,
    ) -> _AtomicHorizonAnnotation | None:
        frames_per_anchor, remainder = divmod(source_fps, anchor_fps)
        if remainder:
            raise ValueError(f"source fps {source_fps} is not divisible by anchor fps {anchor_fps}")
        # A 3 Hz decision covers its complete one-third-second interval rather
        # than supervising only one of the ten contained 30 Hz start frames.
        return (
            self.atomic_horizons.get(episode_index, {})
            .get(arm, {})
            .get(frame_index // frames_per_anchor)
        )

    def atomic_composition(
        self,
        episode_index: int,
        frame_index: int,
        *,
        arm: str = "right",
        source_fps: int = 30,
        anchor_fps: int = 3,
    ) -> _AtomicCompositionAnnotation | None:
        """Return FK's pre-aggregated five-block soft composition target."""

        frames_per_anchor, remainder = divmod(source_fps, anchor_fps)
        if remainder:
            raise ValueError(f"source fps {source_fps} is not divisible by anchor fps {anchor_fps}")
        return (
            self.atomic_compositions.get(episode_index, {})
            .get(arm, {})
            .get(frame_index // frames_per_anchor)
        )


class _ZTTeacherSidecar:
    """Memory-mapped frozen zT Q1/Q3 directions aligned to LeRobot rows."""

    VERSION = 1

    def __init__(self, root: str | Path, expected_rows: int):
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing zT teacher manifest: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(self.manifest.get("version", -1)) != self.VERSION:
            raise ValueError(f"unsupported zT teacher sidecar version: {self.manifest}")
        if self.manifest.get("prompt") != "atomic":
            raise ValueError("zT teacher sidecar must use the atomic prompt")
        self.directions = np.load(self.root / "directions.npy", mmap_mode="r")
        self.valid = np.load(self.root / "valid.npy", mmap_mode="r")
        self.codebook = np.load(self.root / "codebook.npy", mmap_mode="r")
        latent_dim = int(self.manifest["latent_dim"])
        expected_directions = (expected_rows, 2, latent_dim)
        if self.directions.shape != expected_directions or self.directions.dtype != np.float16:
            raise ValueError(
                f"invalid teacher directions: expected float16 {expected_directions}, "
                f"got {self.directions.dtype} {self.directions.shape}"
            )
        if self.valid.shape != (expected_rows,) or self.valid.dtype != np.bool_:
            raise ValueError("teacher valid mask must be bool [dataset_rows]")
        if self.codebook.shape != (2, NUM_ATOMIC_SKILLS, latent_dim):
            raise ValueError("teacher codebook must have shape [2,13,latent_dim]")
        if self.codebook.dtype != np.float32:
            raise ValueError("teacher codebook must be float32")
        code_norms = np.linalg.norm(np.asarray(self.codebook), axis=-1)
        if not np.all(np.isfinite(code_norms)) or not np.allclose(
            code_norms, 1.0, atol=2e-3
        ):
            raise ValueError("teacher codebook must contain finite unit directions")

    def lookup(self, data_index: int) -> tuple[np.ndarray, bool]:
        return (
            np.asarray(self.directions[data_index], dtype=np.float32),
            bool(self.valid[data_index]),
        )


def _has_target_annotation_sidecars(root: Path) -> bool:
    return all(
        (root / "meta" / name).is_file()
        for name in (
            "global_episode_prompts.jsonl",
            "episode_subtasks.jsonl",
            "atomic_horizon_prompts_3hz.jsonl",
        )
    )


def _arrow_numeric_array(column: pa.ChunkedArray, dtype: np.dtype[Any]) -> np.ndarray:
    """Convert numeric Arrow columns without the multi-GB ``to_pylist`` peak."""

    array = column.combine_chunks()
    if pa.types.is_fixed_size_list(array.type):
        values = array.values.to_numpy(zero_copy_only=False)
        return np.array(values, dtype=dtype, copy=True).reshape(len(array), array.type.list_size)
    return np.array(array.to_numpy(zero_copy_only=False), dtype=dtype, copy=True)


class _LeRobotV21CompatDataset(LeRobotV3Dataset):
    """Read target's immutable LeRobot v2.1 layout through the v3 contract.

    The frame parquet and per-episode video layout already match the minimal
    OpenPI reader. Only episode metadata differs: v2.1 stores one JSONL row
    per episode, while v3 stores parquet rows with explicit data/video
    locations. This adapter builds those small rows in memory and never
    rewrites the dataset.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        image_transforms: Any = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        video_backend: str | None = None,
        episode_success_filter: Sequence[str] | None = None,
    ):
        torch.utils.data.Dataset.__init__(self)
        self.root = Path(root)
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.tolerance_s = tolerance_s
        self.video_backend = video_backend or _lerobot_v3.get_safe_default_codec()
        self.episode_success_filter = (
            tuple(value.lower() for value in episode_success_filter)
            if episode_success_filter
            else None
        )
        # The pinned OpenPI reader added these filters after this v2.1
        # compatibility adapter was introduced.  The target dataset does not
        # request them, but its inherited index builder still reads the
        # attributes.
        self.included_episode_indices = None
        self.excluded_episode_indices = None
        self.use_frame_index_timestamps = False

        self.info = json.loads((self.root / "meta" / "info.json").read_text())
        self.features = self.info["features"]
        self.fps = int(self.info["fps"])
        self.video_keys = _lerobot_v3._video_keys(self.features)

        tasks_table = pq.read_table(self.root / "meta" / "tasks.parquet")
        task_text_key = next(key for key in tasks_table.column_names if key != "task_index")
        self.tasks = {
            int(row["task_index"]): row[task_text_key]
            for row in tasks_table.to_pylist()
        }

        data_files = sorted((self.root / "data").glob("chunk-*/*.parquet"))
        data_table = _lerobot_v3._read_tables(data_files)
        self._states = _arrow_numeric_array(data_table["observation.state"], np.dtype(np.float32))
        self._timestamps = _arrow_numeric_array(data_table["timestamp"], np.dtype(np.float32))
        self._frame_index = _arrow_numeric_array(data_table["frame_index"], np.dtype(np.int64))
        self._episode_index = _arrow_numeric_array(data_table["episode_index"], np.dtype(np.int64))
        self._index = _arrow_numeric_array(data_table["index"], np.dtype(np.int64))
        self._task_index = _arrow_numeric_array(data_table["task_index"], np.dtype(np.int64))
        self._actions = (
            _arrow_numeric_array(data_table["action"], np.dtype(np.float32))
            if "action" in data_table.column_names
            else None
        )

        changes = np.flatnonzero(np.diff(self._episode_index)) + 1
        starts = np.r_[0, changes]
        ends = np.r_[changes, len(self._episode_index)]
        bounds = {
            int(self._episode_index[start]): (int(start), int(end))
            for start, end in zip(starts, ends, strict=True)
        }
        episode_rows: dict[int, dict[str, Any]] = {}
        with (self.root / "meta" / "episodes.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                episode_index = int(row["episode_index"])
                start, end = bounds[episode_index]
                converted = dict(row)
                converted["dataset_from_index"] = start
                converted["dataset_to_index"] = end
                for video_key in self.video_keys:
                    converted[f"videos/{video_key}/chunk_index"] = 0
                    converted[f"videos/{video_key}/file_index"] = episode_index
                episode_rows[episode_index] = converted
        if set(episode_rows) != set(bounds):
            raise ValueError("episodes.jsonl and frame parquet contain different episode ids")
        self.episode_rows = episode_rows
        self.episode_indices = sorted(episode_rows)
        self.episode_data_index = self._build_episode_data_index()
        # Newer OpenPI dataset readers expose helpers for episode filtering and
        # shared video-file accounting.  admin123's PI0.5 checkout predates
        # those helpers.  This target adapter never requests episode filters,
        # and its v2.1 layout stores one video file per episode, so the
        # equivalent compatibility values are deterministic.
        if hasattr(self, "_build_video_file_episode_counts"):
            self._video_file_episode_counts = self._build_video_file_episode_counts()
        else:
            self._video_file_episode_counts = {
                video_key: {
                    (
                        int(row[f"videos/{video_key}/chunk_index"]),
                        int(row[f"videos/{video_key}/file_index"]),
                    ): 1
                    for row in self.episode_rows.values()
                }
                for video_key in self.video_keys
            }
        if hasattr(self, "_build_visible_indices"):
            self._visible_indices = self._build_visible_indices()
        else:
            self._visible_indices = np.arange(len(self._episode_index), dtype=np.int64)

        self._extra_numeric = {}
        excluded = {
            "observation.state", "action", "timestamp", "frame_index",
            "episode_index", "index", "task_index",
        }
        for key in data_table.column_names:
            if key in excluded:
                continue
            column = data_table[key]
            if not (pa.types.is_string(column.type) or pa.types.is_large_string(column.type)):
                self._extra_numeric[key] = _arrow_numeric_array(column, np.dtype(column.type.to_pandas_dtype()))

        _lerobot_v3.check_timestamps_sync(
            self._timestamps,
            self._episode_index,
            {key: value.numpy() for key, value in self.episode_data_index.items()},
            self.fps,
            self.tolerance_s,
        )
        self.delta_indices = None
        if self.delta_timestamps is not None:
            _lerobot_v3.check_delta_timestamps(
                self.delta_timestamps, self.fps, self.tolerance_s
            )
            self.delta_indices = _lerobot_v3.get_delta_indices(
                self.delta_timestamps, self.fps
            )


def _read_annotation_columns(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    paths = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet files under {root / 'data'}")
    required = {"annotation.segment_id", "annotation.type", "annotation.prompt", "annotation.atoms"}
    for path in paths:
        missing = required - set(pq.read_schema(path).names)
        if missing:
            raise ValueError(f"{path} lacks atomic annotation fields: {sorted(missing)}")
    table = pa.concat_tables(
        [pq.read_table(path, columns=sorted(required)) for path in paths], promote_options="default"
    )
    return (
        np.asarray(table["annotation.segment_id"].to_pylist(), dtype=np.int64),
        np.asarray(table["annotation.type"].to_pylist(), dtype=np.int64),
        np.asarray(table["annotation.atoms"].to_pylist(), dtype=np.float32),
        [str(value) for value in table["annotation.prompt"].to_pylist()],
    )


class _AtomicRawDataset(Dataset[dict[str, Any]]):
    """Wrap one v3 dataset and provide sidecar metadata without re-decoding video."""

    def __init__(
        self,
        root: Path,
        action_horizon: int,
        zt_teacher_sidecar: str | Path | None = None,
        atomic_composition_sidecar: str | Path | None = None,
        pad_subtask_horizon: bool = False,
    ):
        self.root = root
        self.action_horizon = action_horizon
        self._base: LeRobotV3Dataset | None = self._open_base()
        self._uses_target_sidecars = _has_target_annotation_sidecars(root)
        self._atomic_composition_sidecar = atomic_composition_sidecar
        self._pad_subtask_horizon = bool(pad_subtask_horizon)
        self._target_sidecars: _TargetAnnotationSidecars | None = None
        if self._uses_target_sidecars:
            self._segment_id = self._type = self._atoms = self._prompts = None
            self._target_sidecars = _TargetAnnotationSidecars(
                root, atomic_composition_sidecar
            )
        else:
            self._segment_id, self._type, self._atoms, self._prompts = _read_annotation_columns(root)
        self._tcp_pose: np.ndarray | None = None
        self._zt_teacher_path = (
            None if zt_teacher_sidecar is None else Path(zt_teacher_sidecar)
        )
        self._zt_teacher: _ZTTeacherSidecar | None = None
        if self._type is not None and len(self._type) != len(self.base._states):
            raise ValueError(f"annotation row count does not match frames for {root}")
        if self.base._actions is None:
            raise ValueError(f"{root} has no action column")
        self._sample_indices = self._load_start_prefix_mask()

    def _load_start_prefix_mask(self) -> np.ndarray | None:
        """Return base-view indices after applying an optional start-prefix mask.

        The mask is deliberately row-selection metadata: source parquet, video,
        semantic segments, and 3 Hz atomic annotations remain byte-identical.
        """

        path = self.root / "meta" / "start_inactive_arm_block_mask.json"
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("atomic_block_frames", 0)) != 10:
            raise ValueError(f"{path}: atomic_block_frames must equal 10")
        cutoffs: dict[int, int] = {}
        for row in payload.get("records", []):
            episode = int(row["episode_index"])
            cutoff = int(row["mask_end_frame_exclusive"])
            if cutoff <= 0 or cutoff % 10:
                raise ValueError(f"{path}: episode {episode} cutoff {cutoff} is not block-aligned")
            cutoffs[episode] = max(cutoffs.get(episode, 0), cutoff)
        visible = getattr(self.base, "_visible_indices", None)
        data_indices = (
            np.arange(len(self.base), dtype=np.int64)
            if visible is None
            else np.asarray(visible, dtype=np.int64)
        )
        keep = np.fromiter(
            (
                int(self.base._frame_index[data_index])
                >= cutoffs.get(int(self.base._episode_index[data_index]), 0)
                for data_index in data_indices
            ),
            dtype=np.bool_,
            count=len(data_indices),
        )
        return np.flatnonzero(keep).astype(np.int64, copy=False)

    def _open_base(self) -> LeRobotV3Dataset:
        info = json.loads((self.root / "meta" / "info.json").read_text())
        dataset_type = (
            _LeRobotV21CompatDataset
            if str(info.get("codebase_version", "")).startswith("v2")
            else LeRobotV3Dataset
        )
        return dataset_type(
            self.root,
            delta_timestamps={"action": [t / int(info["fps"]) for t in range(self.action_horizon)]},
        )

    @property
    def base(self) -> LeRobotV3Dataset:
        if self._base is None:
            self._base = self._open_base()
        return self._base

    def _ensure_annotations(self) -> None:
        if self._uses_target_sidecars and self._target_sidecars is None:
            self._target_sidecars = _TargetAnnotationSidecars(
                self.root, self._atomic_composition_sidecar
            )
        elif not self._uses_target_sidecars and self._segment_id is None:
            self._segment_id, self._type, self._atoms, self._prompts = _read_annotation_columns(self.root)

    def _ensure_zt_teacher(self) -> None:
        if self._zt_teacher_path is not None and self._zt_teacher is None:
            self._zt_teacher = _ZTTeacherSidecar(
                self._zt_teacher_path, len(self.base._states)
            )

    def __getstate__(self) -> dict[str, Any]:
        """Do not pickle LeRobot arrays/sidecars into every spawn worker.

        OpenPI's stock loader uses spawn workers.  The atomic sidecar is much
        larger than a usual task string, so each worker reopens its own
        memory-mapped LeRobot arrays and reads annotation parquet lazily.
        """

        state = self.__dict__.copy()
        state["_base"] = None
        state["_segment_id"] = None
        state["_type"] = None
        state["_atoms"] = None
        state["_prompts"] = None
        state["_target_sidecars"] = None
        state["_tcp_pose"] = None
        state["_zt_teacher"] = None
        return state

    @property
    def tcp_pose(self) -> np.ndarray:
        """Memory-map precomputed state/action TCP poses, never run FK in workers."""

        if self._tcp_pose is None:
            bimanual_path = self.root / "meta" / _BIMANUAL_TCP_POSE_SIDECAR
            if self._uses_target_sidecars:
                # Target's reviewed atomic horizons were produced with the
                # coordinated tcp200 contract. Never silently mix them with
                # the legacy right-only tcp100 FK targets.
                path = bimanual_path
            else:
                path = (
                    bimanual_path
                    if bimanual_path.is_file()
                    else self.root / "meta" / _TCP_POSE_SIDECAR
                )
            if not path.is_file():
                raise FileNotFoundError(
                    f"missing {path}; run scripts/precompute_lerobot_tcp_pose.py "
                    f"--dataset-root {self.root} before training"
                )
            if path == bimanual_path:
                metadata_path = self.root / "meta" / BIMANUAL_TCP_POSE_METADATA
                if not metadata_path.is_file():
                    raise FileNotFoundError(f"missing TCP contract metadata: {metadata_path}")
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                expected_offset = [0.0, 0.0, BIMANUAL_TCP_LOCAL_Z_OFFSET_M]
                if metadata.get("tcp_local_offset_m") != expected_offset:
                    raise ValueError(
                        f"{metadata_path}: expected tcp200 offset {expected_offset}, "
                        f"got {metadata.get('tcp_local_offset_m')}"
                    )
            poses = np.load(path, mmap_mode="r")
            expected_columns = 48 if path == bimanual_path else 24
            expected = (len(self.base._states), expected_columns)
            if poses.shape != expected or poses.dtype != np.float32:
                raise ValueError(
                    f"invalid TCP pose sidecar {path}: expected float32 {expected}, "
                    f"got {poses.dtype} {poses.shape}"
                )
            self._tcp_pose = poses
        return self._tcp_pose

    def _tcp_delta(self, data_index: int, action_indices: np.ndarray) -> np.ndarray:
        """Build one coordinated [H,12] target: right 6D followed by left 6D."""

        poses = self.tcp_pose
        if poses.shape[1] == 24:
            arm_layouts = ((0, 12),)
        else:
            # File layout left then right; model/target order right then left.
            arm_layouts = ((24, 36), (0, 12))
        deltas: list[np.ndarray] = []
        for state_offset, action_offset in arm_layouts:
            start_row = poses[data_index, state_offset : state_offset + 12]
            start = CartesianPose(start_row[:3], start_row[3:].reshape(3, 3))
            future_rows = poses[action_indices, action_offset : action_offset + 12]
            deltas.append(
                np.stack(
                    [
                        base_frame_pose_delta(
                            start, CartesianPose(row[:3], row[3:].reshape(3, 3))
                        )
                        for row in future_rows
                    ],
                    axis=0,
                ).astype(np.float32, copy=False)
            )
        if len(deltas) == 1:
            # Backward-compatible single-right-arm corpora have no left FK.
            deltas.append(np.zeros_like(deltas[0]))
        return np.concatenate(deltas, axis=-1)

    def __len__(self) -> int:
        return len(self.base) if self._sample_indices is None else len(self._sample_indices)

    def _view_index(self, index: int) -> int:
        return int(index) if self._sample_indices is None else int(self._sample_indices[int(index)])

    def _data_index(self, index: int) -> int:
        """Map a dataset-view row to the underlying parquet row.

        The v2.1 compatibility reader may expose a filtered view through
        ``_visible_indices``.  The native v3 reader is always an all-row view
        and therefore intentionally has no such private field.
        """

        view_index = self._view_index(index)
        visible_indices = getattr(self.base, "_visible_indices", None)
        return view_index if visible_indices is None else int(visible_indices[view_index])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.base[self._view_index(index)]

    def metadata(self, index: int) -> dict[str, Any]:
        self._ensure_annotations()
        data_index = self._data_index(index)
        episode_index = int(self.base._episode_index[data_index])
        query_indices, _ = self.base._get_query_indices(data_index, episode_index)
        action_indices = np.asarray(query_indices["action"], dtype=np.int64)
        # Atomic InfoNCE/codebook keep their strict 50-step, one-segment contract,
        # but Q1's dual ratio now follows local 1/3-second FK evidence inside it.
        # FK twist reconstruction below is independent and exists for every row.
        task_index = int(self.base._task_index[data_index])
        global_prompt = str(self.base.tasks[task_index])
        frame_index = int(self.base._frame_index[data_index])
        end_frame = int(self.base._frame_index[action_indices[-1]]) + 1
        subtask_prompt = ""
        subtask_boundary_padded = False
        if self._uses_target_sidecars:
            assert self._target_sidecars is not None
            global_prompt = self._target_sidecars.global_prompt(episode_index, global_prompt)
            active_subtask = self._target_sidecars.subtask_at(episode_index, frame_index)
            if self._pad_subtask_horizon and active_subtask is not None:
                subtask_prompt, subtask_end_frame = active_subtask
                action_frames = np.asarray(
                    self.base._frame_index[action_indices], dtype=np.int64
                )
                inside = action_frames < subtask_end_frame
                if not np.all(inside):
                    valid_positions = np.flatnonzero(inside)
                    if not len(valid_positions):
                        raise ValueError(
                            f"subtask at episode={episode_index}, frame={frame_index} "
                            "contains no action in its own interval"
                        )
                    hold_index = action_indices[int(valid_positions[-1])]
                    action_indices = np.where(inside, action_indices, hold_index)
                    end_frame = subtask_end_frame
                    subtask_boundary_padded = True
            else:
                subtask_prompt = self._target_sidecars.subtask_prompt(
                    episode_index, frame_index, end_frame
                )
            atomic_by_arm = [
                self._target_sidecars.atomic_horizon(
                    episode_index, frame_index, arm=arm
                )
                for arm in ("right", "left")
            ]
            composition_by_arm = [
                self._target_sidecars.atomic_composition(
                    episode_index, frame_index, arm=arm
                )
                for arm in ("right", "left")
            ]
            supervised = np.asarray(
                [atomic is not None for atomic in atomic_by_arm], dtype=np.bool_
            )
            horizon_atoms = np.stack(
                [
                    atomic.weights
                    if atomic is not None
                    else np.zeros(NUM_ATOMIC_SKILLS, dtype=np.float32)
                    for atomic in atomic_by_arm
                ]
            )
            present = [
                (arm, atomic.prompt)
                for arm, atomic in zip(("Right arm", "Left arm"), atomic_by_arm, strict=True)
                if atomic is not None
            ]
            if present:
                atomic_prompt = compose_arm_atomic_prompts(present)
            else:
                atomic_prompt = global_prompt
            composition_weights = np.stack(
                [
                    value.weights
                    if value is not None
                    else np.zeros(NUM_ATOMIC_SKILLS, dtype=np.float32)
                    for value in composition_by_arm
                ]
            )
            composition_confidence = np.asarray(
                [value.confidence if value is not None else 0.0 for value in composition_by_arm],
                dtype=np.float32,
            )
            composition_mask = np.asarray(
                [value is not None for value in composition_by_arm], dtype=np.bool_
            )
            if subtask_boundary_padded:
                # The materialized Top-5 target describes the original
                # cross-boundary 50-step horizon. It is invalid after the
                # action tail is replaced by a hold target.
                composition_weights.fill(0.0)
                composition_confidence.fill(0.0)
                composition_mask.fill(False)
        else:
            row_type = self._type[action_indices]
            row_segment = self._segment_id[action_indices]
            row_atoms = self._atoms[action_indices]
            supervised, horizon_atoms = aggregate_horizon_atomic_weights(
                row_atoms, row_type, row_segment
            )
            atomic_prompt = (
                compose_arm_atomic_prompts((("Right arm", self._prompts[data_index]),))
                if supervised
                else global_prompt
            )
            horizon_atoms = np.stack(
                [horizon_atoms, np.zeros(NUM_ATOMIC_SKILLS, dtype=np.float32)]
            )
            supervised = np.asarray([supervised, False], dtype=np.bool_)
            # Legacy datasets have no complete-horizon FK probability
            # sidecar. Do not fabricate a soft target from their strict
            # segment label; the existing single/dual loss already owns it.
            composition_weights = np.zeros((2, NUM_ATOMIC_SKILLS), dtype=np.float32)
            composition_confidence = np.zeros(2, dtype=np.float32)
            composition_mask = np.zeros(2, dtype=np.bool_)
        raw_state = self.base._states[data_index].astype(np.float32, copy=False)
        raw_actions = self.base._actions[action_indices].astype(np.float32, copy=False)
        result = {
            "atomic_prompt": atomic_prompt,
            "global_prompt": global_prompt,
            "subtask_prompt": subtask_prompt or global_prompt,
            "subtask_supervision_mask": bool(subtask_prompt),
            "atomic_weights": horizon_atoms,
            "atomic_supervision_mask": supervised,
            "atomic_composition_weights": composition_weights,
            "atomic_composition_confidence": composition_confidence,
            "atomic_composition_mask": composition_mask,
            # This target exists for every row, including drop/interaction
            # and horizons crossing multiple atomic segments.
            "tcp_twist_delta": self._tcp_delta(data_index, action_indices),
            "raw_state": raw_state,
            "raw_actions": raw_actions,
            "data_index": data_index,
            "episode_index": episode_index,
            "frame_index": frame_index,
            "subtask_boundary_padded": subtask_boundary_padded,
        }
        if self._zt_teacher_path is not None:
            self._ensure_zt_teacher()
            assert self._zt_teacher is not None
            directions, valid = self._zt_teacher.lookup(data_index)
            if bool(np.any(supervised)) and not valid:
                raise ValueError(
                    f"missing zT teacher for strict row {data_index} "
                    f"(episode={episode_index}, frame={frame_index})"
                )
            result["zt_teacher_directions"] = directions
            result["zt_teacher_valid"] = valid
        return result

    def reliable_atomic_indices(self) -> np.ndarray:
        """Return visible rows with at least one reviewed arm-level atom.

        Stage-A zT defines the atomic space and must not spend updates on
        drop/interaction/cross-boundary rows.  Target sidecars make this
        selection without decoding video or running FK in DataLoader workers.
        """

        self._ensure_annotations()
        if self._uses_target_sidecars:
            assert self._target_sidecars is not None
            valid_blocks = {
                (
                    int(episode_index),
                    int(block),
                )
                for episode_index, by_arm in self._target_sidecars.atomic_horizons.items()
                for arm in ("right", "left")
                for block in by_arm.get(arm, {})
            }
            visible_indices = getattr(self.base, "_visible_indices", None)
            all_visible = (
                np.arange(len(self.base), dtype=np.int64)
                if visible_indices is None
                else np.asarray(visible_indices, dtype=np.int64)
            )
            visible = (
                all_visible
                if self._sample_indices is None
                else all_visible[self._sample_indices]
            )
            episodes = self.base._episode_index[visible]
            blocks = self.base._frame_index[visible] // 10
            return np.fromiter(
                (
                    index
                    for index, pair in enumerate(zip(episodes, blocks, strict=True))
                    if (int(pair[0]), int(pair[1])) in valid_blocks
                ),
                dtype=np.int64,
            )

        # Legacy annotated datasets are not used by the production target
        # run, but retain the same semantic contract for focused tests/tools.
        return np.fromiter(
            (
                index
                for index in range(len(self))
                if bool(np.any(self.metadata(index)["atomic_supervision_mask"]))
            ),
            dtype=np.int64,
        )


class _AtomicProcessedDataset(Dataset[dict[str, Any]]):
    """Attach atomic prompt tokens and raw coefficient inputs after π0.5 transforms."""

    def __init__(
        self,
        raw: _AtomicRawDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        max_token_len: int,
        *,
        include_fast: bool,
        subtask_max_token_len: int,
    ):
        # The task prompt must be added before RepackTransform removes task_index.
        prompted = TransformedDataset(raw, [_transforms.PromptFromLeRobotTask(raw.base.tasks)])
        self._core = TransformedDataset(prompted, transforms)
        self._raw = raw
        self._max_token_len = max_token_len
        self._include_fast = include_fast
        self._subtask_max_token_len = subtask_max_token_len

    def __len__(self) -> int:
        return len(self._raw)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self._core[index])
        # The stock prompt is only an intermediate needed by the common
        # transforms. Atomic prompts below are authoritative, so do not leak
        # the unused string into the collated training sample.
        item.pop("prompt", None)
        metadata = self._raw.metadata(int(index))
        state = np.asarray(item["state"], dtype=np.float32)
        # Match the released PI0.5 Marvin contract exactly: MarvinInputs pads
        # the 16 physical CR1 coordinates to the 32-D model interface before
        # normalization, and TokenizePrompt serializes that complete normalized
        # 32-D state (including the 16 zero padding coordinates).
        atomic_tokens, atomic_mask = _paligemma_tokenizer(self._max_token_len).tokenize(
            metadata["atomic_prompt"], state
        )
        global_tokens, global_mask = _paligemma_tokenizer(self._max_token_len).tokenize(
            metadata["global_prompt"], state
        )
        subtask_tokens, subtask_mask = _paligemma_tokenizer(self._max_token_len).tokenize(
            metadata["subtask_prompt"], state
        )
        subtask_target_tokens, subtask_target_mask = _tokenize_subtask_target(
            metadata["subtask_prompt"], self._subtask_max_token_len
        )
        subtask_target_mask &= bool(metadata["subtask_supervision_mask"])
        atomic_target_tokens, atomic_target_mask = _tokenize_subtask_target(
            metadata["atomic_prompt"], self._subtask_max_token_len
        )
        # The target string contains exactly the arms present in
        # ``atomic_prompt``. A one-arm horizon therefore remains a valid AR
        # target for that arm; only completely unlabeled horizons are masked.
        atomic_target_mask &= bool(np.any(metadata["atomic_supervision_mask"]))
        result = {
            **item,
            **metadata,
            # Target's per-episode global prompt is authoritative over the
            # coarser shared LeRobot task table used by the stock transform.
            "tokenized_prompt": global_tokens,
            "tokenized_prompt_mask": global_mask,
            "atomic_prompt_tokens": atomic_tokens,
            "atomic_prompt_mask": atomic_mask,
            "subtask_prompt_tokens": subtask_tokens,
            "subtask_prompt_mask": subtask_mask,
            "subtask_target_tokens": subtask_target_tokens,
            "subtask_target_mask": subtask_target_mask,
            "atomic_target_tokens": atomic_target_tokens,
            "atomic_target_mask": atomic_target_mask,
        }
        if self._include_fast:
            # Kept for the later FAST auxiliary/joint stage. Stage-A Q1/DCT
            # deliberately avoids both these targets and their tokenizer cost.
            fast_tokens, fast_mask, fast_ar_mask, fast_loss_mask = _fast_tokenizer(
                self._max_token_len
            ).tokenize(
                metadata["atomic_prompt"],
                np.asarray(item["state"], dtype=np.float32),
                np.asarray(item["actions"], dtype=np.float32),
            )
            result |= {
                "fast_tokens": fast_tokens,
                "fast_mask": fast_mask,
                "fast_ar_mask": fast_ar_mask,
                "fast_loss_mask": fast_loss_mask,
            }
        return result


class _AtomicTextDataset(Dataset[dict[str, Any]]):
    """Text/state-only Stage-A view; it never calls ``LeRobotV3Dataset.__getitem__``.

    The regular loader must decode three video frames before applying CR1
    transforms.  z_T does not consume those images, so Stage A directly reads
    LeRobot's already-memory-mapped state/action arrays, then applies the same
    Marvin ``adapt_to_pi=True`` conversion, delta-action, and normalization
    order used by admin123's π0.5 checkout.  This is not a different data
    contract: it merely skips unused image decoding.
    """

    def __init__(
        self,
        raw: _AtomicRawDataset,
        *,
        norm_stats: Any,
        use_quantiles: bool,
        max_token_len: int,
        include_fast: bool,
    ):
        self._raw = raw
        self._normalize = _NormalizeWithoutQuantileClipping(
            norm_stats, use_quantiles=use_quantiles
        )
        self._delta_actions = _transforms.DeltaActions(_transforms.make_bool_mask(7, -1, 7, -1))
        self._max_token_len = max_token_len
        self._include_fast = include_fast

    def __len__(self) -> int:
        return len(self._raw)

    def __getitem__(self, index: int) -> dict[str, Any]:
        metadata = self._raw.metadata(int(index))
        # Match MarvinInputs(adapt_to_pi=True) exactly before the no-video
        # fast path pads, converts arm joints to deltas, and normalizes.  The
        # accepted-v3 norm assets were computed in this PI-internal frame.
        state, actions = _adapt_marvin_values_to_pi(
            metadata["raw_state"], metadata["raw_actions"]
        )
        state = _transforms.pad_to_dim(state, 32)
        actions = _transforms.pad_to_dim(actions, 32)
        transformed = self._delta_actions({"state": state, "actions": actions})
        transformed = self._normalize(transformed)
        state = np.asarray(transformed["state"], dtype=np.float32)
        actions = np.asarray(transformed["actions"], dtype=np.float32)
        # This matches the current z_T branch: Q1's language prompt is the
        # local atomic instruction on clean segments, otherwise the task text.
        prompt_tokens, prompt_mask = _paligemma_tokenizer(self._max_token_len).tokenize(
            metadata["atomic_prompt"], state
        )
        subtask_tokens, subtask_mask = _paligemma_tokenizer(self._max_token_len).tokenize(
            metadata["subtask_prompt"], state
        )
        result = {
            **metadata,
            "state": state,
            # Full normalized 50x32 PI0.5 target. Stage A can therefore train
            # the shared Action Expert without decoding any video.
            "actions": actions,
            # OpenPI-normalized relative arm actions in the same right/left
            # order used by the bimanual z_T fusion. Gripper coordinates and
            # checkpoint-padding dimensions are not coefficient targets.
            "joint_delta": np.concatenate(
                [actions[:, 8:15], actions[:, 0:7]], axis=-1
            ).astype(np.float32, copy=False),
            "atomic_prompt_tokens": prompt_tokens,
            "atomic_prompt_mask": prompt_mask,
            "subtask_prompt_tokens": subtask_tokens,
            "subtask_prompt_mask": subtask_mask,
        }
        if self._include_fast:
            fast_tokens, fast_mask, fast_ar_mask, fast_loss_mask = _fast_tokenizer(
                self._max_token_len
            ).tokenize(metadata["atomic_prompt"], state, actions)
            result |= {
                "fast_tokens": fast_tokens,
                "fast_mask": fast_mask,
                "fast_ar_mask": fast_ar_mask,
                "fast_loss_mask": fast_loss_mask,
            }
        return result


def _stack_tree(rows: list[dict[str, Any]], key: str) -> Any:
    first = rows[0][key]
    if isinstance(first, dict):
        return {name: _stack_tree([row[key] for row in rows], name) for name in first}
    stacked = np.stack([np.asarray(row[key]) for row in rows], axis=0)
    # q01/q99 NumPy arithmetic otherwise promotes state/action to float64,
    # while the π0.5 JAX contract is float32 before the bfloat16 backbone.
    return stacked.astype(np.float32) if np.issubdtype(stacked.dtype, np.floating) else stacked


def atomic_collate(rows: list[dict[str, Any]]) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
    if not rows:
        raise ValueError("cannot collate an empty atomic batch")
    sidecar_keys = {
        "atomic_prompt_tokens", "atomic_prompt_mask",
        "subtask_prompt_tokens", "subtask_prompt_mask",
        "subtask_target_tokens", "subtask_target_mask",
        "atomic_target_tokens", "atomic_target_mask",
        "fast_tokens", "fast_mask", "fast_ar_mask", "fast_loss_mask",
    }
    core_keys = [key for key in rows[0] if key not in _RAW_EXTRA_KEYS and key not in sidecar_keys]
    output: dict[str, Any] = {key: _stack_tree(rows, key) for key in core_keys}
    output["atomic_prompt_tokens"] = np.stack([row["atomic_prompt_tokens"] for row in rows], axis=0)
    output["atomic_prompt_mask"] = np.stack([row["atomic_prompt_mask"] for row in rows], axis=0)
    for key in (
        "subtask_prompt_tokens", "subtask_prompt_mask",
        "subtask_target_tokens", "subtask_target_mask",
        "atomic_target_tokens", "atomic_target_mask",
    ):
        output[key] = np.stack([row[key] for row in rows], axis=0)
    for key in ("fast_tokens", "fast_mask", "fast_ar_mask", "fast_loss_mask"):
        if key in rows[0]:
            output[key] = np.stack([row[key] for row in rows], axis=0)
    output["atomic_weights"] = np.stack([row["atomic_weights"] for row in rows], axis=0)
    output["atomic_supervision_mask"] = np.asarray(
        [row["atomic_supervision_mask"] for row in rows], dtype=np.bool_
    )
    output["atomic_composition_weights"] = np.stack(
        [row["atomic_composition_weights"] for row in rows], axis=0
    )
    output["atomic_composition_confidence"] = np.stack(
        [row["atomic_composition_confidence"] for row in rows], axis=0
    )
    output["atomic_composition_mask"] = np.stack(
        [row["atomic_composition_mask"] for row in rows], axis=0
    )
    output["tcp_twist_delta"] = np.stack([row["tcp_twist_delta"] for row in rows], axis=0)
    if "zt_teacher_directions" in rows[0]:
        output["zt_teacher_directions"] = np.stack(
            [row["zt_teacher_directions"] for row in rows], axis=0
        ).astype(np.float32, copy=False)
        output["zt_teacher_valid"] = np.asarray(
            [row["zt_teacher_valid"] for row in rows], dtype=np.bool_
        )
    return output


def _worker_init_fn(_: int) -> None:
    """Match OpenPI's spawn-worker policy and keep workers off GPU memory."""

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


def build_atomic_dataset(
    roots: Sequence[str | Path],
    *,
    norm_assets_dir: str | Path,
    norm_asset_id: str,
    action_horizon: int = 50,
    max_token_len: int = 200,
    include_fast: bool = True,
    subtask_max_token_len: int = 64,
    zt_teacher_sidecar: Sequence[str | Path] | str | Path | None = None,
    atomic_composition_sidecar: str | Path | None = None,
    pad_subtask_horizon: bool = False,
) -> Dataset[dict[str, Any]]:
    """Build the six-dataset atomic training view using the common OpenPI norm."""

    root_paths = tuple(Path(root) for root in roots)
    if not root_paths:
        raise ValueError("at least one dataset root is required")
    if zt_teacher_sidecar is None:
        teacher_paths: tuple[Path | None, ...] = (None,) * len(root_paths)
    elif isinstance(zt_teacher_sidecar, (str, Path)):
        if len(root_paths) != 1:
            raise ValueError("one frozen zT teacher sidecar requires one dataset root")
        teacher_paths = (Path(zt_teacher_sidecar),)
    else:
        teacher_paths = tuple(Path(path) for path in zt_teacher_sidecar)
        if len(teacher_paths) != len(root_paths):
            raise ValueError(
                "frozen zT teacher sidecars must match dataset roots one-to-one: "
                f"{len(teacher_paths)} != {len(root_paths)}"
            )
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
                        # MarvinInputs consumes native camera names and converts
                        # CHW LeRobot frames to π0.5's HWC image contract.
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
    # _AtomicProcessedDataset must tokenize three authoritative sidecar prompts
    # (global/subtask/atomic), so the stock single-prompt invocation would be
    # discarded immediately. Remove only that duplicate work; the replacement
    # calls above preserve its exact full-32-D PI0.5 state serialization.
    model_transforms = [
        transform
        for transform in data_config.model_transforms.inputs
        if not isinstance(transform, _transforms.TokenizePrompt)
    ]
    transforms = (
        [*data_config.repack_transforms.inputs]
        + [*data_config.data_transforms.inputs]
        + [
            _NormalizeWithoutQuantileClipping(
                data_config.norm_stats,
                use_quantiles=data_config.use_quantile_norm,
            )
        ]
        + model_transforms
    )
    datasets = [
        _AtomicProcessedDataset(
            _AtomicRawDataset(
                root,
                action_horizon,
                teacher_path,
                atomic_composition_sidecar,
                pad_subtask_horizon,
            ),
            transforms,
            max_token_len,
            include_fast=include_fast,
            subtask_max_token_len=subtask_max_token_len,
        )
        for root, teacher_path in zip(root_paths, teacher_paths, strict=True)
    ]
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def build_atomic_loader(
    roots: Sequence[str | Path],
    *,
    norm_assets_dir: str | Path,
    norm_asset_id: str,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 0,
    action_horizon: int = 50,
    max_token_len: int = 200,
    include_fast: bool = True,
    subtask_max_token_len: int = 64,
    zt_teacher_sidecar: Sequence[str | Path] | str | Path | None = None,
    atomic_composition_sidecar: str | Path | None = None,
    pad_subtask_horizon: bool = False,
) -> DataLoader:
    dataset = build_atomic_dataset(
        roots,
        norm_assets_dir=norm_assets_dir,
        norm_asset_id=norm_asset_id,
        action_horizon=action_horizon,
        max_token_len=max_token_len,
        include_fast=include_fast,
        subtask_max_token_len=subtask_max_token_len,
        zt_teacher_sidecar=zt_teacher_sidecar,
        atomic_composition_sidecar=atomic_composition_sidecar,
        pad_subtask_horizon=pad_subtask_horizon,
    )
    if include_fast:
        # FAST's HuggingFace processor is much heavier than PaliGemma's
        # sentence piece tokenizer. Build it before workers fork only when a
        # training phase actually consumes FAST action tokens.
        _fast_tokenizer(max_token_len)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        drop_last=True,
        collate_fn=atomic_collate,
        worker_init_fn=_worker_init_fn,
        generator=generator,
    )


def build_atomic_text_dataset(
    roots: Sequence[str | Path],
    *,
    norm_assets_dir: str | Path,
    norm_asset_id: str,
    atomic_composition_sidecar: str | Path | None = None,
    action_horizon: int = 50,
    max_token_len: int = 200,
    include_fast: bool = True,
    reliable_atomic_only: bool = False,
) -> Dataset[dict[str, Any]]:
    """Build the no-video Stage-A z_T/FAST dataset view."""

    root_paths = tuple(Path(root) for root in roots)
    if not root_paths:
        raise ValueError("at least one dataset root is required")
    # Ask the same admin123 Marvin factory for the canonical normalization tree.
    # Model transforms are intentionally not applied: this view creates both
    # a normal PaliGemma Q1 sequence and a second FAST sequence itself.
    bridge_config = pi0_config.Pi0Config(pi05=True, max_token_len=max_token_len)
    data_factory = _training_config.LeRobotMarvinDataConfig(
        repo_id=str(root_paths[0]),
        prompt_from_task=True,
        adapt_to_pi=True,
        assets=_training_config.AssetsConfig(
            assets_dir=str(norm_assets_dir), asset_id=norm_asset_id
        ),
    )
    data_config = data_factory.create(Path(norm_assets_dir), bridge_config)
    datasets = []
    for root in root_paths:
        raw = _AtomicRawDataset(
            root,
            action_horizon,
            atomic_composition_sidecar=atomic_composition_sidecar,
        )
        dataset: Dataset[dict[str, Any]] = _AtomicTextDataset(
            raw,
            norm_stats=data_config.norm_stats,
            use_quantiles=data_config.use_quantile_norm,
            max_token_len=max_token_len,
            include_fast=include_fast,
        )
        if reliable_atomic_only:
            indices = raw.reliable_atomic_indices()
            if not len(indices):
                raise ValueError(f"{root}: no reliable atomic zT rows")
            dataset = Subset(dataset, indices.tolist())
        datasets.append(dataset)
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def atomic_text_collate(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Collate the static numeric-only Stage-A batch."""

    if not rows:
        raise ValueError("cannot collate an empty atomic text batch")
    keys = [
        "state",
        "actions",
        "atomic_prompt_tokens",
        "atomic_prompt_mask",
        "atomic_weights",
        "atomic_supervision_mask",
        "atomic_composition_weights",
        "atomic_composition_confidence",
        "atomic_composition_mask",
        "tcp_twist_delta",
        "joint_delta",
    ]
    if "fast_tokens" in rows[0]:
        keys += ["fast_tokens", "fast_mask", "fast_ar_mask", "fast_loss_mask"]
    result: dict[str, np.ndarray] = {}
    for key in keys:
        value = np.stack([np.asarray(row[key]) for row in rows], axis=0)
        result[key] = value.astype(np.float32) if np.issubdtype(value.dtype, np.floating) else value
    return result


def build_atomic_text_loader(
    roots: Sequence[str | Path],
    *,
    norm_assets_dir: str | Path,
    norm_asset_id: str,
    atomic_composition_sidecar: str | Path | None = None,
    batch_size: int,
    num_workers: int = 4,
    seed: int = 0,
    action_horizon: int = 50,
    max_token_len: int = 200,
    include_fast: bool = True,
    reliable_atomic_only: bool = False,
) -> DataLoader:
    """Build Stage A with no video decode; workers only prepare token arrays."""

    dataset = build_atomic_text_dataset(
        roots,
        norm_assets_dir=norm_assets_dir,
        norm_asset_id=norm_asset_id,
        atomic_composition_sidecar=atomic_composition_sidecar,
        action_horizon=action_horizon,
        max_token_len=max_token_len,
        include_fast=include_fast,
        reliable_atomic_only=reliable_atomic_only,
    )
    if include_fast:
        _fast_tokenizer(max_token_len)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
        drop_last=True,
        collate_fn=atomic_text_collate,
        worker_init_fn=_worker_init_fn,
        generator=generator,
    )


def batch_to_observation(batch: dict[str, Any]) -> tuple[_model.Observation, np.ndarray]:
    """Convert collated core fields to OpenPI's normal observation contract."""

    core = {key: value for key, value in batch.items() if key not in _RAW_EXTRA_KEYS and key not in {
        "atomic_prompt_tokens", "atomic_prompt_mask",
        "subtask_prompt_tokens", "subtask_prompt_mask",
        "subtask_target_tokens", "subtask_target_mask",
        "atomic_target_tokens", "atomic_target_mask",
        "fast_tokens", "fast_mask", "fast_ar_mask", "fast_loss_mask",
        "atomic_weights", "atomic_supervision_mask",
        "atomic_composition_weights", "atomic_composition_confidence",
        "atomic_composition_mask", "tcp_twist_delta",
        "zt_teacher_directions", "zt_teacher_valid",
    }}
    actions = core.pop("actions")
    return _model.Observation.from_dict(core), actions


def text_batch_to_observation(batch: dict[str, np.ndarray]) -> _model.Observation:
    """Create the exact text/state observation consumed by ``compute_text_stage_loss``."""

    return _model.Observation(
        images={},
        image_masks={},
        state=batch["state"],
        tokenized_prompt=batch["atomic_prompt_tokens"],
        tokenized_prompt_mask=batch["atomic_prompt_mask"],
        token_ar_mask=None,
        token_loss_mask=None,
    )
