from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path

from lerobot.common.datasets.utils import check_delta_timestamps
from lerobot.common.datasets.utils import check_timestamps_sync
from lerobot.common.datasets.utils import get_delta_indices
from lerobot.common.datasets.video_utils import decode_video_frames
from lerobot.common.datasets.video_utils import get_safe_default_codec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _video_keys(features: dict[str, dict]) -> list[str]:
    return [key for key, ft in features.items() if ft["dtype"] == "video"]


def _read_tables(paths: list[Path]) -> pa.Table:
    if not paths:
        raise FileNotFoundError("No parquet files found.")
    tables = [pq.read_table(path) for path in paths]
    if len(tables) == 1:
        return tables[0]
    return pa.concat_tables(tables)


class LeRobotV3Dataset(torch.utils.data.Dataset):
    """Minimal local-reader for LeRobot v3 datasets.

    This keeps the sample contract of the v2.1 LeRobotDataset used by OpenPI:
    random access over frames, synchronous video decode, and delta-timestamp
    querying for action chunks.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        video_backend: str | None = None,
    ):
        super().__init__()
        self.root = Path(root)
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.tolerance_s = tolerance_s
        self.video_backend = video_backend if video_backend else get_safe_default_codec()

        info = _load_json(self.root / "meta" / "info.json")
        self.info = info
        self.features = info["features"]
        self.fps = int(info["fps"])
        self.video_keys = _video_keys(self.features)

        tasks_table = pq.read_table(self.root / "meta" / "tasks.parquet")
        task_rows = tasks_table.to_pylist()
        task_text_key = next(k for k in tasks_table.column_names if k != "task_index")
        self.tasks = {int(row["task_index"]): row[task_text_key] for row in task_rows}

        episode_files = sorted((self.root / "meta" / "episodes").glob("chunk-*/*.parquet"))
        if not episode_files:
            raise FileNotFoundError(f"No v3 episode metadata found under {self.root / 'meta' / 'episodes'}")
        episodes_table = _read_tables(episode_files)
        self.episode_rows = {int(row["episode_index"]): row for row in episodes_table.to_pylist()}
        self.episode_indices = sorted(self.episode_rows)
        self.episode_data_index = self._build_episode_data_index()

        data_files = sorted((self.root / "data").glob("chunk-*/*.parquet"))
        data_table = _read_tables(data_files)
        self._states = np.asarray(data_table["observation.state"].to_pylist(), dtype=np.float32)
        self._timestamps = np.asarray(data_table["timestamp"].to_pylist(), dtype=np.float32)
        self._frame_index = np.asarray(data_table["frame_index"].to_pylist(), dtype=np.int64)
        self._episode_index = np.asarray(data_table["episode_index"].to_pylist(), dtype=np.int64)
        self._index = np.asarray(data_table["index"].to_pylist(), dtype=np.int64)
        self._task_index = np.asarray(data_table["task_index"].to_pylist(), dtype=np.int64)

        self._actions = None
        if "action" in data_table.column_names:
            self._actions = np.asarray(data_table["action"].to_pylist(), dtype=np.float32)

        self._extra_numeric: dict[str, np.ndarray] = {}
        for key in data_table.column_names:
            if key in {
                "observation.state",
                "action",
                "timestamp",
                "frame_index",
                "episode_index",
                "index",
                "task_index",
            }:
                continue
            values = data_table[key].to_pylist()
            if values and not isinstance(values[0], str):
                self._extra_numeric[key] = np.asarray(values)

        timestamps = self._timestamps
        episode_indices = self._episode_index
        check_timestamps_sync(
            timestamps,
            episode_indices,
            {k: v.numpy() for k, v in self.episode_data_index.items()},
            self.fps,
            self.tolerance_s,
        )

        self.delta_indices = None
        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

    def _build_episode_data_index(self) -> dict[str, torch.Tensor]:
        starts = [int(self.episode_rows[ep_idx]["dataset_from_index"]) for ep_idx in self.episode_indices]
        ends = [int(self.episode_rows[ep_idx]["dataset_to_index"]) for ep_idx in self.episode_indices]
        return {
            "from": torch.LongTensor(starts),
            "to": torch.LongTensor(ends),
        }

    def __len__(self) -> int:
        return len(self._timestamps)

    def _get_episode_bounds(self, ep_idx: int) -> tuple[int, int]:
        row = self.episode_rows[ep_idx]
        return int(row["dataset_from_index"]), int(row["dataset_to_index"])

    def _get_query_indices(self, idx: int, ep_idx: int) -> tuple[dict[str, list[int]], dict[str, torch.Tensor]]:
        ep_start, ep_end = self._get_episode_bounds(ep_idx)
        query_indices = {
            key: [max(ep_start, min(ep_end - 1, idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {
            f"{key}_is_pad": torch.BoolTensor([(idx + delta < ep_start) or (idx + delta >= ep_end) for delta in delta_idx])
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        result = {}
        for key, q_idx in query_indices.items():
            if key in self.video_keys:
                continue
            if key == "action" and self._actions is not None:
                result[key] = torch.from_numpy(self._actions[q_idx])
            elif key == "observation.state":
                result[key] = torch.from_numpy(self._states[q_idx])
            elif key == "timestamp":
                result[key] = torch.from_numpy(self._timestamps[q_idx])
            elif key == "frame_index":
                result[key] = torch.from_numpy(self._frame_index[q_idx])
            elif key == "episode_index":
                result[key] = torch.from_numpy(self._episode_index[q_idx])
            elif key == "index":
                result[key] = torch.from_numpy(self._index[q_idx])
            elif key == "task_index":
                result[key] = torch.from_numpy(self._task_index[q_idx])
            elif key in self._extra_numeric:
                result[key] = torch.from_numpy(self._extra_numeric[key][q_idx])
        return result

    def _video_path(self, ep_idx: int, video_key: str) -> Path:
        row = self.episode_rows[ep_idx]
        chunk_index = int(row[f"videos/{video_key}/chunk_index"])
        file_index = int(row[f"videos/{video_key}/file_index"])
        return self.root / "videos" / video_key / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        for key in self.video_keys:
            if query_indices is not None and key in query_indices:
                query_timestamps[key] = self._timestamps[query_indices[key]].tolist()
            else:
                query_timestamps[key] = [current_ts]
        return query_timestamps

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            frames = decode_video_frames(self._video_path(ep_idx, vid_key), query_ts, self.tolerance_s, self.video_backend)
            item[vid_key] = frames.squeeze(0)
        return item

    def __getitem__(self, idx: int) -> dict:
        item = {
            "observation.state": torch.from_numpy(self._states[idx]),
            "timestamp": torch.tensor(self._timestamps[idx]),
            "frame_index": torch.tensor(self._frame_index[idx]),
            "episode_index": torch.tensor(self._episode_index[idx]),
            "index": torch.tensor(self._index[idx]),
            "task_index": torch.tensor(self._task_index[idx]),
        }
        if self._actions is not None:
            item["action"] = torch.from_numpy(self._actions[idx])
        for key, values in self._extra_numeric.items():
            item[key] = torch.from_numpy(np.asarray(values[idx]))

        ep_idx = int(item["episode_index"].item())

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if self.video_keys:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            item = {**self._query_videos(query_timestamps, ep_idx), **item}

        if self.image_transforms is not None:
            for cam in self.video_keys:
                item[cam] = self.image_transforms(item[cam])

        task_idx = int(item["task_index"].item())
        item["task"] = self.tasks[task_idx]
        return item
