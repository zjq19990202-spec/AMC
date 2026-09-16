#!/usr/bin/env python3
"""Send one real-HDF5-derived slow/fast pair to a running force server."""

from __future__ import annotations

import argparse
import time

import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12099)
    args = parser.parse_args()
    values = np.load(args.npz)
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    print("metadata", client.get_server_metadata(), flush=True)

    def payload(tag: str, kind: str, offset: int) -> dict:
        return {
            "state": values[f"{tag}_state"].astype(np.float32),
            "images": {
                "cam_high": values[f"{tag}_high"],
                "cam_left_wrist": values[f"{tag}_left"],
                "cam_right_wrist": values[f"{tag}_right"],
            },
            "force_state_history_120hz": values[f"{tag}_q120"].astype(np.float32),
            "left_force_history_120hz": values[f"{tag}_lf"].astype(np.float32),
            "right_force_history_120hz": values[f"{tag}_rf"].astype(np.float32),
            "force_history_mask": np.ones(120, dtype=np.bool_),
            "_force_request_kind": kind,
            "_force_cycle_id": 1,
            "_force_update_offset": offset,
            "_cr1_consume_prompt": kind == "slow_prepare",
        }

    slow = payload("slow", "slow_prepare", 0)
    start = time.perf_counter()
    slow_result = client.infer(slow)
    slow_actions = np.asarray(slow_result["actions"])
    print("slow", time.perf_counter() - start, slow_actions.shape, np.isfinite(slow_actions).all(), flush=True)

    fast = payload("fast", "fast_update", 25)
    fast["_force_executed_actions"] = values["executed"].astype(np.float32)
    start = time.perf_counter()
    fast_result = client.infer(fast)
    fast_actions = np.asarray(fast_result["actions"])
    prefix_error = np.max(np.abs(fast_actions[:25] - fast["_force_executed_actions"]))
    suffix_rmse = float(np.sqrt(np.mean(np.square(fast_actions[25:] - slow_actions[25:]))))
    print("fast", time.perf_counter() - start, fast_actions.shape, np.isfinite(fast_actions).all(),
          "prefix", prefix_error, "suffix_vs_slow_rmse", suffix_rmse, flush=True)
    assert slow_actions.shape == fast_actions.shape == (50, 16)
    assert np.isfinite(slow_actions).all() and np.isfinite(fast_actions).all()
    assert prefix_error == 0.0


if __name__ == "__main__":
    main()
