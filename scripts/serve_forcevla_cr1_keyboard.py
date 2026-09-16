#!/usr/bin/env python3
"""Serve the reproduced CR1 ForceVLA checkpoint with keyboard SUBtasks."""

from __future__ import annotations

import argparse
import json
import logging
import socket
from pathlib import Path
from typing import Any

import numpy as np

from serve_layerwise_pi05_keyboard import KeyboardPromptController, PromptBank
from serve_layerwise_pi05_keyboard import load_prompt_mapping, parse_prompt_assignment


LOGGER = logging.getLogger(__name__)
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


class ForceVlaKeyboardPolicy:
    """Adapt the CR1 force bridge payload to the reproduced ForceVLA policy."""

    def __init__(self, policy: Any, prompt_bank: PromptBank, metadata: dict[str, Any]) -> None:
        self._policy = policy
        self._prompt_bank = prompt_bank
        self.metadata = metadata

    @staticmethod
    def _latest_force(obs: dict[str, Any], key: str) -> np.ndarray:
        history = np.asarray(obs.get(key), dtype=np.float32)
        if history.ndim != 2 or history.shape[1] != 6 or history.shape[0] == 0:
            raise ValueError(f"{key} must have shape [N,6], got {history.shape}")
        mask = np.asarray(obs.get("force_history_mask", np.ones(history.shape[0])), dtype=bool)
        if mask.shape != (history.shape[0],):
            raise ValueError(f"force_history_mask must have shape ({history.shape[0]},)")
        valid = np.flatnonzero(mask & np.all(np.isfinite(history), axis=1))
        if valid.size == 0:
            raise ValueError(f"{key} contains no valid force sample")
        return history[valid[-1]]

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        if "_rserl_event" in obs:
            return {"ok": True, "event": str(obs["_rserl_event"])}

        state = np.asarray(obs.get("state"), dtype=np.float32).reshape(-1)
        if state.shape != (16,) or not np.all(np.isfinite(state)):
            raise ValueError(f"state must be finite shape (16,), got {state.shape}")
        source_images = obs.get("images")
        if not isinstance(source_images, dict) or "cam_high" not in source_images:
            raise ValueError("images must contain cam_high")
        images: dict[str, np.ndarray] = {}
        for name in CAMERAS:
            if name not in source_images:
                raise ValueError(f"images must contain {name}")
            image = np.asarray(source_images[name], dtype=np.uint8)
            if image.shape != (3, 224, 224):
                raise ValueError(f"{name} must have shape (3,224,224), got {image.shape}")
            images[name] = image

        key, prompt, revision, _ = self._prompt_bank.acquire(
            one_shot=False, consume_transition=False, wait_for_prompt=False
        )
        # create_trained_policy applies the data transforms directly at inference;
        # the LeRobot-only RepackTransform is not part of this runtime path.
        model_obs = {
            "state": state,
            "base_image": images["cam_high"],
            "left_wrist_image": images["cam_left_wrist"],
            "right_wrist_image": images["cam_right_wrist"],
            "left_force": self._latest_force(obs, "left_force_history_120hz"),
            "right_force": self._latest_force(obs, "right_force_history_120hz"),
            "prompt": prompt,
        }
        result = self._policy.infer(model_obs)
        actions = np.asarray(result.get("actions"), dtype=np.float32)
        if actions.shape != (50, 16) or not np.all(np.isfinite(actions)):
            raise ValueError(f"ForceVLA must return finite (50,16), got {actions.shape}")
        return {
            **result,
            "actions": actions,
            "active_prompt_key": key,
            "active_prompt": prompt,
            "prompt_revision": revision,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--prompt", action="append", type=parse_prompt_assignment, default=[])
    parser.add_argument("--initial-key", default="1")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=12000)
    parser.add_argument("--quit-key", default="q")
    parser.add_argument("--list-prompts", action="store_true")
    args = parser.parse_args()

    prompts = load_prompt_mapping(args.prompt_file, args.prompt)
    print(json.dumps(prompts, ensure_ascii=False, indent=2))
    if args.list_prompts:
        return
    prompt_bank = PromptBank(prompts, args.initial_key)

    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config

    LOGGER.info("Restoring reproduced ForceVLA checkpoint from %s", args.checkpoint)
    core = policy_config.create_trained_policy(
        config.get_config("forcevla_cr1_bimanual_full"), args.checkpoint
    )
    metadata = {
        **core.metadata,
        "protocol": "cr1-forcevla-bimanual-keyboard-v1",
        "camera_names": list(CAMERAS),
        "state_dim": 16,
        "action_dim": 16,
        "action_horizon": 50,
        "force_input": "latest-valid-left-right-6d-wrench",
        "force_fast_loop": "off",
        "ready_gate_query_enabled": False,
        "prompt_keys": prompts,
    }
    policy = ForceVlaKeyboardPolicy(core, prompt_bank, metadata)
    keyboard = KeyboardPromptController(prompt_bank, quit_key=args.quit_key)
    LOGGER.info(
        "Serving reproduced ForceVLA on %s:%d host=%s; press 1-7 to switch Vase SUBtasks",
        args.host,
        args.port,
        socket.gethostname(),
    )
    try:
        keyboard.start()
        websocket_policy_server.WebsocketPolicyServer(
            policy=policy, host=args.host, port=args.port, metadata=metadata
        ).serve_forever()
    finally:
        keyboard.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
