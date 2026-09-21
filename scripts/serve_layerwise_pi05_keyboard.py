#!/usr/bin/env python3
"""Serve an Atomic layerwise PI0.5 checkpoint with keyboard-selectable prompts.

The websocket contract matches ``cr1_force/serving/openpi_policy_client_bridge.py``:
the client sends a native 16-D CR1 state and up to three CHW RGB images, while
the server returns a 50 x 16 absolute-action chunk. Configure up to 60
single-key prompts using ``0-9``, ``a-z`` or ``A-Z``. The quit key defaults to
``q`` but can be moved when that physical key is part of a prompt layout. In
the default one-shot mode, only a recent key-down/repeat event
authorizes one inference request; stale presses expire instead of being saved
for a later action-chunk boundary.

Example::

    PYTHONPATH=$PWD/vendor/pi0.5/src:$PWD/vendor/pi0.5/packages/openpi-client/src:$PWD/src \
      /mnt/cunchu/zjq/pi0.5_env/.venv/bin/python \
      scripts/serve_layerwise_pi05_keyboard.py \
      --prompt '1=prompt one' --prompt '2=prompt two' --prompt 'a=prompt three'

When executed from this external-drive checkout, the default checkpoint and
normalization paths resolve to the versioned folders under ``runtime_assets``.
Explicit command-line paths still override those defaults, which is useful
after staging the source into remote RAM or when selecting the auxiliary run.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import os
import signal
import socket
import string
import sys
import termios
import threading
import time
import tty
import zlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
STATE_DIM = 16
ACTION_DIM = 16
ACTION_HORIZON = 50
DEFAULT_QUIT_KEY = "q"
VALID_PROMPT_KEYS = frozenset(string.digits + string.ascii_letters)
CLI_PROMPT_KEYS = VALID_PROMPT_KEYS - frozenset(("q", "Q"))
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ASSETS = PROJECT_ROOT / "runtime_assets"
DEFAULT_CHECKPOINT = RUNTIME_ASSETS / "checkpoints" / "target2058_zm_25000"
DEFAULT_NORM_ASSETS_DIR = RUNTIME_ASSETS / "norm"
DEFAULT_NORM_ASSET_ID = "openpi_norm_compact_accepted_v3_no_short_no_idle"
DEFAULT_DATASET_ROOT = Path("/mnt/cunchu/zjq/target/lerobot_compact_accepted_v3_no_short_no_idle")


def compose_horizon_prompts(prompts: Sequence[str]) -> str:
    """Match training's ordered, de-duplicated cross-SUBtask prompt contract."""

    cleaned: list[str] = []
    for value in prompts:
        prompt = " ".join(str(value).strip().split()).rstrip(" .;")
        if prompt and (not cleaned or prompt != cleaned[-1]):
            cleaned.append(prompt)
    return "; then ".join(cleaned)


def parse_prompt_assignment(value: str) -> tuple[str, str]:
    """Parse one ``KEY=TEXT`` command-line assignment."""

    key, separator, prompt = value.partition("=")
    key = key.strip()
    prompt = prompt.strip()
    if not separator or key not in CLI_PROMPT_KEYS or not prompt:
        raise argparse.ArgumentTypeError(
            "prompt must be KEY=TEXT where KEY is one ASCII letter/digit; "
            "q/Q require a prompt file plus a different --quit-key"
        )
    return key, prompt


def load_prompt_mapping(
    prompt_file: Path | None,
    assignments: Sequence[tuple[str, str]],
) -> dict[str, str]:
    """Load a JSON prompt map and apply command-line overrides."""

    prompts: dict[str, str] = {}
    if prompt_file is not None:
        payload = json.loads(prompt_file.expanduser().read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("--prompt-file must contain a JSON object")
        for raw_key, raw_prompt in payload.items():
            key = str(raw_key).strip()
            prompt = str(raw_prompt).strip()
            if key not in VALID_PROMPT_KEYS or not prompt:
                raise ValueError(
                    "prompt-file keys must be one ASCII letter/digit "
                    "and values must be non-empty strings"
                )
            prompts[key] = prompt
    prompts.update(assignments)
    if not prompts:
        raise ValueError("configure at least one prompt with --prompt-file or --prompt")
    return dict(sorted(prompts.items()))


class PromptBank:
    """Thread-safe prompt selection and short-lived one-shot inference gate."""

    def __init__(
        self,
        prompts: Mapping[str, str],
        initial_key: str,
        *,
        active_window_s: float = 0.25,
    ) -> None:
        self._prompts = dict(prompts)
        if initial_key not in self._prompts:
            raise ValueError(
                f"initial prompt key {initial_key!r} is not configured; "
                f"available={list(self._prompts)}"
            )
        self._key = initial_key
        self._revision = 0
        self._pending = False
        self._active_window_s = active_window_s
        self._pending_until = 0.0
        self._last_consumed_key: str | None = None
        self._paused = False
        self._condition = threading.Condition()

    @property
    def prompts(self) -> dict[str, str]:
        return dict(self._prompts)

    def select(self, key: str) -> tuple[str, str, int]:
        """Select a prompt and briefly authorize one inference.

        Terminal input has key-down/repeat events but no key-up event. Treat a
        recent key event as "currently pressed": the authorization expires
        quickly instead of being stored until some future chunk request.
        Repeats refresh one slot, so holding a key cannot queue robot chunks.
        """

        with self._condition:
            if key not in self._prompts:
                raise KeyError(key)
            self._key = key
            self._revision += 1
            self._paused = False
            self._pending = True
            self._pending_until = time.monotonic() + self._active_window_s
            self._condition.notify_all()
            return self._key, self._prompts[self._key], self._revision

    def toggle_paused(self) -> tuple[bool, str, str, int]:
        """Toggle continuous inference while preserving the selected prompt."""

        with self._condition:
            self._paused = not self._paused
            self._revision += 1
            self._pending = False
            self._condition.notify_all()
            return self._paused, self._key, self._prompts[self._key], self._revision

    def is_paused(self) -> bool:
        with self._condition:
            return self._paused

    def snapshot(self) -> tuple[str, str, int]:
        with self._condition:
            return self._key, self._prompts[self._key], self._revision

    def has_pending_prompt(self) -> bool:
        with self._condition:
            if self._pending and time.monotonic() > self._pending_until:
                self._pending = False
            return self._pending

    def acquire(
        self,
        *,
        one_shot: bool,
        consume_transition: bool = True,
        wait_for_prompt: bool = True,
    ) -> tuple[str, str, int, bool]:
        """Return one model prompt, optionally composing one SUBtask transition.

        The first executable inference after a semantic key change receives
        ``old; then new``, exactly like a training horizon crossing two SUBtask
        segments. Later inferences with the same selected key receive only the
        new prompt. Discarded warmups set ``consume_transition=False`` so they
        neither create nor consume this transition.
        """

        with self._condition:
            if self._paused:
                raise LookupError("prompt inference is paused")
            if one_shot:
                while not self._pending or time.monotonic() > self._pending_until:
                    self._pending = False
                    if not wait_for_prompt:
                        raise LookupError("no live key event")
                    LOGGER.info("No prompt armed; next inference is paused waiting for a key")
                    self._condition.wait()
                self._pending = False
            base_prompt = self._prompts[self._key]
            normalized_base_prompt = compose_horizon_prompts((base_prompt,))
            model_prompt = normalized_base_prompt
            transition_composed = False
            if consume_transition:
                if self._last_consumed_key is not None:
                    previous_prompt = self._prompts[self._last_consumed_key]
                    model_prompt = compose_horizon_prompts((previous_prompt, base_prompt))
                    transition_composed = model_prompt != normalized_base_prompt
                self._last_consumed_key = self._key
            return self._key, model_prompt, self._revision, transition_composed


class KeyboardPromptController:
    """Read configured single-key prompt choices without requiring Enter."""

    def __init__(self, prompt_bank: PromptBank, *, quit_key: str = DEFAULT_QUIT_KEY) -> None:
        self._prompt_bank = prompt_bank
        self._quit_keys = frozenset((quit_key.lower(), quit_key.upper()))
        self._fd: int | None = None
        self._saved_terminal: list[Any] | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not sys.stdin.isatty():
            raise RuntimeError(
                "keyboard control requires a TTY; use --no-keyboard for a fixed prompt"
            )
        self._fd = sys.stdin.fileno()
        self._saved_terminal = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._thread = threading.Thread(
            target=self._run, name="prompt-keyboard", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        if self._fd is not None and self._saved_terminal is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved_terminal)
        self._fd = None
        self._saved_terminal = None

    def _run(self) -> None:
        assert self._fd is not None
        while True:
            raw = os.read(self._fd, 1)
            if not raw:
                return
            key = raw.decode("utf-8", errors="ignore")
            if key == " ":
                paused, selected_key, prompt, revision = self._prompt_bank.toggle_paused()
                LOGGER.info(
                    "Prompt %s by space key=%s revision=%d text=%r",
                    "paused" if paused else "resumed",
                    selected_key,
                    revision,
                    prompt,
                )
            elif key in self._prompt_bank.prompts:
                selected_key, prompt, revision = self._prompt_bank.select(key)
                LOGGER.info(
                    "Prompt active key=%s revision=%d text=%r; usable only inside the live key window",
                    selected_key,
                    revision,
                    prompt,
                )
            elif key in self._quit_keys:
                LOGGER.info("Keyboard quit requested")
                os.kill(os.getpid(), signal.SIGINT)
                return


class AtomicLayerwisePolicy:
    """Minimal original-PI0.5 inference/output path for ``AtomicPi05``.

    ``AtomicPi05.sample_actions_rtc`` intentionally falls back to a fresh
    sample.  Calling ``sample_actions`` directly makes that behavior explicit
    and avoids passing unsupported CFG/RTC keyword arguments into the atomic
    model.  Output transforms restore the exact pre-normalization state before
    ``AbsoluteActions`` adds joint deltas.
    """

    def __init__(
        self,
        model: Any,
        *,
        input_transforms: Sequence[Any],
        output_transforms: Sequence[Any],
        normalize_type: type,
        rng_seed: int,
        num_steps: int,
        metadata: Mapping[str, Any],
    ) -> None:
        import jax
        import jax.numpy as jnp
        from openpi.models import model as openpi_model
        from openpi.shared import nnx_utils

        self._jax = jax
        self._jnp = jnp
        self._observation_type = openpi_model.Observation
        self._input_transforms = tuple(input_transforms)
        self._output_transforms = tuple(output_transforms)
        self._normalize_type = normalize_type
        self._rng = jax.random.key(rng_seed)
        self._num_steps = num_steps
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self.metadata = dict(metadata)

    def infer(
        self,
        obs: dict[str, Any],
        inference_delay: int = 0,
        execution_horizon: int = ACTION_HORIZON,
        clear_prefix: bool = False,
        noise_plot: bool = False,
    ) -> dict[str, Any]:
        del inference_delay, execution_horizon, clear_prefix, noise_plot
        values = self._jax.tree.map(lambda value: value, obs)
        pre_normalization_state: np.ndarray | None = None
        for transform in self._input_transforms:
            if isinstance(transform, self._normalize_type) and "state" in values:
                pre_normalization_state = np.asarray(values["state"]).copy()
            values = transform(values)
        inputs = self._jax.tree.map(
            lambda value: self._jnp.asarray(value)[np.newaxis, ...], values
        )
        self._rng, sample_rng = self._jax.random.split(self._rng)
        actions = self._sample_actions(
            sample_rng,
            self._observation_type.from_dict(inputs),
            num_steps=self._num_steps,
        )
        outputs: dict[str, Any] = {
            "state": np.asarray(inputs["state"][0]),
            "actions": np.asarray(actions[0]),
        }
        for transform in self._output_transforms:
            outputs = transform(outputs)
            if (
                pre_normalization_state is not None
                and isinstance(transform, self._normalize_type)
            ):
                # Defensive only: Normalize is not expected in the output path.
                outputs["state"] = pre_normalization_state.copy()
            if transform.__class__.__name__ == "Unnormalize" and pre_normalization_state is not None:
                # AbsoluteActions must add deltas to the exact raw CR1 state,
                # not a value reconstructed through a normalization round trip.
                outputs["state"] = pre_normalization_state.copy()
        return outputs


class Cr1KeyboardPromptPolicy:
    """Validate the CR1 bridge contract and override its prompt."""

    def __init__(
        self,
        policy: Any,
        prompt_bank: PromptBank,
        metadata: Mapping[str, Any],
        *,
        one_shot: bool,
        compose_prompt_transition: bool = True,
        trace_csv: Path | None = None,
        trace_images_dir: Path | None = None,
    ) -> None:
        self._policy = policy
        self._prompt_bank = prompt_bank
        self._one_shot = one_shot
        self._compose_prompt_transition = compose_prompt_transition
        self._applied_revision = -1
        self._trace_csv = trace_csv.expanduser() if trace_csv is not None else None
        self._trace_images_dir = trace_images_dir.expanduser() if trace_images_dir is not None else None
        self._trace_lock = threading.Lock()
        self._trace_request_id = 0
        self.metadata = dict(metadata)

    def _write_trace_csv(
        self,
        *,
        state: np.ndarray,
        images: Mapping[str, np.ndarray],
        actions: np.ndarray,
        key: str,
        prompt: str,
        revision: int,
        transition_composed: bool,
        consume_prompt: bool,
        inference_delay: int,
        execution_horizon: int,
    ) -> None:
        if self._trace_csv is None:
            return
        with self._trace_lock:
            self._trace_request_id += 1
            request_id = self._trace_request_id
            path = self._trace_csv
            path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not path.exists() or path.stat().st_size == 0
            fieldnames = [
                "timestamp",
                "request_id",
                "action_step",
                "prompt_key",
                "prompt_revision",
                "transition_composed",
                "consume_prompt",
                "prompt",
                "inference_delay",
                "execution_horizon",
                *[f"state_{i}" for i in range(STATE_DIM)],
                *[f"action_{i}" for i in range(ACTION_DIM)],
                *[f"{name}_crc32" for name in CAMERAS],
                *[f"{name}_image" for name in CAMERAS],
            ]
            image_crc32 = {
                name: (f"{zlib.crc32(np.ascontiguousarray(images[name]).tobytes()):08x}" if name in images else "")
                for name in CAMERAS
            }
            image_paths = {name: "" for name in CAMERAS}
            if self._trace_images_dir is not None:
                from PIL import Image

                image_dir = self._trace_images_dir
                image_dir.mkdir(parents=True, exist_ok=True)
                for name, image in images.items():
                    image_path = image_dir / f"request_{request_id:06d}_{name}.png"
                    Image.fromarray(np.moveaxis(image, 0, -1), mode="RGB").save(image_path)
                    image_paths[name] = str(image_path.resolve())
            timestamp = dt.datetime.now().astimezone().isoformat(timespec="milliseconds")
            with path.open("a", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                for action_step, action in enumerate(actions):
                    row: dict[str, Any] = {
                        "timestamp": timestamp,
                        "request_id": request_id,
                        "action_step": action_step,
                        "prompt_key": key,
                        "prompt_revision": revision,
                        "transition_composed": int(transition_composed),
                        "consume_prompt": int(consume_prompt),
                        "prompt": prompt,
                        "inference_delay": inference_delay,
                        "execution_horizon": execution_horizon,
                    }
                    row.update({f"state_{i}": float(value) for i, value in enumerate(state)})
                    row.update({f"action_{i}": float(value) for i, value in enumerate(action)})
                    row.update({f"{name}_crc32": image_crc32[name] for name in CAMERAS})
                    row.update(
                        {
                            f"{name}_image": image_paths[name]
                            for name in CAMERAS
                        }
                    )
                    writer.writerow(row)
            LOGGER.info("Recorded inference request_id=%d to %s", request_id, path)

    def infer(
        self,
        obs: dict[str, Any],
        inference_delay: int = 0,
        execution_horizon: int = ACTION_HORIZON,
        clear_prefix: bool = False,
        noise_plot: bool = False,
    ) -> dict[str, Any]:
        if "_rserl_event" in obs:
            return {"ok": True, "event": str(obs["_rserl_event"])}
        if "_rserl_gate_query" in obs:
            return {
                "ok": True,
                "ready": (
                    not self._prompt_bank.is_paused()
                    and (not self._one_shot or self._prompt_bank.has_pending_prompt())
                ),
            }

        if self._prompt_bank.is_paused():
            return {
                "waiting_for_prompt": True,
                "prompt_grant_consumed": False,
                "paused": True,
            }

        # cr1_force marks warmup chunks that it will discard. They may compile
        # and prime the model using the current prompt, but must not consume the
        # operator's one-shot authorization.
        consume_prompt = bool(obs.pop("_cr1_consume_prompt", True))

        # Do not hold a WebSocket inference request open while waiting for the
        # keyboard. The CR1 loop will keep replacing its cached observation and
        # retrying.  Therefore the first request after a key press contains the
        # newest image/state, even if the action chunk has been empty for a long
        # time. A stale key event expires, so it cannot authorize a later chunk.
        # This is deliberately not the separate ready-query protocol.
        if self._one_shot and consume_prompt and not self._prompt_bank.has_pending_prompt():
            return {
                "waiting_for_prompt": True,
                "prompt_grant_consumed": False,
            }

        state = np.asarray(obs.get("state"), dtype=np.float32).reshape(-1)
        if state.shape != (STATE_DIM,):
            raise ValueError(f"state must have shape ({STATE_DIM},), got {state.shape}")
        if not np.all(np.isfinite(state)):
            raise ValueError("state contains NaN or Inf")

        source_images = obs.get("images")
        if not isinstance(source_images, dict):
            raise TypeError("images must be a mapping")
        if "cam_high" not in source_images:
            raise ValueError("images must contain cam_high")
        if extra := sorted(set(source_images) - set(CAMERAS)):
            raise ValueError(f"unexpected camera keys: {extra}")

        images: dict[str, np.ndarray] = {}
        for name, raw_image in source_images.items():
            image = np.asarray(raw_image)
            if image.shape != (3, 224, 224):
                raise ValueError(f"{name} must have shape (3, 224, 224), got {image.shape}")
            images[name] = image.astype(np.uint8, copy=False)

        try:
            key, prompt, revision, transition_composed = self._prompt_bank.acquire(
                one_shot=self._one_shot and consume_prompt,
                consume_transition=consume_prompt and self._compose_prompt_transition,
                wait_for_prompt=False,
            )
        except LookupError:
            # The live-key window may expire while this request is validating
            # its observation. Never turn that race into a blocking request.
            return {
                "waiting_for_prompt": True,
                "prompt_grant_consumed": False,
            }
        if transition_composed:
            LOGGER.info(
                "Composed one training-style SUBtask transition for prompt key=%s: %s",
                key,
                prompt,
            )
        prompt_changed = revision != self._applied_revision
        self._applied_revision = revision
        model_obs: dict[str, Any] = {
            "state": state,
            "images": images,
            "prompt": prompt,
        }
        result = self._policy.infer(
            model_obs,
            inference_delay=inference_delay,
            execution_horizon=execution_horizon,
            clear_prefix=clear_prefix,
            noise_plot=noise_plot,
        )
        actions = np.asarray(result.get("actions"), dtype=np.float32)
        if actions.shape != (ACTION_HORIZON, ACTION_DIM):
            raise ValueError(
                f"policy must return ({ACTION_HORIZON}, {ACTION_DIM}), got {actions.shape}"
            )
        if not np.all(np.isfinite(actions)):
            raise ValueError("policy returned NaN or Inf")
        self._write_trace_csv(
            state=state,
            images=images,
            actions=actions,
            key=key,
            prompt=prompt,
            revision=revision,
            transition_composed=transition_composed,
            consume_prompt=consume_prompt,
            inference_delay=inference_delay,
            execution_horizon=execution_horizon,
        )
        return {
            **result,
            "actions": actions,
            "active_prompt_key": key,
            "active_prompt": prompt,
            "prompt_revision": revision,
            "prompt_changed": prompt_changed,
            "prompt_transition_composed": transition_composed,
            "prompt_grant_consumed": self._one_shot and consume_prompt,
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--plain-pi05",
        action="store_true",
        help="Restore a stock Pi0Config(pi05=True) checkpoint instead of AtomicPi05.",
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--norm-assets-dir", type=Path, default=DEFAULT_NORM_ASSETS_DIR)
    parser.add_argument("--norm-asset-id", default=DEFAULT_NORM_ASSET_ID)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument(
        "--trace-csv",
        type=Path,
        help=(
            "Append every executable inference as 50 CSV rows containing prompt, "
            "raw 16-D state, returned 16-D action, and per-camera frame CRC32."
        ),
    )
    parser.add_argument(
        "--trace-images-dir",
        type=Path,
        help="Save every available camera image from every executable inference as PNG.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        type=parse_prompt_assignment,
        default=[],
        metavar="KEY=TEXT",
        help="Repeat for letter/digit keys; overrides the same key from --prompt-file.",
    )
    parser.add_argument(
        "--initial-key",
        help="Initially displayed prompt key. One-shot mode still starts paused.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("one-shot", "continuous"),
        default="one-shot",
        help=(
            "one-shot consumes one key press per inference and waits when none is armed; "
            "continuous reuses the selected prompt"
        ),
    )
    parser.add_argument(
        "--key-active-window-ms",
        type=float,
        default=250.0,
        help=(
            "In one-shot mode, accept a key only this many milliseconds after its "
            "latest key-down/repeat event; expired presses are never saved for a future chunk."
        ),
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-token-len", type=int, default=200)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--quit-key",
        choices=tuple(string.digits + string.ascii_lowercase),
        default=DEFAULT_QUIT_KEY,
        help="Single keyboard key used to stop the server (default: q).",
    )
    parser.add_argument("--rtc30", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--disable-prompt-transition-compose",
        action="store_true",
        help="On a key change, send only the newly selected SUBtask instead of 'old; then new'.",
    )
    parser.add_argument(
        "--coefficient-target",
        choices=("joint_delta", "tcp_twist"),
        default="joint_delta",
    )
    parser.add_argument("--invert-grippers", action="store_true")
    parser.add_argument(
        "--no-keyboard",
        action="store_true",
        help="Serve without terminal input; only valid in continuous mode.",
    )
    parser.add_argument(
        "--list-prompts",
        action="store_true",
        help="Print the resolved prompt mapping and exit before loading JAX/checkpoint.",
    )
    args = parser.parse_args(argv)
    if args.max_token_len <= 0 or args.num_steps <= 0:
        parser.error("--max-token-len and --num-steps must be positive")
    if args.key_active_window_ms <= 0:
        parser.error("--key-active-window-ms must be positive")
    return args


def build_policy(args: argparse.Namespace, prompt_bank: PromptBank) -> Cr1KeyboardPromptPolicy:
    import jax
    import jax.numpy as jnp
    from openpi import transforms
    from openpi.models import model as openpi_model
    from openpi.models import pi0_config
    from openpi.training import config as training_config

    from atomic_latent_vla.pi05.config import AtomicPi05Config
    from atomic_latent_vla.pi05.training_data import (
        _NormalizeWithoutQuantileClipping,
    )

    params_dir = args.checkpoint.expanduser() / "params"
    if not params_dir.is_dir():
        raise FileNotFoundError(f"checkpoint params not found: {params_dir}")
    norm_file = args.norm_assets_dir.expanduser() / args.norm_asset_id / "norm_stats.json"
    if not norm_file.is_file():
        raise FileNotFoundError(f"normalization asset not found: {norm_file}")
    if not hasattr(training_config, "LeRobotMarvinDataConfig"):
        raise RuntimeError(
            "the active OpenPI checkout lacks LeRobotMarvinDataConfig; use the same "
            "bundled admin123 OpenPI source under vendor/pi0.5/src"
        )
    if args.invert_grippers:
        raise ValueError(
            "admin123 Marvin adapt_to_pi=True owns the gripper conversion; "
            "--invert-grippers is incompatible with this checkpoint contract"
        )

    coefficient_dim = 14 if args.coefficient_target == "joint_delta" else 12
    if args.plain_pi05:
        model_config = pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=ACTION_HORIZON,
            max_token_len=args.max_token_len,
        )
        LOGGER.info("Restoring stock PI0.5 checkpoint from %s", args.checkpoint)
    else:
        model_config = AtomicPi05Config(
            max_token_len=args.max_token_len,
            coefficient_target_kind=args.coefficient_target,
            coefficient_target_dim=coefficient_dim,
            enable_layerwise_atomic_flow=True,
            fast_action_ce_loss_weight=0.0,
            subtask_ce_loss_weight=0.0,
        )
        LOGGER.info("Restoring Atomic layerwise checkpoint from %s", args.checkpoint)
    params = openpi_model.restore_params(params_dir, dtype=jnp.bfloat16)
    model = model_config.load(params)
    model.eval()

    # Use the released PI0.5 tokenizer/normalization transform factory while
    # retaining the AtomicPi05 model itself. This is the exact training/eval
    # bridge used by this repository.
    transform_model_config = pi0_config.Pi0Config(
        pi05=True, max_token_len=args.max_token_len
    )
    data_factory = training_config.LeRobotMarvinDataConfig(
        repo_id=str(args.dataset_root.expanduser()),
        prompt_from_task=True,
        adapt_to_pi=True,
        assets=training_config.AssetsConfig(
            assets_dir=str(args.norm_assets_dir.expanduser()),
            asset_id=args.norm_asset_id,
        ),
    )
    data_config = data_factory.create(
        args.norm_assets_dir.expanduser(), transform_model_config
    )
    if data_config.norm_stats is None:
        raise RuntimeError(f"failed to load norm stats from {norm_file}")

    initial_key, initial_prompt, _ = prompt_bank.snapshot()
    input_transforms = [
        transforms.InjectDefaultPrompt(initial_prompt),
        *data_config.data_transforms.inputs,
        _NormalizeWithoutQuantileClipping(
            data_config.norm_stats,
            use_quantiles=data_config.use_quantile_norm,
        ),
        *data_config.model_transforms.inputs,
    ]
    output_transforms = [
        *data_config.model_transforms.outputs,
        transforms.Unnormalize(
            data_config.norm_stats, use_quantiles=data_config.use_quantile_norm
        ),
        *data_config.data_transforms.outputs,
    ]
    rtc_enabled = bool(getattr(args, "rtc30", False))
    metadata = {
        "protocol": (
            "cr1-marvin-adapted-plain-pi05-keyboard-v1"
            if args.plain_pi05
            else "cr1-marvin-adapted-atomic-layerwise-pi05-keyboard-v1"
        ),
        "camera_names": list(CAMERAS),
        "required_camera_names": ["cam_high"],
        "state_dim": STATE_DIM,
        "action_horizon": ACTION_HORIZON,
        "action_dim": ACTION_DIM,
        "ready_gate_query_enabled": False,
        "nonblocking_prompt_wait": args.prompt_mode == "one-shot" and not rtc_enabled,
        "rtc_enabled": rtc_enabled,
        "prompt_keys": prompt_bank.prompts,
        "initial_prompt_key": initial_key,
        "prompt_mode": args.prompt_mode,
        "key_active_window_ms": args.key_active_window_ms,
        "prompt_transition_contract": "old; then new exactly once after a semantic key change",
        "prompt_applies_at": "next_websocket_inference_request",
        "norm_asset_id": args.norm_asset_id,
        "max_token_len": args.max_token_len,
        "layerwise_atomic_flow": not args.plain_pi05,
        "exact_raw_state_delta_decode": True,
    }
    if rtc_enabled or args.plain_pi05:
        from openpi.policies import policy as openpi_policy

        core_policy = openpi_policy.Policy(
            model,
            rng=jax.random.key(args.seed),
            transforms=input_transforms,
            output_transforms=output_transforms,
            sample_kwargs={"num_steps": args.num_steps, "use_correction": False},
            metadata=metadata,
            norm_stats=data_config.norm_stats,
            use_quantile_norm=data_config.use_quantile_norm,
        )
    else:
        core_policy = AtomicLayerwisePolicy(
            model,
            input_transforms=input_transforms,
            output_transforms=output_transforms,
            normalize_type=_NormalizeWithoutQuantileClipping,
            rng_seed=args.seed,
            num_steps=args.num_steps,
            metadata=metadata,
        )
    return Cr1KeyboardPromptPolicy(
        core_policy,
        prompt_bank,
        metadata,
        one_shot=args.prompt_mode == "one-shot",
        compose_prompt_transition=not args.disable_prompt_transition_compose,
        trace_csv=args.trace_csv,
        trace_images_dir=args.trace_images_dir,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    prompts = load_prompt_mapping(args.prompt_file, args.prompt)
    if args.quit_key in {key.lower() for key in prompts}:
        raise ValueError(
            f"--quit-key {args.quit_key!r} conflicts with a configured prompt key"
        )
    initial_key = args.initial_key or next(iter(prompts))
    if initial_key not in prompts:
        raise ValueError(
            f"--initial-key {initial_key!r} is not configured; available={list(prompts)}"
        )
    if args.no_keyboard and args.prompt_mode == "one-shot":
        raise ValueError("--no-keyboard cannot be combined with --prompt-mode one-shot")
    prompt_bank = PromptBank(
        prompts,
        initial_key,
        active_window_s=args.key_active_window_ms / 1000.0,
    )
    print(json.dumps(prompt_bank.prompts, ensure_ascii=False, indent=2))
    if args.list_prompts:
        return

    policy = build_policy(args, prompt_bank)
    from openpi.serving import websocket_policy_server

    LOGGER.info(
        "Serving %s host=%s port=%d hostname=%s checkpoint=%s "
        "contract=state[16]+CHW-RGB->actions[50,16]",
        "stock PI0.5 (LA4VLA/plain)" if args.plain_pi05 else "Atomic layerwise PI0.5",
        args.host,
        args.port,
        socket.gethostname(),
        args.checkpoint,
    )
    LOGGER.info(
        "Press a configured key to select a prompt; press %s to quit",
        args.quit_key,
    )
    if args.prompt_mode == "one-shot":
        LOGGER.info(
            "One-shot mode: startup is paused; only a currently pressed/repeating key "
            "authorizes one inference (%.0f ms window); expired presses are discarded",
            args.key_active_window_ms,
        )
    else:
        LOGGER.info("Continuous mode: the selected prompt is reused until another key is pressed")
    LOGGER.info(
        "Prompt changes apply at websocket inference boundaries; cr1_force may finish its "
        "already-buffered action chunk before the pause or new prompt takes effect"
    )

    keyboard = KeyboardPromptController(prompt_bank, quit_key=args.quit_key)
    try:
        if not args.no_keyboard:
            keyboard.start()
        websocket_policy_server.WebsocketPolicyServer(
            policy=policy,
            host=args.host,
            port=args.port,
            metadata=policy.metadata,
        ).serve_forever()
    finally:
        keyboard.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
