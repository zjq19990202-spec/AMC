#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _server_module():
    path = Path(__file__).parents[1] / "scripts" / "serve_layerwise_pi05_keyboard.py"
    spec = importlib.util.spec_from_file_location("layerwise_keyboard_server", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_prompt_assignments_and_overrides(tmp_path: Path) -> None:
    server = _server_module()
    prompt_file = tmp_path / "prompts.json"
    prompt_file.write_text('{"1": "first", "2": "old second"}', encoding="utf-8")
    prompts = server.load_prompt_mapping(
        prompt_file,
        [server.parse_prompt_assignment("2=new second"), server.parse_prompt_assignment("5=fifth")],
    )
    assert prompts == {"1": "first", "2": "new second", "5": "fifth"}


def test_external_drive_runtime_assets_are_default_paths() -> None:
    server = _server_module()
    args = server.parse_args(["--prompt", "1=move left"])
    assert args.checkpoint == (
        Path(server.__file__).resolve().parents[1]
        / "runtime_assets"
        / "checkpoints"
        / "target2058_zm_25000"
    )
    assert args.norm_assets_dir == (
        Path(server.__file__).resolve().parents[1] / "runtime_assets" / "norm"
    )
    assert args.norm_asset_id == "openpi_norm_compact_accepted_v3_no_short_no_idle"


@pytest.mark.parametrize("value", ["", "1", "q=reserved", "Q=reserved", "2=", "!=text", "10=text"])
def test_bad_prompt_assignments_are_rejected(value: str) -> None:
    server = _server_module()
    with pytest.raises(argparse.ArgumentTypeError):
        server.parse_prompt_assignment(value)


def test_prompt_bank_supports_more_keys_and_each_press_arms_one_inference() -> None:
    server = _server_module()
    bank = server.PromptBank({"1": "one", "2": "two", "a": "alpha"}, "1")
    assert bank.snapshot() == ("1", "one", 0)
    assert bank.has_pending_prompt() is False
    assert bank.select("1") == ("1", "one", 1)
    assert bank.acquire(one_shot=True) == ("1", "one", 1, False)
    assert bank.has_pending_prompt() is False
    assert bank.select("a") == ("a", "alpha", 2)


def test_repeated_key_events_collapse_into_one_pending_inference() -> None:
    server = _server_module()
    bank = server.PromptBank({"1": "one"}, "1")
    for _ in range(100):
        bank.select("1")
    assert bank.has_pending_prompt() is True
    assert bank.acquire(one_shot=True) == ("1", "one", 100, False)
    assert bank.has_pending_prompt() is False


def test_old_key_press_expires_instead_of_authorizing_a_future_chunk() -> None:
    server = _server_module()
    bank = server.PromptBank({"1": "one"}, "1", active_window_s=0.01)
    bank.select("1")
    time.sleep(0.02)
    assert bank.has_pending_prompt() is False


def test_space_pause_preserves_prompt_and_select_resumes() -> None:
    server = _server_module()
    bank = server.PromptBank({"1": "one", "2": "two"}, "1")
    assert bank.is_paused() is False
    assert bank.toggle_paused() == (True, "1", "one", 1)
    assert bank.is_paused() is True
    with pytest.raises(LookupError, match="paused"):
        bank.acquire(one_shot=False)

    assert bank.toggle_paused() == (False, "1", "one", 2)
    assert bank.acquire(one_shot=False) == ("1", "one", 2, False)

    bank.toggle_paused()
    assert bank.select("2") == ("2", "two", 4)
    assert bank.is_paused() is False


def test_continuous_policy_returns_waiting_while_space_paused() -> None:
    server = _server_module()

    class FakePolicy:
        def __init__(self) -> None:
            self.calls = 0

        def infer(self, _obs, **_kwargs):
            self.calls += 1
            return {"actions": np.zeros((50, 16), dtype=np.float32)}

    fake = FakePolicy()
    bank = server.PromptBank({"1": "one"}, "1")
    policy = server.Cr1KeyboardPromptPolicy(fake, bank, {}, one_shot=False)
    bank.toggle_paused()
    assert policy.infer({"_rserl_gate_query": "ready"})["ready"] is False
    waiting = policy.infer(
        {
            "state": np.zeros(16, dtype=np.float32),
            "images": {"cam_high": np.zeros((3, 224, 224), dtype=np.uint8)},
        }
    )
    assert waiting == {
        "waiting_for_prompt": True,
        "prompt_grant_consumed": False,
        "paused": True,
    }
    assert fake.calls == 0


def test_one_shot_acquire_waits_until_a_key_is_pressed() -> None:
    server = _server_module()
    bank = server.PromptBank({"1": "one", "b": "beta"}, "1")
    result = []
    started = threading.Event()

    def acquire() -> None:
        started.set()
        result.append(bank.acquire(one_shot=True))

    worker = threading.Thread(target=acquire)
    worker.start()
    assert started.wait(timeout=1.0)
    worker.join(timeout=0.05)
    assert worker.is_alive()
    bank.select("b")
    worker.join(timeout=1.0)
    assert not worker.is_alive()
    assert result == [("b", "beta", 1, False)]


def test_prompt_switch_composes_training_style_transition_exactly_once() -> None:
    server = _server_module()
    bank = server.PromptBank(
        {"1": " grasp the orange. ", "2": "place it in the box;"}, "1"
    )

    bank.select("1")
    assert bank.acquire(one_shot=True) == ("1", "grasp the orange", 1, False)
    bank.select("2")
    assert bank.acquire(one_shot=True) == (
        "2",
        "grasp the orange; then place it in the box",
        2,
        True,
    )
    bank.select("2")
    assert bank.acquire(one_shot=True) == ("2", "place it in the box", 3, False)


def test_discarded_warmup_does_not_consume_pending_prompt_transition() -> None:
    server = _server_module()
    bank = server.PromptBank({"1": "first", "2": "second"}, "1")
    bank.select("1")
    assert bank.acquire(one_shot=True)[1:] == ("first", 1, False)

    bank.select("2")
    assert bank.acquire(one_shot=False, consume_transition=False) == (
        "2", "second", 2, False
    )
    assert bank.has_pending_prompt() is True
    assert bank.acquire(one_shot=True) == (
        "2", "first; then second", 2, True
    )


def test_cr1_contract_overrides_prompt_and_validates_output() -> None:
    server = _server_module()

    class FakePolicy:
        def __init__(self) -> None:
            self.calls = []

        def infer(self, obs, **kwargs):
            self.calls.append((obs, kwargs))
            return {"actions": np.zeros((50, 16), dtype=np.float32)}

    fake = FakePolicy()
    bank = server.PromptBank({"1": "one", "2": "two"}, "1")
    policy = server.Cr1KeyboardPromptPolicy(fake, bank, {}, one_shot=False)
    obs = {
        "state": np.zeros(16, dtype=np.float32),
        "images": {"cam_high": np.zeros((3, 224, 224), dtype=np.uint8)},
    }
    first = policy.infer(obs, inference_delay=5, execution_horizon=25)
    assert first["active_prompt_key"] == "1"
    assert first["prompt_changed"] is True
    assert fake.calls[-1][0]["prompt"] == "one"
    assert fake.calls[-1][1]["inference_delay"] == 5

    bank.select("2")
    second = policy.infer(obs)
    assert second["active_prompt"] == "one; then two"
    assert second["prompt_revision"] == 1
    assert second["prompt_changed"] is True
    assert second["prompt_transition_composed"] is True
    assert fake.calls[-1][0]["prompt"] == "one; then two"


def test_executable_inference_is_appended_to_csv_trace(tmp_path: Path) -> None:
    server = _server_module()

    class FakePolicy:
        def infer(self, _obs, **_kwargs):
            actions = np.arange(50 * 16, dtype=np.float32).reshape(50, 16)
            return {"actions": actions}

    trace_csv = tmp_path / "inference.csv"
    images_dir = tmp_path / "images"
    bank = server.PromptBank({"1": "open drawer"}, "1")
    policy = server.Cr1KeyboardPromptPolicy(
        FakePolicy(),
        bank,
        {},
        one_shot=True,
        trace_csv=trace_csv,
        trace_images_dir=images_dir,
    )
    bank.select("1")
    obs = {
        "state": np.arange(16, dtype=np.float32),
        "images": {
            "cam_high": np.zeros((3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.ones((3, 224, 224), dtype=np.uint8),
        },
    }
    result = policy.infer(obs, inference_delay=5, execution_horizon=25)
    assert result["actions"].shape == (50, 16)

    import csv

    with trace_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 50
    assert {row["request_id"] for row in rows} == {"1"}
    assert rows[0]["prompt"] == "open drawer"
    assert rows[0]["state_14"] == "14.0"
    assert rows[49]["action_step"] == "49"
    assert rows[49]["action_15"] == "799.0"
    assert rows[0]["cam_high_crc32"]
    assert rows[0]["cam_left_wrist_crc32"]
    assert rows[0]["cam_right_wrist_crc32"] == ""
    assert Path(rows[0]["cam_high_image"]) == (
        images_dir / "request_000001_cam_high.png"
    ).resolve()
    assert (images_dir / "request_000001_cam_high.png").is_file()
    assert (images_dir / "request_000001_cam_left_wrist.png").is_file()
    assert not (images_dir / "request_000001_cam_right_wrist.png").exists()

    bank.select("1")
    obs["images"]["cam_high"] = np.full((3, 224, 224), 2, dtype=np.uint8)
    policy.infer(obs, inference_delay=5, execution_horizon=25)
    with trace_csv.open(newline="", encoding="utf-8") as stream:
        all_rows = list(csv.DictReader(stream))
    assert len(all_rows) == 100
    assert {row["request_id"] for row in all_rows} == {"1", "2"}
    assert Path(all_rows[50]["cam_high_image"]) == (
        images_dir / "request_000002_cam_high.png"
    ).resolve()
    assert (images_dir / "request_000002_cam_high.png").is_file()
    assert all_rows[0]["cam_high_crc32"] != all_rows[50]["cam_high_crc32"]


def test_cr1_contract_accepts_bridge_control_messages_without_images() -> None:
    server = _server_module()
    policy = server.Cr1KeyboardPromptPolicy(
        SimpleNamespace(infer=lambda *_args, **_kwargs: None),
        server.PromptBank({"1": "one"}, "1"),
        {},
        one_shot=False,
    )
    assert policy.infer({"_rserl_gate_query": "ready"})["ready"] is True
    assert policy.infer({"_rserl_event": "episode_start"}) == {
        "ok": True,
        "event": "episode_start",
    }


def test_one_shot_ready_gate_reflects_pending_prompt() -> None:
    server = _server_module()
    bank = server.PromptBank({"1": "one"}, "1")
    policy = server.Cr1KeyboardPromptPolicy(
        SimpleNamespace(infer=lambda *_args, **_kwargs: None),
        bank,
        {},
        one_shot=True,
    )
    assert policy.infer({"_rserl_gate_query": "ready"})["ready"] is False
    bank.select("1")
    assert policy.infer({"_rserl_gate_query": "ready"})["ready"] is True


def test_one_shot_formal_request_returns_without_inference_until_key() -> None:
    server = _server_module()

    class FakePolicy:
        def __init__(self) -> None:
            self.calls = 0

        def infer(self, _obs, **_kwargs):
            self.calls += 1
            return {"actions": np.zeros((50, 16), dtype=np.float32)}

    fake = FakePolicy()
    bank = server.PromptBank({"1": "one"}, "1")
    policy = server.Cr1KeyboardPromptPolicy(fake, bank, {}, one_shot=True)
    obs = {
        "state": np.zeros(16, dtype=np.float32),
        "images": {"cam_high": np.zeros((3, 224, 224), dtype=np.uint8)},
    }

    waiting = policy.infer(dict(obs))
    assert waiting == {
        "waiting_for_prompt": True,
        "prompt_grant_consumed": False,
    }
    assert fake.calls == 0

    bank.select("1")
    result = policy.infer(dict(obs))
    assert result["actions"].shape == (50, 16)
    assert result["prompt_grant_consumed"] is True
    assert fake.calls == 1


def test_discarded_bridge_warmup_does_not_consume_one_shot_prompt() -> None:
    server = _server_module()

    class FakePolicy:
        def infer(self, _obs, **_kwargs):
            return {"actions": np.zeros((50, 16), dtype=np.float32)}

    bank = server.PromptBank({"1": "one"}, "1")
    policy = server.Cr1KeyboardPromptPolicy(
        FakePolicy(),
        bank,
        {},
        one_shot=True,
    )
    obs = {
        "state": np.zeros(16, dtype=np.float32),
        "images": {"cam_high": np.zeros((3, 224, 224), dtype=np.uint8)},
        "_cr1_consume_prompt": False,
    }
    warmup = policy.infer(obs)
    assert warmup["prompt_grant_consumed"] is False
    assert bank.has_pending_prompt() is False

    bank.select("1")
    obs["_cr1_consume_prompt"] = True
    executable = policy.infer(obs)
    assert executable["prompt_grant_consumed"] is True
    assert bank.has_pending_prompt() is False


def test_policy_composes_switch_only_for_next_executable_inference() -> None:
    server = _server_module()

    class FakePolicy:
        def __init__(self) -> None:
            self.prompts = []

        def infer(self, obs, **_kwargs):
            self.prompts.append(obs["prompt"])
            return {"actions": np.zeros((50, 16), dtype=np.float32)}

    fake = FakePolicy()
    bank = server.PromptBank({"1": "pick orange.", "2": "place orange."}, "1")
    policy = server.Cr1KeyboardPromptPolicy(fake, bank, {}, one_shot=True)
    base_obs = {
        "state": np.zeros(16, dtype=np.float32),
        "images": {"cam_high": np.zeros((3, 224, 224), dtype=np.uint8)},
    }

    bank.select("1")
    first = policy.infer(dict(base_obs))
    assert first["active_prompt"] == "pick orange"
    assert first["prompt_transition_composed"] is False

    bank.select("2")
    warmup = policy.infer({**base_obs, "_cr1_consume_prompt": False})
    assert warmup["active_prompt"] == "place orange"
    assert warmup["prompt_transition_composed"] is False
    assert bank.has_pending_prompt() is True

    switched = policy.infer(dict(base_obs))
    assert switched["active_prompt"] == "pick orange; then place orange"
    assert switched["prompt_transition_composed"] is True

    bank.select("2")
    steady = policy.infer(dict(base_obs))
    assert steady["active_prompt"] == "place orange"
    assert steady["prompt_transition_composed"] is False
    assert fake.prompts == [
        "pick orange",
        "place orange",
        "pick orange; then place orange",
        "place orange",
    ]
