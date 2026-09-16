#!/usr/bin/env python3
"""Serve the local LA4VLA-mixed stock PI0.5 checkpoint for CR1.

This entry point pins the model and preprocessing contract used by the local
LA4VLA baseline while reusing the production CR1 keyboard/WebSocket server.
Prompt assignments remain command-line arguments, so the same launcher can be
used for Fruit, Vase, Plug, or any other native SUBtask bank.
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

from serve_layerwise_pi05_keyboard import main


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "runtime_assets"
    / "checkpoints"
    / "la4vla_mixed50_union2375_masked_subtask_35k"
)
DEFAULT_NORM_DIR = PROJECT_ROOT / "runtime_assets" / "norm"
DEFAULT_NORM_ID = "openpi_norm_union2375_allframes_v1"


def _has_option(arguments: list[str], option: str) -> bool:
    return option in arguments or any(value.startswith(f"{option}=") for value in arguments)


def _arguments(user_arguments: list[str]) -> list[str]:
    arguments = list(user_arguments)
    defaults = (
        ("--checkpoint", str(DEFAULT_CHECKPOINT)),
        ("--norm-assets-dir", str(DEFAULT_NORM_DIR)),
        ("--norm-asset-id", DEFAULT_NORM_ID),
        ("--max-token-len", "200"),
        ("--host", "0.0.0.0"),
        ("--port", "12000"),
        ("--prompt-mode", "continuous"),
    )
    for option, value in defaults:
        if not _has_option(arguments, option):
            arguments.extend((option, value))
    if "--plain-pi05" not in arguments:
        arguments.append("--plain-pi05")
    if "--disable-prompt-transition-compose" not in arguments:
        arguments.append("--disable-prompt-transition-compose")
    return arguments


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(_arguments(sys.argv[1:]))
