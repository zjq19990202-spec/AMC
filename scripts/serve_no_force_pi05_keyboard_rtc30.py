#!/usr/bin/env python3
"""Fixed production entry point for no-force Atomic PI0.5 30-step RTC."""

from __future__ import annotations

import logging
import sys

from serve_layerwise_pi05_keyboard import main


if __name__ == "__main__":
    # RTC prefetch continuously reuses the selected SUBtask. Keyboard presses
    # switch that selection instead of granting only one inference request.
    logging.basicConfig(level=logging.INFO, force=True)
    main([*sys.argv[1:], "--rtc30", "--prompt-mode", "continuous"])
