#!/usr/bin/env python3
"""Serve the final 512-D spherical B2 force checkpoint with strict 5x10 RTC."""

from __future__ import annotations

import logging
import sys

from serve_force_pi05_keyboard_2x25 import main


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main([*sys.argv[1:], "--model-recipe", "spherical-b2-final"])
