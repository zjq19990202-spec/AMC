#!/usr/bin/env python3
"""Stable entry point for the original no-force, full 50-step PI0.5 server."""

import logging

from serve_layerwise_pi05_keyboard import main


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
