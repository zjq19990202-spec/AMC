#!/usr/bin/env python3
"""Production entry point for force-conditioned Atomic PI0.5 5x10 inference."""

import logging

from serve_force_pi05_keyboard_2x25 import main


if __name__ == "__main__":
    # The implementation module's __main__ block is not executed when it is
    # imported here. Configure INFO explicitly so prompt key changes, warmup
    # phases, and serving state remain visible in the interactive terminal.
    logging.basicConfig(level=logging.INFO, force=True)
    main()
