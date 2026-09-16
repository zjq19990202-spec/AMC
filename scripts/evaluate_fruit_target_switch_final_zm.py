#!/usr/bin/env python3
"""Run the multiframe evaluator through the checkpoint's final-zM route."""

import evaluate_fruit_target_switch as single
import evaluate_fruit_target_switch_multiframe as evaluator


if __name__ == "__main__":
    evaluator._sample_variants_layerwise = single._sample_variants
    evaluator.main()
