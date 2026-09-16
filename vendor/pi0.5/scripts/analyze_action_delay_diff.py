"""
Analyze the distribution of state/action changes across inference_delay steps in the dataset.

For each episode and each timestep t, computes:
  - state[t+delay] - state[t]           (absolute state change)
  - action[t+delay] - action[t]         (raw delta action difference)

This tells you how much the prefix_actions target needs to be corrected
when the robot has moved by `inference_delay` steps between two inferences.

Usage:
    uv run scripts/analyze_action_delay_diff.py \
        --repo-id /home/featurize/data/pickbottle \
        --inference-delay 7
"""

import argparse
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


DELTA_DIMS = list(range(7)) + list(range(8, 15))   # joint dims (delta actions)
ABS_DIMS   = [7, 15]                                # gripper dims (absolute)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--inference-delay", type=int, default=7)
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Cap number of episodes to analyze (default: all)")
    args = parser.parse_args()

    delay = args.inference_delay
    print(f"Loading dataset from {args.repo_id} ...")
    ds = LeRobotDataset(args.repo_id)

    state_diffs   = []  # [t+delay] - [t], shape (state_dim,)
    action_diffs  = []  # action[t+delay] - action[t], shape (action_dim,)

    ep_indices = ds.episode_data_index
    n_episodes = len(ep_indices["from"])
    if args.max_episodes is not None:
        n_episodes = min(n_episodes, args.max_episodes)

    for ep_idx in range(n_episodes):
        start = int(ep_indices["from"][ep_idx])
        end   = int(ep_indices["to"][ep_idx])  # exclusive
        ep_len = end - start

        for t in range(ep_len - delay):
            row_t     = ds[start + t]
            row_t_del = ds[start + t + delay]

            state_t     = np.asarray(row_t["observation.state"])
            state_t_del = np.asarray(row_t_del["observation.state"])
            action_t     = np.asarray(row_t["action"])
            action_t_del = np.asarray(row_t_del["action"])

            state_diffs.append(state_t_del - state_t)
            action_diffs.append(action_t_del - action_t)

    state_diffs  = np.stack(state_diffs)   # (N, state_dim)
    action_diffs = np.stack(action_diffs)  # (N, action_dim)

    print(f"\n=== delay={delay}, N={len(state_diffs)} pairs ===\n")

    print("State diff (abs) per dim — mean | std | max:")
    for i in range(state_diffs.shape[1]):
        d = np.abs(state_diffs[:, i])
        print(f"  state[{i:2d}]  mean={d.mean():.4f}  std={d.std():.4f}  max={d.max():.4f}")

    print("\nAction diff (abs) per dim — mean | std | max:")
    action_dim = action_diffs.shape[1]
    for i in range(action_dim):
        d = np.abs(action_diffs[:, i])
        tag = "delta" if i in DELTA_DIMS else "abs"
        print(f"  action[{i:2d}] ({tag})  mean={d.mean():.4f}  std={d.std():.4f}  max={d.max():.4f}")

    print("\nDelta-joint dims overall:")
    delta = np.abs(action_diffs[:, DELTA_DIMS])
    print(f"  mean={delta.mean():.4f}  std={delta.std():.4f}  p95={np.percentile(delta,95):.4f}  max={delta.max():.4f}")

    print("\nGripper dims overall:")
    grip = np.abs(action_diffs[:, ABS_DIMS])
    print(f"  mean={grip.mean():.4f}  std={grip.std():.4f}  p95={np.percentile(grip,95):.4f}  max={grip.max():.4f}")


if __name__ == "__main__":
    main()
