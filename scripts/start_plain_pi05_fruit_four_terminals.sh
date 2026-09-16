#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
DOWNLOAD_DIR=/home/admin123/下载
HOME_POSE=${HOME_POSE:-new}

if [[ "$HOME_POSE" != "old" && "$HOME_POSE" != "new" ]]; then
  echo "HOME_POSE must be old or new" >&2
  exit 2
fi

for required in \
  "$PROJECT_ROOT/scripts/start_plain_pi05_fruit_inference.sh" \
  "$DOWNLOAD_DIR/ar.exp" \
  "$DOWNLOAD_DIR/policy_no_rtc20.exp" \
  "$DOWNLOAD_DIR/rl_50.exp"; do
  if [[ ! -f "$required" ]]; then
    echo "Required launcher is missing: $required" >&2
    exit 2
  fi
done

launch_terminal() {
  local title=$1
  local command=$2
  gnome-terminal --title="$title" -- bash -lc \
    "$command; terminal_status=\$?; echo; echo '[process exited]' \"\$terminal_status\"; exec bash -i"
}

launch_terminal "1 PI0.5 25K Fruit inference" \
  "exec env CUDA_VISIBLE_DEVICES=1 '$PROJECT_ROOT/scripts/start_plain_pi05_fruit_inference.sh'"

launch_terminal "2 CR1 RoboticsService" \
  "exec expect '$DOWNLOAD_DIR/ar.exp'"

launch_terminal "3 CR1 policy no-RTC20" \
  "echo 'Waiting for PI0.5 server on :12000...'; \
   until ss -ltn | grep -q ':12000 '; do sleep 1; done; \
   echo 'PI0.5 server ready; starting no-force policy bridge'; \
   exec expect '$DOWNLOAD_DIR/policy_no_rtc20.exp'"

launch_terminal "4 CR1 rl_deploy $HOME_POSE pose" \
  "echo 'Waiting for no-RTC20 policy bridge on cr1-106...'; \
   until ssh -o BatchMode=yes -o ConnectTimeout=2 cr1-106 \
     \"pgrep -f './policy_dds auto.*--policy-execution-horizon 20.*--policy-sync-inference' >/dev/null\" 2>/dev/null; do sleep 1; done; \
   echo 'Policy bridge ready; starting rl_deploy with $HOME_POSE home pose'; \
   exec expect '$DOWNLOAD_DIR/rl_50.exp' '$HOME_POSE'"

echo "Opened four terminals: PI0.5 Fruit, RoboticsService, no-RTC20 policy bridge, rl_deploy ($HOME_POSE pose)."
