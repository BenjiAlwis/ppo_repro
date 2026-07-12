#!/usr/bin/env bash
# run_experiments.sh
# Launches the full CORE experiment grid for the PPO reproduction:
#   4 environments x 3 modes x 3 seeds = 36 runs.
#
# Runs are sequential by default (safe on a single GPU). To parallelise,
# see the note at the bottom.
#
# Usage:
#   bash run_experiments.sh                 # full grid, 1M steps each
#   bash run_experiments.sh 200000          # quick smoke test, 200k steps each
#
# Estimated time (single modern GPU): ~20-60 min per 1M-step run.

set -e

# Headless rendering (RunPod / clusters). Harmless on a desktop.
export MUJOCO_GL="${MUJOCO_GL:-egl}"

# Auto-pick the persistent volume on RunPod so results survive a pod restart.
if [ -d /workspace ]; then
  LOG_DIR="/workspace/runs"
else
  LOG_DIR="runs"
fi

TIMESTEPS="${1:-1000000}"

ENVS=("HalfCheetah-v5" "Hopper-v5" "Walker2d-v5" "InvertedPendulum-v5")
MODES=("clip" "noclip" "kl")
SEEDS=(1 2 3)

echo "=== PPO reproduction grid ==="
echo "timesteps per run: ${TIMESTEPS}"
echo "envs:  ${ENVS[*]}"
echo "modes: ${MODES[*]}"
echo "seeds: ${SEEDS[*]}"
echo "total runs: $(( ${#ENVS[@]} * ${#MODES[@]} * ${#SEEDS[@]} ))"
echo "log dir: ${LOG_DIR}   MUJOCO_GL=${MUJOCO_GL}"
echo "============================="

for env in "${ENVS[@]}"; do
  for mode in "${MODES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      echo ">>> ${env} | ${mode} | seed ${seed}"
      python ppo_continuous.py \
        --env-id "${env}" \
        --mode "${mode}" \
        --seed "${seed}" \
        --total-timesteps "${TIMESTEPS}" \
        --log-dir "${LOG_DIR}"
    done
  done
done

echo "=== all runs complete; generating figures ==="
python plot_results.py --log-dir "${LOG_DIR}" --out figures
echo "=== done. See ./figures ==="

# ---------------------------------------------------------------------------
# To parallelise across N GPUs or processes, replace the inner python call with
# a job queued to GNU parallel or a scheduler, e.g.:
#   sem -j 4 "CUDA_VISIBLE_DEVICES=\$((RANDOM%4)) python ppo_continuous.py ..."
# Keep an eye on GPU memory: each MuJoCo PPO run is small (<1 GB), so several
# fit on one GPU.
# ---------------------------------------------------------------------------
