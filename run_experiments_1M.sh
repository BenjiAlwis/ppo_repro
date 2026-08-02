#!/usr/bin/env bash
# run_experiments_1M.sh
# Full final-paper grid: 7 MuJoCo environments x 3 objectives x 3 seeds at 1M steps.
#   = 63 runs. Matches the paper's continuous-control suite (Figure 3).
#
# Usage:
#   bash run_experiments_1M.sh            # sequential (safe, ~30-35 h)
#   bash run_experiments_1M.sh 4          # run 4 jobs in parallel (~10 h)
#
# Parallelism is safe here: these small-MLP runs are env-bound, not GPU-bound,
# and each uses <1 GB VRAM, so several share one GPU comfortably.
#
# Logs to a SEPARATE dir (/workspace/runs_1M) so 1M results never mix with the
# earlier 300k grid. Downloads/plots should compare like-for-like budgets.

set -e

PARALLEL="${1:-1}"                       # number of concurrent jobs (default 1 = sequential)
TIMESTEPS=1000000
LOGDIR="/workspace/runs_1M"

# The paper's 7-environment continuous-control suite (Gymnasium -v5):
ENVS=(
  "HalfCheetah-v5"
  "Hopper-v5"
  "Walker2d-v5"
  "InvertedPendulum-v5"
  "InvertedDoublePendulum-v5"
  "Reacher-v5"
  "Swimmer-v5"
)
MODES=("clip" "noclip" "kl")
SEEDS=(1 2 3)

TOTAL=$(( ${#ENVS[@]} * ${#MODES[@]} * ${#SEEDS[@]} ))

echo "=== PPO 1M reproduction grid ==="
echo "timesteps per run: ${TIMESTEPS}"
echo "envs:  ${ENVS[*]}"
echo "modes: ${MODES[*]}"
echo "seeds: ${SEEDS[*]}"
echo "total runs: ${TOTAL}"
echo "parallel jobs: ${PARALLEL}"
echo "log dir: ${LOGDIR}"
echo "==============================="

mkdir -p "${LOGDIR}"

# Build the full job list
JOBS=()
for env in "${ENVS[@]}"; do
  for mode in "${MODES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      JOBS+=("${env}|${mode}|${seed}")
    done
  done
done

run_one() {
  local spec="$1"
  IFS='|' read -r env mode seed <<< "${spec}"
  echo ">>> START ${env} | ${mode} | seed ${seed}"
  python ppo_continuous.py \
    --env-id "${env}" \
    --mode "${mode}" \
    --seed "${seed}" \
    --total-timesteps "${TIMESTEPS}" \
    --log-dir "${LOGDIR}"
  echo ">>> DONE  ${env} | ${mode} | seed ${seed}"
}
export -f run_one

if [ "${PARALLEL}" -le 1 ]; then
  # Sequential
  for spec in "${JOBS[@]}"; do run_one "${spec}"; done
else
  # Parallel with a simple job-slot limiter (no GNU parallel dependency)
  active=0
  for spec in "${JOBS[@]}"; do
    run_one "${spec}" &
    active=$(( active + 1 ))
    if [ "${active}" -ge "${PARALLEL}" ]; then
      wait -n          # wait for any one job to finish, then launch the next
      active=$(( active - 1 ))
    fi
  done
  wait                 # wait for the final batch
fi

echo "=== all ${TOTAL} runs complete; generating figures ==="
python plot_results.py --log-dir "${LOGDIR}" --out /workspace/figures_1M
python analyze_grid.py --log-dir "${LOGDIR}" --csv /workspace/figures_1M/summary_1M.csv
echo "=== done. See /workspace/figures_1M ==="
