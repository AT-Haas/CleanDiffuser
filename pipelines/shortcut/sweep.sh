#!/bin/bash
# SLURM array job: train Shortcut Flow planners across D4RL MuJoCo envs × seeds.
#
# Submit with:    sbatch pipelines/shortcut/sweep.sh
#
# Layout:    SLURM_ARRAY_TASK_ID ∈ {0, …, 11}
#            envs (3) × seeds (4)  →  12 jobs
#
# Tweak the SBATCH directives to match your cluster (partition name, walltime,
# memory). The module-load line below is for the typical Lmod CUDA module —
# change if your cluster uses a different naming scheme.

#SBATCH --job-name=shortcut_flow
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/shortcut_%A_%a.out
#SBATCH --error=logs/shortcut_%A_%a.err

set -euo pipefail

# ---- Cluster modules ----------------------------------------------------
module load cuda/12.1 cudnn/8.9 || true   # adjust to your cluster

# ---- Conda env ----------------------------------------------------------
# Make sure conda is on PATH (varies by cluster image).
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate cleandiffuser

# ---- Runtime environment ------------------------------------------------
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl                       # GPU offscreen rendering
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin"
export PYTHONHASHSEED=0                    # determinism for the data loader

# ---- Sweep grid ---------------------------------------------------------
ENVS=(hopper-medium-v2 halfcheetah-medium-v2 walker2d-medium-v2)
SEEDS=(0 1 2 3)

N_SEEDS=${#SEEDS[@]}
ENV_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))

ENV=${ENVS[$ENV_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "── array id $SLURM_ARRAY_TASK_ID  →  env=$ENV  seed=$SEED ──"

# ---- Run ----------------------------------------------------------------
cd "$HOME/projects/CleanDiffuser"

mkdir -p logs

python pipelines/shortcut/shortcut_d4rl_mujoco.py \
    task="$ENV" \
    seed="$SEED" \
    devices=1 \
    use_wandb=true \
    wandb_project=cleandiffuser-shortcut
