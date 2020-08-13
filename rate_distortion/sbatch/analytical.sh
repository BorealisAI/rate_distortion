#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=30G
#SBATCH --gres=gpu:1
#SBATCH --time="-1"
#SBATCH --array=0%1

$CUDA_VISIBLE_DEVICES
list=(
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=icml_linear_vae100"
)
echo "Starting task $SLURM_ARRAY_TASK_ID: ${list[SLURM_ARRAY_TASK_ID]}"
eval ${list[SLURM_ARRAY_TASK_ID]}
