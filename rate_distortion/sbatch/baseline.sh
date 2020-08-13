#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=30G
#SBATCH --gres=gpu:1
#SBATCH --time="-1"
#SBATCH --array=0-5%6

$CUDA_VISIBLE_DEVICES
list=(
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=deep_gan2_GP_baseline_icml"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=deep_gan10_GP_baseline_icml"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=deep_gan100_GP_baseline_icml"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=vae2_rd_baseline_icml"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=vae10_rd_baseline_icml"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=vae100_rd_baseline_icml"
)
echo "Starting task $SLURM_ARRAY_TASK_ID: ${list[SLURM_ARRAY_TASK_ID]}"
eval ${list[SLURM_ARRAY_TASK_ID]}
