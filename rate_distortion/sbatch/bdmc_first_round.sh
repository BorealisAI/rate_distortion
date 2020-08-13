#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=30G
#SBATCH --gres=gpu:1
#SBATCH --time="-1"
#SBATCH --array=0-8%9

$CUDA_VISIBLE_DEVICES
list=(
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=BDMC_icml_aae2"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=BDMC_icml_aae10"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=BDMC_icml_aae100"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=BDMC_icml_deep_gan2_GP"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=BDMC_icml_deep_gan10_GP"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=BDMC_icml_deep_gan100_GP"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=BDMC_icml_vae2_rd"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=BDMC_icml_vae10_rd"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=BDMC_icml_vae100"
)
echo "Starting task $SLURM_ARRAY_TASK_ID: ${list[SLURM_ARRAY_TASK_ID]}"
echo $CUDA_VISIBLE_DEVICES
eval ${list[SLURM_ARRAY_TASK_ID]}
