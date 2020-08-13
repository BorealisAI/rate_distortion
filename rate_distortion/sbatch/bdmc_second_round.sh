#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=30G
#SBATCH --gres=gpu:1
#SBATCH --time="-1"
#SBATCH --array=0-8%9

$CUDA_VISIBLE_DEVICES
list=(
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=BDMC_icml_aae2_rerun"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=BDMC_icml_aae10_rerun"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=BDMC_icml_aae100_rerun"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=BDMC_icml_deep_gan2_GP_rerun"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=BDMC_icml_deep_gan10_GP_rerun"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=BDMC_icml_deep_gan100_GP_rerun"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=BDMC_icml_vae2_rd_rerun"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=BDMC_icml_vae10_rd_rerun"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=BDMC_icml_vae100_rerun"
)
echo "Starting task $SLURM_ARRAY_TASK_ID: ${list[SLURM_ARRAY_TASK_ID]}"
echo $CUDA_VISIBLE_DEVICES
eval ${list[SLURM_ARRAY_TASK_ID]}
