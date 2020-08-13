#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --gres=gpu:1
#SBATCH --time="-1"
#SBATCH --array=0-12%13

$CUDA_VISIBLE_DEVICES
list=(
"cd ~/00sync/rate_distortion && python -m rate_distortion.plot --hparam_set=GANs_icml_rerun"
"cd ~/00sync/rate_distortion && python -m rate_distortion.plot --hparam_set=GAN_BDMC_icml_rerun"
"cd ~/00sync/rate_distortion && python -m rate_distortion.plot --hparam_set=VAE_BDMC_icml_rerun"
"cd ~/00sync/rate_distortion && python -m rate_distortion.plot --hparam_set=AAE_BDMC_icml_rerun"
"cd ~/00sync/rate_distortion && python -m rate_distortion.plot --hparam_set=GAN_VAE_AAE_icml_rerun"
"cd ~/00sync/rate_distortion && python -m rate_distortion.plot --hparam_set=blurry_VAE_icml_rerun"
"cd ~/00sync/rate_distortion && python -m rate_distortion.plot --hparam_set=mixture_icml_rerun"
"cd ~/00sync/rate_distortion && python -m rate_distortion.plot --hparam_set=CIFAR10_icml_rerun"
"cd ~/00sync/rate_distortion && python -m rate_distortion.plot --hparam_set=GANs_fid_icml_rerun"
"cd ~/00sync/rate_distortion && python -m rate_distortion.plot --hparam_set=GVA_fid_icml_rerun"
"cd ~/00sync/rate_distortion && python -m rate_distortion.plot --hparam_set=GAN_baseline_icml_rerun"
"cd ~/00sync/rate_distortion && python -m rate_distortion.plot --hparam_set=VAE_baseline_icml_rerun"
"cd ~/00sync/rate_distortion && python -m rate_distortion.plot --hparam_set=icml_random_seeds_rerun"
)
echo "Starting task $SLURM_ARRAY_TASK_ID: ${list[SLURM_ARRAY_TASK_ID]}"
eval ${list[SLURM_ARRAY_TASK_ID]}
