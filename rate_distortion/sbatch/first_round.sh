#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=30G
#SBATCH --gres=gpu:1
#SBATCH --time="-1"
#SBATCH --array=0-43%10

$CUDA_VISIBLE_DEVICES
list=(
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=icml_CIFAR10_dcgan100_V"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=icml_aae2"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=icml_aae10"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=icml_aae100"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=icml_vae10_mp_010"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=icml_vae10_mp_020"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=icml_vae10_mp_050"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=icml_vae10_mp_080"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=icml_vae10_mp_090"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=icml_vae10_mp_099"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=icml_vae10_blur5"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=icml_vae10_blur2"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=icml_vae10_blur1"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=icml_vae10_blur01"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=icml_vae10_blur05"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=icml_deep_gan5_GP"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=icml_deep_gan10_GP"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=icml_deep_gan2_GP"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=icml_shallow_gan5_GP"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=icml_shallow_gan10_GP"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=icml_shallow_gan2_GP"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=icml_shallow_gan100_GP"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=icml_deep_gan100_GP"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=fid_icml_aae2"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=fid_icml_aae10"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=fid_icml_deep_gan5_GP"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=fid_icml_deep_gan10_GP"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=fid_icml_deep_gan2_GP"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=fid_icml_shallow_gan5_GP"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=fid_icml_shallow_gan10_GP"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=fid_icml_shallow_gan2_GP"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=fid_icml_vae10_rd"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=fid_icml_vae2_rd"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=icml_CIFAR10_dcgan5_BRE"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=icml_CIFAR10_dcgan10_BRE"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=icml_CIFAR10_dcgan100_BRE"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=icml_CIFAR10_dcgan5_SN"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=icml_CIFAR10_dcgan10_SN"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=icml_CIFAR10_dcgan100_SN"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=icml_CIFAR10_dcgan5_GP"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=icml_CIFAR10_dcgan10_GP"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=icml_CIFAR10_dcgan100_GP"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=icml_CIFAR10_dcgan5_V"
"cd ~/00sync/rate_distortion && python -m rate_distortion.main --hparam_set=icml_CIFAR10_dcgan10_V"

)
echo "Starting task $SLURM_ARRAY_TASK_ID: ${list[SLURM_ARRAY_TASK_ID]}"
eval ${list[SLURM_ARRAY_TASK_ID]}
