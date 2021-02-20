#!/bin/bash -e
#SBATCH --job-name=cifar10_resnet
#SBATCH --output=log/exp_cifar10_resnet_%a.out
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4

module load CUDA
module load Python/3.7.3-gimkl-2018b
source /nesi/project/uoa02977/baard_exp2/venv/bin/activate

python /nesi/project/uoa02977/baard_exp2/run/exp_cifar10_resnet.py -i $SLURM_ARRAY_TASK_ID
