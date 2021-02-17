#!/bin/bash -e
#SBATCH --job-name=baard_exp1
#SBATCH --output=log/%x_%j.out
#SBATCH --time=10:00:00
#SBATCH --mem=8G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load CUDA
module load Python/3.7.3-gimkl-2018b
source /nesi/project/uoa02977/baard_exp2/venv/bin/activate

python ./run/sklearn_attack_against_baard.py -d banknote -m svm -i 0 -a fgsm -e 0.3 1.0
