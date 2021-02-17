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

python /nesi/project/uoa02977/baard_exp2/experiments/train_classifier.py -d banknote -m dnn -i 0
