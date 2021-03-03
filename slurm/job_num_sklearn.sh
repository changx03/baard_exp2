#!/bin/bash -e
#SBATCH --job-name=num_sklearn
#SBATCH --output=log/exp_num_sklearn_%a.out
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --array=0-9

module load Python/3.7.3-gimkl-2018b
source /nesi/project/uoa02977/baard_exp2/venv/bin/activate

python /nesi/project/uoa02977/baard_exp2/run/exp_num_sklearn.py -i $SLURM_ARRAY_TASK_ID
