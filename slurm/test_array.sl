#!/bin/bash -e
#SBATCH --job-name=test_array
#SBATCH --output=log/test_array_%a.out
#SBATCH --time=00:01:00
#SBATCH --mem=1G
#SBATCH --array=1-3

echo $SLURM_ARRAY_TASK_ID
