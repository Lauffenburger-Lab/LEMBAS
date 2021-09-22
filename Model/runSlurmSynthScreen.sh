#!/bin/bash
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --mail-user=avlant@mit.edu
#SBATCH --array=0-14
#############################################

# Load module
module load python3/3.6.4

python3 synthNetDataScreen.py --selectedCondition $SLURM_ARRAY_TASK_ID