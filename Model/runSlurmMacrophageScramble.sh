#!/bin/bash
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --mail-user=avlant@mit.edu
#SBATCH --array=0-19
#SBATCH --nice=100000
#SBATCH --exclude=c[5-22]
#############################################

# Load module
module load python3/3.6.4

python3 macrophageNetCrossValidationScramble.py --leaveOut $SLURM_ARRAY_TASK_ID