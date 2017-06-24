#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --time=8:00:00
#SBATCH --partition=normal
#SBATCH --output=sbatch_output.txt

module load daint-gpu
module load cudatoolkit

pwd

./run_square_perf.sh TSMTTSM DR 64 64
