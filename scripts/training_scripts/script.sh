#!/usr/bin/bash
#SBATCH --job-name CLD01
#SBATCH --account DD-23-91
#SBATCH --partition qgpu
#SBATCH --time 120:00:00
#SBATCH --nodes 1
#SBATCH --gpus 4

ml purge
ml OpenMPI/4.1.4-GCC-11.3.0

srun train_CLD01.sh

