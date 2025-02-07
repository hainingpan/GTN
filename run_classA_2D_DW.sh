#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --exclude=cuda00[1-8],gpuc00[1-2],pascal0[01-10],gpu00[5-8],gpu0[10-14],gpu0[17-18],volta0[01-03]
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --mem=8000
#SBATCH --cpus-per-task=1
#SBATCH --output=L20.out
#SBATCH --error=L20.err

 
cd $PWD

module purge

module load singularity

srun singularity exec --nv /scratch/hp636/pytorch.sif python run_classA_2D_DW.py --L 10 --nshell 2 --tf 10 --seed0 0  --es 20 --truncate
