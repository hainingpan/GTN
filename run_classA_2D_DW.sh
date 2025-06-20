#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --exclude=cuda00[1-8],gpuc00[1-2],pascal0[01-10],gpu00[5-8],gpu0[10-14],gpu0[17-18],volta0[01-03]
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --mem=8000
#SBATCH --cpus-per-task=1
#SBATCH --output=L30_truncate.out
#SBATCH --error=L30_truncate.err

 
cd $PWD

module purge

module load singularity

# srun singularity exec --nv /scratch/hp636/pytorch.sif python run_classA_2D_DW.py --L 20 --nshell 2 --tf 2 --seed0 1  --es 10 --truncate
# srun singularity exec --nv /scratch/hp636/pytorch.sif python run_classA_2D_DW.py --L $L --nshell $nshell --tf 2 --seed0 $seed  --es 10 --truncate
# srun singularity exec --nv /scratch/hp636/pytorch.sif python run_classA_2D_DW.py --L $L --nshell $nshell --tf 10 --seed0 $seed  --es 1 --truncate
# srun singularity exec --nv /scratch/hp636/pytorch.sif python run_classA_2D_DW.py --L 30 --nshell 2 --tf 2 --seed0 0  --es 1 --truncate
srun singularity exec --nv /scratch/hp636/pytorch.sif python run_classA_2D_DW.py --L 30 --nshell 2 --tf 1 --seed0 1  --es 4 --truncate


