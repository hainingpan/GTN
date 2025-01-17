#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
##SBATCH --exclude=cuda00[1-8],gpuc00[1-2],pascal0[01-10],gpu00[5-8],gpu0[10-14],gpu0[17-18],volta0[01-03]
#SBATCH --exclude=cuda00[1-8],gpuc00[1-2],pascal0[01-10],volta0[01-03],gpu00[5-6]
#SBATCH --time=7:20:00
#SBATCH --ntasks=1
#SBATCH --mem=8000
#SBATCH --cpus-per-task=1
#SBATCH --output=ARRARIDX.out
#SBATCH --error=ARRARIDX.err

 
cd $PWD

module purge

module load singularity

PARAMS_FILE="$PWD/params.txt"
# read -r  L mu nshell<<< $(sed -n "ARRARIDXp" $PARAMS_FILE)
read -r  L mu nshell sigma<<< $(sed -n "ARRARIDXp" $PARAMS_FILE)

# srun singularity exec --nv /scratch/hp636/pytorch.sif python run_classA_2D_EE.py --L $L --nshell $nshell --mu $mu --es 50

# normal script (most common)
srun singularity exec --nv /scratch/hp636/pytorch.sif python run_classA_2D_EE.py --L $L --nshell $nshell --mu $mu --es 50 --sigma $sigma --seed0 0 --tf 2

# to be merge with es -50
# srun singularity exec --nv /scratch/hp636/pytorch.sif python run_classA_2D_EE.py --L $L --nshell $nshell --mu $mu --es 250 --sigma $sigma --seed0 50

# Chern average ensemble
# srun singularity exec --nv /scratch/hp636/pytorch.sif python run_classA_2D_ensemble_ave.py --L $L --nshell $nshell --mu $mu --es 50 --sigma $sigma --seed0 0

# To compute L log L
# srun singularity exec --nv /scratch/hp636/pytorch.sif python run_classA_2D_EE2.py --L $L --nshell $nshell --mu $mu --es 200 --sigma 0

# srun singularity exec --nv /scratch/hp636/pytorch.sif python run_classA_2D_EE_mp.py --L $L --nshell $nshell --mu $mu --es 50