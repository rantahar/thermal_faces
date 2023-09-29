#!/bin/bash -l
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

module load miniconda cuda
source activate thermal

python ./subsections.py

