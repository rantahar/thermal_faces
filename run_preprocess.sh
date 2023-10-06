#!/bin/bash -l
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --mem=32G

module load miniconda cuda
source activate thermal

python ./process_data.py

