#!/bin/bash -l
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

module load miniconda cuda
source activate thermal

python train_subsection_model.py --units=32 --negatives=5 --region_size=32 --data_path=data_both --save_path=saved_both --num_epochs=10000
