#!/bin/bash -l
#SBATCH --mail-user=msaee007@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name="pointnet_tune"
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=256g
#SBATCH --time=1-0:15:00


conda activate /rhome/msaee007/bigdata/conda_packages

python parameter_tuning.py
