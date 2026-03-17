#!/bin/bash -l
#SBATCH --mail-user=msaee007@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name="walkability_"
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128g
#SBATCH --time=6-0:00:00


conda activate /rhome/msaee007/bigdata/conda_packages

# python resnet_baseline.py
python pointnet_exp.py
# python pointnet_hybrid.py