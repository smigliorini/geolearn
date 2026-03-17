#!/bin/bash -l
#SBATCH --mail-user=msaee007@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name="hybrid_selectivity"
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=256g
#SBATCH --time=5-0:00:00


conda activate /rhome/msaee007/bigdata/conda_packages
export PATH=/rhome/msaee007/bigdata/conda_packages/bin/:$PATH

python resnet_baseline.py
python pointnet_main_exp.py
# python pointnet_hybrid_exp.py
