#!/bin/bash -l
#SBATCH --mail-user=msaee007@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name="pointnet_tune"
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=256g
#SBATCH --time=5-0:00:00


conda activate /rhome/msaee007/bigdata/conda_packages

python resnet_baseline.py synth
python resnet_baseline.py weather
