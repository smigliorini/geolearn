#!/bin/bash -l
#SBATCH --mail-user=msaee007@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name="unet_weather_clustering"
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=256g
#SBATCH --time=7-0:00:00


conda activate /rhome/msaee007/bigdata/conda_packages

python unet_segmentation2.py
python unet_segmentation.py