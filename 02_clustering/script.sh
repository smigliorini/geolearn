#!/bin/bash -l
#SBATCH --mail-user=msaee007@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name="pointnet_clustering_para"
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=256g
#SBATCH --time=6-0:15:00


conda activate /rhome/msaee007/bigdata/conda_packages

# python pointnet_segmentation.py
# python pointnet_segmentation2.py
# python unet_segmentation.py
# python unet_segmentation2.py
python results_summary_pn.py
python results_summary_un.py
python results_summary_un2.py