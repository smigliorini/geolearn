#!/bin/bash -l
#SBATCH --mail-user=msaee007@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name="syno_hybrid"
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=256g
#SBATCH --time=5-0:00:00


conda activate /rhome/msaee007/bigdata/conda_packages
export PATH=/rhome/msaee007/bigdata/conda_packages/bin/:$PATH

# python resnet_train.exp synth
# python resnet_train.exp weather
# python pointnet_train.py
python p1_pred_summary.py synth pointnet
python p1_pred_summary.py synth resnet
# python p1_pred_summary.py weather pointnet
# python p1_pred_summary.py weather resnet
# python scalability_test.py
python p1_tables.py