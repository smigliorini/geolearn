#!/bin/bash -l
#SBATCH --mail-user=msaee007@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name="cluster_with_stdbscan"
#SBATCH -p epyc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64g
#SBATCH --time=7-0:15:00

conda activate /rhome/msaee007/bigdata/conda_packages

python stdbscan_processor.py
