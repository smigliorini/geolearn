#!/bin/bash -l
#SBATCH --mail-user=msaee007@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name="poly2vectail"
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128g
#SBATCH --time=1-0:00:00


conda activate /rhome/msaee007/bigdata/conda_packages
export PATH=/rhome/msaee007/bigdata/conda_packages/bin/:$PATH

# pip install -r /rhome/msaee007/requirements.txt --upgrade --force-reinstall
# pip install triangle

# pip install triangle
# pip install torchmetrics
# python p1_poly2vec_transformer.py synth
# python p1_poly2vec_transformer.py weather
# python p3_poly2vec_transformer.py
# python p4_poly2vec_transformer.py
# python p2_poly2vec_transformer2.py
python p2_poly2vec_transformer_results2.py

# python p1_poly2vec_transformer.py synth
# python p1_poly2vec_transformer.py weather
# python p2_poly2vec_transformer.py
