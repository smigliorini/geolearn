#!/bin/bash -l
#SBATCH --mail-user=msaee007@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name="synth_gen"
#SBATCH -p epyc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128g
#SBATCH --time=6-0:15:00


export SPARK_HOME=/rhome/msaee007/bigdata/spark_folder/spark-3.5.3-bin-hadoop3
export BEAST_HOME=/rhome/msaee007/bigdata/spark_folder/beast-0.10.1-RC2
export PATH=$PATH:$SPARK_HOME/bin:$BEAST_HOME/bin

conda activate /rhome/msaee007/bigdata/conda_packages
export PATH=/rhome/msaee007/bigdata/conda_packages/bin/:$PATH

# spark warmup 
# beast summary /rhome/msaee007/bigdata/pointnet_data/synthetic_data/points_tmp.csv 'iformat:point(x,y)' -skipheader separator:, -boxcounting

# generate data
# python generator_main.py uniform 25
# python generator_main.py gaussian 25
# python generator_main.py diagonal 3
# python generator_main.py sierpinski 25
# python generator_main.py bit 17
# python generator_main.py parcel 9
# python normalization_and_testval_split.py
# python label_weather_data.py
# python runtime_summary.py
python p1_true_values_summary.py

