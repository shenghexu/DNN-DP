#!/bin/bash
#
#SBATCH --job-name=TST
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:p40:1
#SBATCH --mem=32GB
#SBATCH --time=5-20
#SBATCH --output=TS_%A.out
#SBATCH --mail-type=END
#SBATCH --mail-user=sx475@nyu.edu



module purge
module load python3/intel/3.6.3
source ~/pyenv/py3.6.3/bin/activate



echo
echo "Hostname: $(hostname)"
echo

./Train_TSP_20.sh
