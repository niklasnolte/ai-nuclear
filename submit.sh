#!/usr/bin/bash

#SBATCH -p gpu
#SBATCH -c 1
#SBATCH -t 0-10:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --gpu-freq=high
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=trifinos@mit.edu

module load Anaconda3/5.0.1-fasrc02
module load cuda/11.4.2-fasrc01
module load cudnn/8.2.2.26_cuda11.4-fasrc01
source activate ai

export PYTHONUNBUFFERED=TRUE

python ai-nuclear.py
