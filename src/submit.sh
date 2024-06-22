#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -J results4.py
#BSUB -W 01:30
#BSUB -o output_%J.out

source ~/venv/mice/bin/activate

python3 -u results4.py "31al" 2>&1
