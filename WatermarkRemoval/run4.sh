#!/bin/bash

#SBATCH -c2
#SBATCH --time=7-0
#SBATCH --mem-per-cpu=4096
#SBATCH --mail-type=BEGIN,END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --array=0-99

export PYTHONPATH=$PYTHONPATH:"$(dirname "$(pwd)")"/Marabou

start=$(($SLURM_ARRAY_TASK_ID*10))
finish=$((start+9))

python3 WatermarkVerification4.py --model mnist.w.wm --num_of_inputs 5 --epsilon_max 300 --start $start --finish $finish

