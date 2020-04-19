#!/bin/bash

#SBATCH -c8
#SBATCH --time=10-0
#SBATCH --mem-per-cpu=4096
#SBATCH --mail-type=BEGIN,END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jjgold@cs.huji.ac.il    # Where to send mail	

export PYTHONPATH=$PYTHONPATH:"$(dirname "$(pwd)")"/Marabou

python3 SATexample.py --model ACASXU_2_9_all_corrected

