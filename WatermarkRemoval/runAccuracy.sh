#!/bin/bash

#SBATCH -c2
#SBATCH --time=7-0
#SBATCH --mem-per-cpu=4096
#SBATCH --mail-type=BEGIN,END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jjgold@cs.huji.ac.il    # Where to send mail	

export PYTHONPATH=$PYTHONPATH:"$(dirname "$(pwd)")"/Marabou

python3 accuracyTest.py
