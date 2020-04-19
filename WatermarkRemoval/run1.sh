#!/bin/bash

#SBATCH -c2
#SBATCH --time=2-0
#SBATCH --mem-per-cpu=4096
#SBATCH --mail-type=BEGIN,END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jjgold@cs.huji.ac.il    # Where to send mail	

export PYTHONPATH=$PYTHONPATH:"$(dirname "$(pwd)")"/Marabou

python3 WatermarkVerification1.py --model mnist.w.wm --epsilon_max 0.5 --epsilon_interval 0.001

