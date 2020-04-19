#!/bin/bash
#SBATCH --time=7-0

export PYTHONPATH=$PYTHONPATH:"$(dirname "$(pwd)")"/Marabou

python3 WatermarkVerification3gurobi.py --model mnist.w.wm --num_of_inputs 50
python3 WatermarkVerification3gurobi.py --model mnist.w.wm --num_of_inputs 75
# python3 WatermarkVerification2.py --model test --epsilon_max 100
