# WatermarkVerification

### Requirements
- Python 3
- tensorflow (`pip3 install tensorflow`)
- numpy (`pip3 install numpy`)
- Marabou repository (`git clone https://github.com/guykatzz/Marabou.git`)
- nn-verification repository (`git clone https://github.com/adiyoss/nn-verification.git`)

## Setup
**Make sure all three repositories are in the same folder.**
- Install Marabou
    `cd Marabou`
    `make`
    `cd maraboupy`
    `make`
    `export PYTHONPATH=$PYTHONPATH:$HOME/<path-to-marabou-repo>`
- Install maraboupy as instracted [here](https://github.com/guykatzz/Marabou/tree/master/maraboupy)
- Create a neural net to verify as instracted [here](https://github.com/adiyoss/nn-verification) 

## Run
`python WatermarkVerification1.py --model <Model name>` 
For example: `python WatermarkVerification1.py --model mnist.w.wm` 