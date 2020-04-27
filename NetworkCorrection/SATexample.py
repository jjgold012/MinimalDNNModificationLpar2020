import sys
import warnings
import numpy as np
from maraboupy import MarabouUtils
from maraboupy import Marabou
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ACASXU_2_9', help='the name of the model')
args = parser.parse_args()

model_name = args.model
file_name = './ProtobufNetworks/{}.pb'.format(model_name)

net1 = Marabou.read_tf(file_name)


inputVars = net1.inputVars[0][0]

# Bounds for input 0: [ -0.3284228772, 0.6798577687 ]
# Bounds for input 1: [ -0.5000000551, 0.5000000551 ]
# Bounds for input 2: [ -0.5000000551, 0.5000000551 ]
# Bounds for input 3: [ -0.5000000000, 0.5000000000 ]
# Bounds for input 4: [ -0.5000000000, 0.5000000000 ]
net1.setLowerBound(inputVars[0], -0.3284228772)
net1.setUpperBound(inputVars[0], 0.6798577687)
net1.setLowerBound(inputVars[1], -0.5)
net1.setUpperBound(inputVars[1], 0.5)
net1.setLowerBound(inputVars[2], -0.5)
net1.setUpperBound(inputVars[2], 0.5)
net1.setLowerBound(inputVars[3], -0.5)
net1.setUpperBound(inputVars[3], 0.5)
net1.setLowerBound(inputVars[4], -0.5)
net1.setUpperBound(inputVars[4], 0.5)

outputVars = net1.outputVars[0]

# property: output 3 is minimal
MarabouUtils.addInequality(net1, [outputVars[3], outputVars[0]], [1, -1], 0)
MarabouUtils.addInequality(net1, [outputVars[3], outputVars[1]], [1, -1], 0)
MarabouUtils.addInequality(net1, [outputVars[3], outputVars[2]], [1, -1], 0)
MarabouUtils.addInequality(net1, [outputVars[3], outputVars[4]], [1, -1], 0)

options = Marabou.createOptions(dnc=True, verbosity=0, initialDivides=2)
vals, stats = net1.solve(options=options)

if vals:
    print('SAT')
    out_file = open('./data/{}_input.csv'.format(model_name), 'w')
    out_file.write('{},{},{},{},{}\n'.format(vals[inputVars[0]],
                                            vals[inputVars[1]],
                                            vals[inputVars[2]],
                                            vals[inputVars[3]],
                                            vals[inputVars[4]]))
    out_file.close()

    out_file = open('./data/{}_output.csv'.format(model_name), 'w')
    out_file.write('{},{},{},{},{}\n'.format(vals[outputVars[0]],
                                            vals[outputVars[1]],
                                            vals[outputVars[2]],
                                            vals[outputVars[3]],
                                            vals[outputVars[4]]))
    out_file.close()
else:
    print('{} UNSAT'.format(model_name))

