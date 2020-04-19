import numpy as np
import os
import argparse
import utils
from copy import deepcopy
from maraboupy import MarabouUtils
from maraboupy.Marabou import createOptions
from WatermarkVerification1 import *
import MarabouNetworkTFWeightsAsVar
sat = 'SAT'
unsat = 'UNSAT'

class WatermarkVerification2(WatermarkVerification):

    def epsilonABS(self, network, epsilon_var):
        epsilon2 = network.getNewVariable()
        MarabouUtils.addEquality(network, [epsilon2, epsilon_var], [1, -2], 0)
        
        relu_epsilon2 = network.getNewVariable()
        network.addRelu(epsilon2, relu_epsilon2)
        
        abs_epsilon = network.getNewVariable()
        MarabouUtils.addEquality(network, [abs_epsilon, relu_epsilon2, epsilon_var], [1, -1, 1], 0)
        return abs_epsilon

    def evaluateSingleOutput(self, epsilon, network, prediction, output):
        outputVars = network.outputVars[0]
        abs_epsilons = list()
        for k in network.matMulLayers.keys():
            n, m = network.matMulLayers[k]['vals'].shape
            print(n,m)
            for i in range(n):
                for j in range(m):
                    if j in [prediction, output]:
                        epsilon_var = network.epsilons[i][j]
                        network.setUpperBound(epsilon_var, epsilon)
                        network.setLowerBound(epsilon_var, -epsilon)
                        abs_epsilon_var = self.epsilonABS(network, epsilon_var)
                        abs_epsilons.append(abs_epsilon_var)
                    else:
                        epsilon_var = network.epsilons[i][j]
                        network.setUpperBound(epsilon_var, 0)
                        network.setLowerBound(epsilon_var, 0)

        e = MarabouUtils.Equation(EquationType=MarabouUtils.MarabouCore.Equation.LE)
        for i in range(len(abs_epsilons)):
            e.addAddend(1, abs_epsilons[i])
        e.setScalar(epsilon)
        network.addEquation(e)

        MarabouUtils.addInequality(network, [outputVars[prediction], outputVars[output]], [1, -1], 0)
        return network.solve(verbose=True)


    def run(self, model_name, start, finish):
        filename = './ProtobufNetworks/last.layer.{}.pb'.format(model_name)
        
        lastlayer_inputs = np.load('./data/{}.lastlayer.input.npy'.format(model_name))
        predictions = np.load('./data/{}.prediction.npy'.format(model_name))
        # num_of_inputs_to_run = lastlayer_inputs.shape[0]
        epsilons_vals = np.array([])

        start = start if start > 0 else 0
        finish = finish if finish > 0 else (lastlayer_inputs.shape[0]-1)
        out_file = open('./data/results/problem2test/{}_{}-{}.csv'.format(model_name, start, finish), 'w')
        out_file.write('unsat-epsilon,sat-epsilon,original-prediction,sat-prediction\n')
        out_file.flush()
        for i in range(start, finish+1):
            
            prediction = np.argmax(predictions[i])
            inputVals = np.reshape(lastlayer_inputs[i], (1, lastlayer_inputs[i].shape[0]))
            network = MarabouNetworkTFWeightsAsVar.read_tf_weights_as_var(filename=filename, inputVals=inputVals)
            
            unsat_epsilon, sat_epsilon, sat_vals = self.findEpsilonInterval(network, prediction)
            out_file.write('{},{},{},{}\n'.format(unsat_epsilon, sat_epsilon, prediction, sat_vals[2]))
            out_file.flush()
        
            all_vals = sat_vals[1][0]
            epsilons_vars = network.matMulLayers[0]['epsilons']
            newVals = np.array([[all_vals[epsilons_vars[j][i]] for i in range(epsilons_vars.shape[1])] for j in range(epsilons_vars.shape[0])])
            newVals = np.reshape(newVals, (1, newVals.shape[0], newVals.shape[1]))
            epsilons_vals = newVals if epsilons_vals.size==0 else np.append(epsilons_vals, newVals, axis=0)
        
        out_file.close()
        np.save('./data/results/problem2test/{}_{}-{}.vals'.format(model_name, start, finish), epsilons_vals)


    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='the name of the model')
    parser.add_argument('--epsilon_max', default=100, help='max epsilon value')
    parser.add_argument('--epsilon_interval', default=0.01, help='epsilon smallest change')
    parser.add_argument('--start', default=-1, help='max epsilon value')
    parser.add_argument('--finish', default=-1, help='epsilon smallest change')
    
    args = parser.parse_args()
    epsilon_max = float(args.epsilon_max)
    epsilon_interval = float(args.epsilon_interval)  
    start = int(args.start)  
    finish = int(args.finish)  

    model_name = args.model
    MODELS_PATH = './Models'
    problem = WatermarkVerification2(epsilon_max, epsilon_interval)
    problem.run(model_name, start, finish)