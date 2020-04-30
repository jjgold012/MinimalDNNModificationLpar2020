import numpy as np
import os
import argparse
from pprint import pprint
from itertools import chain
from copy import deepcopy
from maraboupy import Marabou
from maraboupy import MarabouUtils
from maraboupy import MarabouCore
from time import process_time 
from MarabouNetworkWeightsVars import read_tf_weights_as_var
sat = 'SAT'
unsat = 'UNSAT'

class WatermarkRemoval:
    def __init__(self, epsilon_max, epsilon_interval):
        self.epsilon_max = epsilon_max
        self.epsilon_interval = epsilon_interval

    def findEpsilonInterval(self, network, prediction):
        sat_epsilon = self.epsilon_max
        unsat_epsilon = 0.0
        sat_vals = None
        epsilon = sat_epsilon
        while abs(sat_epsilon - unsat_epsilon) > self.epsilon_interval:
            status, vals, out = self.evaluateEpsilon(epsilon, deepcopy(network), prediction)
            if status == sat:
                sat_epsilon = epsilon
                sat_vals = (status, vals, out)
            else:
                unsat_epsilon = epsilon
            epsilon = (sat_epsilon + unsat_epsilon)/2
        return unsat_epsilon, sat_epsilon , sat_vals


    def epsilonABS(self, network, epsilon_var):
        epsilon2 = network.getNewVariable()
        MarabouUtils.addEquality(network, [epsilon2, epsilon_var], [1, -2], 0)
        
        relu_epsilon2 = network.getNewVariable()
        network.addRelu(epsilon2, relu_epsilon2)
        
        abs_epsilon = network.getNewVariable()
        MarabouUtils.addEquality(network, [abs_epsilon, relu_epsilon2, epsilon_var], [1, -1, 1], 0)
        return abs_epsilon


    def evaluateEpsilon(self, epsilon, network, prediction):
        outputVars = network.outputVars
        abs_epsilons = list()
        preds = list()
        predIndices = np.flip(np.argsort(prediction, axis=1), axis=1)        
        for i in range(outputVars.shape[0]):
            preds.append((predIndices[i][0], predIndices[i][1]))
        n, m = network.epsilons.shape
        print(n,m)
        for i in range(n):
            for j in range(m):
                if j in list(chain.from_iterable(preds)):
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

        for i in range(outputVars.shape[0]):
            MarabouUtils.addInequality(network, [outputVars[i][preds[i][0]], outputVars[i][preds[i][1]]], [1, -1], 0)
        
        options = Marabou.createOptions(numWorkers=6, dnc=False)
        stats = network.solve(verbose=False, options=options)
        newOut = predIndices[:,1]
        if stats[0]:
            return sat, stats, newOut
        else:
            return unsat, stats, newOut

    def run(self, model_name, numOfInputs, start, finish):        
        filename = './ProtobufNetworks/last.layer.{}.pb'.format(model_name)
        lastlayer_inputs = np.load('./data/{}.lastlayer.input.npy'.format(model_name))
        predictions = np.load('./data/{}.prediction.npy'.format(model_name))
        random_samples = predictions if numOfInputs==1 else np.load('./data/random/{}.wm.1000.random_samples.npy'.format(numOfInputs))
        epsilons_vals = np.array([])

        start = start if start > 0 else 0
        finish = finish if finish > 0 else (random_samples.shape[0]-1)
        out_file = open('./data/results/nonLinear/{}.{}.wm_{}-{}.time.csv'.format(model_name, numOfInputs, start, finish), 'w')
        out_file.write('unsat-epsilon,sat-epsilon,original-prediction,second-best-prediction,time\n')

        for i in range(start, finish+1):
            lastlayer_input = lastlayer_inputs[i].reshape(1, lastlayer_inputs[i].shape[0]) if numOfInputs==1 else np.array([lastlayer_inputs[j] for j in random_samples[i]])
            prediction = predictions[i].reshape(1, predictions[i].shape[0]) if numOfInputs==1 else np.array([predictions[j] for j in random_samples[i]])
            # if numOfInputs==1:
            #     lastlayer_input = lastlayer_inputs[i].reshape(1, lastlayer_inputs[i].shape[0])
            #     prediction = predictions[i].reshape(1, predictions[i].shape[0])
            network = read_tf_weights_as_var(filename=filename, inputVals=lastlayer_input)
            t1 = process_time()
            unsat_epsilon, sat_epsilon, sat_vals = self.findEpsilonInterval(network, prediction)
            t = process_time() - t1
            predIndices = np.flip(np.argsort(prediction, axis=1), axis=1)
            oldPred = predIndices[:,0]
            secondPred = predIndices[:,1]
            out_file.write('{},{},"{}","{}",{}\n'.format(unsat_epsilon, sat_epsilon, oldPred, secondPred, t))
            out_file.flush()

            all_vals = sat_vals[1][0]
            epsilons_vars = network.epsilons
            newVals = np.array([[all_vals[epsilons_vars[j][i]] for i in range(epsilons_vars.shape[1])] for j in range(epsilons_vars.shape[0])])
            newVals = np.reshape(newVals, (1, newVals.shape[0], newVals.shape[1]))
            epsilons_vals = newVals if epsilons_vals.size==0 else np.append(epsilons_vals, newVals, axis=0)
        
        out_file.close()
        np.save('./data/results/nonLinear/{}.{}.wm_{}-{}.vals'.format(model_name, numOfInputs, start, finish), epsilons_vals)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='the name of the model')
    parser.add_argument('--epsilon_max', default=100, help='max epsilon value')
    parser.add_argument('--epsilon_interval', default=0.01, help='epsilon smallest change')
    parser.add_argument('--num_of_inputs', default=2, help='the number of inputs that needs to be falsify')
    parser.add_argument('--start', default=-1, help='start index')
    parser.add_argument('--finish', default=-1, help='finish index')
    args = parser.parse_args()
    epsilon_max = float(args.epsilon_max)
    epsilon_interval = float(args.epsilon_interval)  
    numOfInputs = int(args.num_of_inputs)
    model_name = args.model
    start = int(args.start)  
    finish = int(args.finish)  

    MODELS_PATH = './Models'
    problem = WatermarkRemoval(epsilon_max, epsilon_interval)
    problem.run(model_name, numOfInputs, start, finish)