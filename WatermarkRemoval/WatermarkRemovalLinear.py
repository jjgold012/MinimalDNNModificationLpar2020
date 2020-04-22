import numpy as np
import os
import argparse
import utils
from pprint import pprint
from copy import deepcopy
from maraboupy import Marabou
from maraboupy import MarabouUtils
from maraboupy import MarabouCore
from tensorflow import keras
from functools import reduce
from gurobipy import *
from time import process_time 
from WatermarkVerification1 import *
import MarabouNetworkWeightsVars
sat = 'SAT'
unsat = 'UNSAT'

class WatermarkRmovalLinear:

    def getNetworkSolution(self, network):
        equations = network.equList
        numOfVar = network.numVars
        networkEpsilons = network.epsilons
        epsilonsShape = networkEpsilons.shape 
        model = Model("my model")
        modelVars = model.addVars(numOfVar, lb=-GRB.INFINITY, ub=GRB.INFINITY)
        epsilon = model.addVar(name="epsilon")
        model.setObjective(epsilon, GRB.MINIMIZE)
        for i in range(epsilonsShape[0]):
            for j in range(epsilonsShape[1]):
                model.addConstr(modelVars[networkEpsilons[i][j]], GRB.LESS_EQUAL, epsilon)
                model.addConstr(modelVars[networkEpsilons[i][j]], GRB.GREATER_EQUAL, -1*epsilon)

        for eq in equations:
            addends = map(lambda addend: modelVars[addend[1]] * addend[0], eq.addendList)
            eq_left = reduce(lambda x,y: x+y, addends)
            if eq.EquationType == MarabouCore.Equation.EQ:
                model.addConstr(eq_left, GRB.EQUAL, eq.scalar)
            if eq.EquationType == MarabouCore.Equation.LE:
                model.addConstr(eq_left, GRB.LESS_EQUAL, eq.scalar)
            if eq.EquationType == MarabouCore.Equation.GE:
                model.addConstr(eq_left, GRB.GREATER_EQUAL, eq.scalar)
                
        model.optimize()
        epsilons_vals = np.array([[modelVars[networkEpsilons[i][j]].x for j in range(epsilonsShape[1])] for i in range(epsilonsShape[0])])
        all_vals = np.array([modelVars[i].x for i in range(numOfVar)])
        return epsilon.x, epsilons_vals, all_vals 

    def findEpsilon(self, network, prediction):
        outputVars = network.outputVars
        
        predIndices = np.flip(np.argsort(prediction, axis=1), axis=1)        
        for i in range(outputVars.shape[0]):
            maxPred = predIndices[i][0]
            secondMaxPred = predIndices[i][1]
            MarabouUtils.addInequality(network, [outputVars[i][maxPred], outputVars[i][secondMaxPred]], [1, -1], 0)
        results = self.getNetworkSolution(network)
        newOutput = np.array([[results[2][outputVars[i][j]] for j in range(outputVars.shape[1])] for i in range(outputVars.shape[0])])
        return results, predIndices[:,0], predIndices[:,1], newOutput
        
    def run(self, model_name, numOfInputs):        
        filename = './ProtobufNetworks/last.layer.{}.pb'.format(model_name)
        random_samples = np.load('./data/random/{}.wm.1000.random_samples.npy'.format(numOfInputs))
        lastlayer_inputs = np.load('./data/{}.lastlayer.input.npy'.format(model_name))
        predictions = np.load('./data/{}.prediction.npy'.format(model_name))
        epsilons_vals = np.array([])
        out_file = open('./data/results/linear/{}.{}.wm.time.csv'.format(model_name, numOfInputs), 'w')
        out_file.write('sat-epsilon,original-prediction,second-best-prediction, time\n')
        for i in range(random_samples.shape[0]):
            lastlayer_input = np.array([lastlayer_inputs[j] for j in random_samples[i]])  
            prediction = np.array([predictions[j] for j in random_samples[i]])  
            network = MarabouNetworkWeightsVars.read_tf_weights_as_var(filename=filename, inputVals=lastlayer_input)
            t1 = process_time()
            results, oldPred, secondPred, newOutput = self.findEpsilon(network, prediction)
            t = process_time() - t1
            out_file.write('{},"{}","{}",{}\n'.format(results[0], oldPred, secondPred, t))
            out_file.flush()

            newVals = np.reshape(results[1], (1, results[1].shape[0], results[1].shape[1]))
            epsilons_vals = newVals if epsilons_vals.size==0 else np.append(epsilons_vals, newVals, axis=0)
        
        out_file.close()
        # np.save('./data/results/linear/{}.{}.wm.vals'.format(model_name, numOfInputs), epsilons_vals)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='the name of the model')
    parser.add_argument('--num_of_inputs', default=2, help='the number of inputs that needs to be falsify')
    args = parser.parse_args()
    numOfInputs = int(args.num_of_inputs)
    model_name = args.model
    MODELS_PATH = './Models'
    problem = WatermarkRmovalLinear()
    problem.run(model_name, numOfInputs)