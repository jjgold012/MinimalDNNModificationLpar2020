import sys
sys.path.append('../')
import numpy as np
import argparse
from maraboupy import MarabouUtils
from maraboupy import MarabouCore
from maraboupy import Marabou
from gurobipy import *
from WatermarkVerification import MarabouNetworkTFWeightsAsVar2
from functools import reduce
# from gurobipy import *
from copy import deepcopy
from pprint import pprint

sat = 'SAT'
unsat = 'UNSAT'
class findCorrection:

    def __init__(self, epsilon_max, epsilon_interval, correct_diff, lp):
        self.epsilon_max = epsilon_max
        self.epsilon_interval = epsilon_interval
        self.correct_diff = correct_diff
        self.lp = lp

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
        # epsilons_vals = np.array([[modelVars[networkEpsilons[i][j]].x for j in range(epsilonsShape[1])] for i in range(epsilonsShape[0])])
        all_vals = np.array([modelVars[i].x for i in range(numOfVar)])
        return epsilon.x, epsilon.x, all_vals 

    def findEpsilon(self, network):
        outputVars = network.outputVars
        
        for i in range(outputVars.shape[0]):
            MarabouUtils.addInequality(network, [outputVars[i][0], outputVars[i][2]], [1, -1], self.correct_diff)
            MarabouUtils.addInequality(network, [outputVars[i][0], outputVars[i][3]], [1, -1], self.correct_diff)
            MarabouUtils.addInequality(network, [outputVars[i][0], outputVars[i][4]], [1, -1], self.correct_diff)
            MarabouUtils.addInequality(network, [outputVars[i][1], outputVars[i][2]], [1, -1], self.correct_diff)
            MarabouUtils.addInequality(network, [outputVars[i][1], outputVars[i][3]], [1, -1], self.correct_diff)
            MarabouUtils.addInequality(network, [outputVars[i][1], outputVars[i][4]], [1, -1], self.correct_diff)
                
            # MarabouUtils.addInequality(network, [outputVars[i][outputNum], outputVars[i][2]], [1, -1], self.correct_diff)
            # MarabouUtils.addInequality(network, [outputVars[i][outputNum], outputVars[i][3]], [1, -1], self.correct_diff)
            # MarabouUtils.addInequality(network, [outputVars[i][outputNum], outputVars[i][4]], [1, -1], self.correct_diff)
        return self.getNetworkSolution(network)
    


    def epsilonABS(self, network, epsilon_var):
        epsilon2 = network.getNewVariable()
        MarabouUtils.addEquality(network, [epsilon2, epsilon_var], [1, -2], 0)
        
        relu_epsilon2 = network.getNewVariable()
        network.addRelu(epsilon2, relu_epsilon2)
        
        abs_epsilon = network.getNewVariable()
        MarabouUtils.addEquality(network, [abs_epsilon, relu_epsilon2, epsilon_var], [1, -1, 1], 0)
        return abs_epsilon

    def evaluateEpsilon(self, epsilon, network):
        # for outputNum in [0, 1]:
        outputVars = network.outputVars
        abs_epsilons = list()
        n, m = network.epsilons.shape
        print(n,m)
        for i in range(n):
            for j in range(m):
                epsilon_var = network.epsilons[i][j]
                network.setUpperBound(epsilon_var, epsilon)
                network.setLowerBound(epsilon_var, -epsilon)
                abs_epsilon_var = self.epsilonABS(network, epsilon_var)
                abs_epsilons.append(abs_epsilon_var)
                    
        e = MarabouUtils.Equation(EquationType=MarabouUtils.MarabouCore.Equation.LE)
        for i in range(len(abs_epsilons)):
            e.addAddend(1, abs_epsilons[i])
        e.setScalar(epsilon)
        network.addEquation(e)
        for i in range(outputVars.shape[0]):
            MarabouUtils.addInequality(network, [outputVars[i][0], outputVars[i][2]], [1, -1], self.correct_diff)
            MarabouUtils.addInequality(network, [outputVars[i][0], outputVars[i][3]], [1, -1], self.correct_diff)
            MarabouUtils.addInequality(network, [outputVars[i][0], outputVars[i][4]], [1, -1], self.correct_diff)
            MarabouUtils.addInequality(network, [outputVars[i][1], outputVars[i][2]], [1, -1], self.correct_diff)
            MarabouUtils.addInequality(network, [outputVars[i][1], outputVars[i][3]], [1, -1], self.correct_diff)
            MarabouUtils.addInequality(network, [outputVars[i][1], outputVars[i][4]], [1, -1], self.correct_diff)
            # MarabouUtils.addInequality(network, [outputVars[i][outputNum], outputVars[i][2]], [1, -1], self.correct_diff)
            # MarabouUtils.addInequality(network, [outputVars[i][outputNum], outputVars[i][3]], [1, -1], self.correct_diff)
            # MarabouUtils.addInequality(network, [outputVars[i][outputNum], outputVars[i][4]], [1, -1], self.correct_diff)
        vals = network.solve(verbose=True)
        if vals[0]:
            return sat, vals
        else:
            return unsat, vals
    
    def findEpsilonInterval(self, network):
        sat_epsilon = self.epsilon_max
        unsat_epsilon = 0.0
        sat_vals = None
        epsilon = sat_epsilon
        while abs(sat_epsilon - unsat_epsilon) > self.epsilon_interval:
            status, vals = self.evaluateEpsilon(epsilon, deepcopy(network))
            if status == sat:
                sat_epsilon = epsilon
                sat_vals = vals[0]
            else:
                unsat_epsilon = epsilon
            epsilon = (sat_epsilon + unsat_epsilon)/2
        return unsat_epsilon, sat_epsilon , sat_vals


    def run(self, model_name, num):
        filename = './ProtobufNetworks/last.layer.{}.pb'.format(model_name)
        orig_model_name = 'ACASXU_2_9'
        lastlayer_inputs = np.load('./data/{}.lastlayer.input.npy'.format(orig_model_name))
        if num >= 0:
            lastlayer_inputs = lastlayer_inputs[:num]
        network = MarabouNetworkTFWeightsAsVar2.read_tf_weights_as_var(filename=filename, inputVals=lastlayer_inputs)
        unsat_epsilon, sat_epsilon, sat_vals = self.findEpsilon(network) if self.lp else self.findEpsilonInterval(network)
        predictions = np.load('./data/{}.prediction.npy'.format(model_name))
        prediction = np.argmin(predictions, axis=1)
        if num >= 0:
            predictions = predictions[:num]
            prediction = np.argmin(predictions, axis=0)
        
        num = num if num >= 0 else 'all'
        
        outFile = open('./data/{}_0to{}_lp.txt'.format(model_name, num-1), 'w') if self.lp else open('./data/{}_{}.txt'.format(model_name, num), 'w')
        print('Prediction vector:', file=outFile)
        print(predictions, file=outFile)
        print('\nPrediction vector min:', file=outFile)
        print(prediction, file=outFile)
        print('\n(unsat_epsilon, sat_epsilon)', file=outFile)
        print('({},{})'.format(unsat_epsilon, sat_epsilon), file=outFile)
        output_vars = network.outputVars
        output_vals = np.array([[sat_vals[output_vars[j][i]] for i in range(output_vars.shape[1])] for j in range(output_vars.shape[0])])    
        print('\nOutput vector:', file=outFile)
        print(output_vals, file=outFile)
        print('\nOutput vector min:', file=outFile)
        print(np.argmin(output_vals, axis=1), file=outFile)

        epsilons_vars = network.epsilons
        epsilons_vals = np.array([[sat_vals[epsilons_vars[j][i]] for i in range(epsilons_vars.shape[1])] for j in range(epsilons_vars.shape[0])])    
        
        if self.lp:
            np.save('./data/{}_0to{}3_lp.vals'.format(model_name, num-1), epsilons_vals)
        else:
            np.save('./data/{}_0to{}4.vals'.format(model_name, num-1), epsilons_vals)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='ACASXU_2_9', help='the name of the model')
    parser.add_argument('--input_num', default=-1, help='the name of the model')
    parser.add_argument('--correct_diff', default=0.001, help='the input to correct')
    parser.add_argument('--epsilon_max', default=5, help='max epsilon value')
    parser.add_argument('--epsilon_interval', default=0.0001, help='epsilon smallest change')
    parser.add_argument('--lp', action='store_true', help='solve lp')
    
    args = parser.parse_args()
    epsilon_max = float(args.epsilon_max)
    epsilon_interval = float(args.epsilon_interval)  
    correct_diff = - float(args.correct_diff)  
    input_num = int(args.input_num)  
    
    model_name = args.model
    MODELS_PATH = './Models'
    problem = findCorrection(epsilon_max, epsilon_interval, correct_diff, args.lp)
    problem.run(model_name, input_num)