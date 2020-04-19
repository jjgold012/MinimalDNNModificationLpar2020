import numpy as np
import tensorflow as tf
import os
import utils
from tensorflow import keras
from pprint import pprint
from csv import DictReader, DictWriter
from gurobipy import *

model_name = 'mnist.w.wm'
MODELS_PATH = './Models'

# epsilons = np.load('./data/results/problem4/{}.4.wm.vals.npy'.format(model_name))
# randomSamples = np.load('./data/random/4.wm.1000.random_samples.npy')
# wm_images = np.load('./data/wm.set.npy')
# wm_images = wm_images.reshape(wm_images.shape[0], wm_images.shape[1], wm_images.shape[2],1)

# for j in range(10):
#     net_model = utils.load_model(os.path.join(MODELS_PATH, model_name+'_model.json'), os.path.join(MODELS_PATH, model_name+'_model.h5'))

#     weights = net_model.get_weights()
#     weights[-1] = weights[-1] + epsilons[j]

#     net_model.set_weights(weights)
#     net_model.compile(optimizer=tf.train.AdamOptimizer(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    
#     predictions = np.load('./data/{}.prediction.npy'.format(model_name))
#     predIndices = np.flip(np.argsort(predictions, axis=1), axis=1)     
#     first = predIndices[:,0] 
#     second = predIndices[:,1]
#     newPred = net_model.predict(wm_images)
#     for i in randomSamples[j]:
#         print('original prediction: {}, new prediction: {}'.format(first[i], second[i]))
#         print(newPred[i])
#     c = np.array([newPred[i][first[i]] - newPred[i][second[i]] for i in randomSamples[j]])
#     print(c)


# datafile1 = open('./data/results/problem3/{}.1.wm.csv'.format(model_name))
# datafile2 = open('./data/results/problem2/{}.csv'.format(model_name))
# file_reader = DictReader(datafile1)
# sat_vals1 = np.array([float(line['sat-epsilon']) for line in file_reader])
# file_reader = DictReader(datafile2)
# sat_vals2 = np.array([float(line['sat-epsilon']) for line in file_reader])
# predictions = np.load('./data/{}.prediction.npy'.format(model_name))
# predIndices = np.flip(np.sort(predictions, axis=1), axis=1)     
# first = predIndices[:,0] 
# second = predIndices[:,1]
# diff = first-second
# a = np.argsort(diff)
# a = set(a[:40])
# b = np.argsort(sat_vals1)
# b = set(b[:40])
# c = np.argsort(sat_vals2)
# c = set(c[:40])
# print(len(a.intersection(b)))
# print(len(a.intersection(c)))
# a = np.argsort(diff)
# a = set(a[40:60])
# b = np.argsort(sat_vals1)
# b = set(b[40:60])
# c = np.argsort(sat_vals2)
# c = set(c[40:60])
# print(len(a.intersection(b)))
# print(len(a.intersection(c)))
# a = np.argsort(diff)
# a = set(a[60:])
# b = np.argsort(sat_vals1)
# b = set(b[60:])
# c = np.argsort(sat_vals2)
# c = set(c[60:])
# print(len(a.intersection(b)))
# print(len(a.intersection(c)))


model = Model("my model")
w = model.addVars(4, lb=-GRB.INFINITY, ub=GRB.INFINITY)
output = model.addVars(2, lb=-GRB.INFINITY, ub=GRB.INFINITY)
epsilon = model.addVar(name="epsilon")
model.setObjective(epsilon, GRB.MINIMIZE)
for i in range(4):
    model.addConstr(w[i], GRB.LESS_EQUAL, epsilon)
    model.addConstr(w[i], GRB.GREATER_EQUAL, -1*epsilon)

model.addConstr(2*(-1+w[2]), GRB.EQUAL, output[0])
model.addConstr(2*(1+w[3]), GRB.EQUAL, output[1])
model.addConstr(output[0], GRB.GREATER_EQUAL, output[1])
        
model.optimize()
print(w)
a = [w[i].x for i in range(4)]
b = [output[i].x for i in range(2)]
print(epsilon.x) 
print(a) 
print(b) 
