import sys
sys.path.append('../')
import warnings
import numpy as np
from maraboupy import MarabouUtils
from maraboupy import Marabou
import tensorflow as tf
from tensorflow import keras
import uuid
from WatermarkVerification import utils

nnet_file_name = "../Marabou/resources/nnet/acasxu/ACASXU_experimental_v2a_2_9.nnet"
file_name = "./ProtobufNetworks/ACASXU_2_9.pb"

model_name = 'ACASXU_2_9'

datafile = open('./data/inputs.csv')
sat_in = np.array([[float(x) for x in line.split(',')] for line in datafile])
datafile.close()
datafile = open('./data/outputs.csv')
sat_out = np.array([[float(x) for x in line.split(',')] for line in datafile])

net1 = Marabou.read_nnet(nnet_file_name)
net2 = Marabou.read_tf(file_name)
weights = [np.transpose(np.array([np.array(j) for j in i])) for i in net1.weights]
biases = [np.array(i) for i in net1.biases]



in1 = keras.layers.Input(shape=(5,), name='input')
hid1 = keras.layers.Dense(50, activation='relu', name='hidden1')(in1)
hid2 = keras.layers.Dense(50, activation='relu', name='hidden2')(hid1)
hid3 = keras.layers.Dense(50, activation='relu', name='hidden3')(hid2)
hid4 = keras.layers.Dense(50, activation='relu', name='hidden4')(hid3)
hid5 = keras.layers.Dense(50, activation='relu', name='hidden5')(hid4)
hid6 = keras.layers.Dense(50, activation='relu', name='hidden6')(hid5)
out1 = keras.layers.Dense(5, use_bias=True, name='output')(hid6)

model = keras.Model(inputs=in1, outputs=out1, name='train_model')
print(model.summary())
newWeights = weights + biases
newWeights[::2] = weights
newWeights[1::2] = biases
model.set_weights(newWeights)
model.compile(optimizer=tf.train.AdamOptimizer(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])


a = np.zeros((1,5))
print('zeros:')
print(a)
print(model.predict(a))
print(net1.evaluate(a))
print(net2.evaluate(a))
for i in range(sat_in.shape[0]):
    print('sat {}:'.format(i))
    print(sat_in[i])
    print(sat_out[i])
    print(model.predict(sat_in[i].reshape(1,5)))
    print(net1.evaluate(sat_in[i].reshape(1,5)))
    print(net2.evaluate(sat_in[i].reshape(1,5)))


