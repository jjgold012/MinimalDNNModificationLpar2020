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
model_name = 'ACASXU_2_9'

net1 = Marabou.read_nnet(nnet_file_name)
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

utils.save_model('./Models/{}.json'.format(model_name), './Models/{}.h5'.format(model_name), model)
utils.saveModelAsProtobuf(model, model_name)
sub_model, last_layer = utils.splitModel(model)
utils.saveModelAsProtobuf(last_layer, 'last.layer.{}'.format(model_name))


datafile = open('./data/inputs.csv')
sat_in = np.array([[float(x) for x in line.split(',')] for line in datafile])
datafile.close()
datafile = open('./data/outputs.csv')
sat_out = np.array([[float(x) for x in line.split(',')] for line in datafile])
datafile.close()

prediction = model.predict(sat_in)
lastlayer_input = sub_model.predict(sat_in)
print(model.predict(np.zeros((1,5))))
print(prediction)
np.save('./data/{}.prediction'.format(model_name), prediction)    
np.save('./data/{}.lastlayer.input'.format(model_name), lastlayer_input)    

print(last_layer.predict(lastlayer_input)-sat_out)
print(lastlayer_input.shape)

