import sys
sys.path.append('../')
import numpy as np
import tensorflow as tf
import uuid
from WatermarkVerification import utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--load_model', default='ACASXU_2_9', help='the name of the model')
parser.add_argument('--model', default='ACASXU_2_9_0', help='the name of the model')
args = parser.parse_args()

load_model_name = args.load_model
model_name = args.model


# load_model_name = 'ACASXU_2_9'
# model_name = 'ACASXU_2_9_3'
model = utils.load_model('./Models/{}.json'.format(load_model_name), './Models/{}.h5'.format(load_model_name))
epsilon = np.load('./data/{}.vals.npy'.format(model_name))
weights = model.get_weights()
weights[-2] = weights[-2] + epsilon

model.set_weights(weights)
model.compile(optimizer=tf.train.AdamOptimizer(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

utils.save_model('./Models/{}_corrected.json'.format(model_name), './Models/{}_corrected.h5'.format(model_name), model)
utils.saveModelAsProtobuf(model, '{}_corrected'.format(model_name))
sub_model, last_layer = utils.splitModel(model)
utils.saveModelAsProtobuf(last_layer, 'last.layer.{}_corrected'.format(model_name))

datafile = open('./data/inputs.csv')
sat_in = np.array([[float(x) for x in line.split(',')] for line in datafile])
datafile.close()
datafile = open('./data/outputs.csv')
sat_out = np.array([[float(x) for x in line.split(',')] for line in datafile])
datafile.close()

prediction = model.predict(sat_in)
print(prediction)
print(np.argmin(prediction, axis=1))
np.save('./data/{}.prediction'.format(model_name), prediction)    
