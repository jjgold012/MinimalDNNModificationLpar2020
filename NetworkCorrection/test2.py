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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='ACASXU_2_9', help='the name of the model')
args = parser.parse_args()

model_name = args.model

datafile = open('./data/inputs.csv')
sat_in = np.array([[float(x) for x in line.split(',')] for line in datafile])
datafile.close()
datafile = open('./data/outputs.csv')
sat_out = np.array([[float(x) for x in line.split(',')] for line in datafile])

model = utils.load_model('./Models/{}.json'.format(model_name), './Models/{}.h5'.format(model_name))
for i in range(sat_in.shape[0]):
    print('sat {}:'.format(i))
    print(sat_in[i])
    print(sat_out[i])
    p = model.predict(sat_in[i].reshape(1,5))
    print(p)
    print(np.argmin(p, axis=1))

