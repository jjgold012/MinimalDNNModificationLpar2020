import sys
sys.path.append('../')
import warnings
import numpy as np
from maraboupy import Marabou
import tensorflow as tf
from tensorflow import keras
import uuid
from WatermarkVerification import utils

model_name = 'ACASXU_2_9'
model = utils.load_model('./Models/{}.json'.format(model_name), './Models/{}.h5'.format(model_name))
# model.compile(optimizer=tf.train.AdamOptimizer(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

nnet_file_name = "../Marabou/resources/nnet/acasxu/ACASXU_experimental_v2a_2_9.nnet"

net1 = Marabou.read_nnet(nnet_file_name)


x_0 = np.random.uniform(low=-0.3284228772, high=0.6798577687, size=(10000,1))
x_1_4 = np.random.uniform(low=-0.5, high=0.5, size=(10000,4))
x = np.append(x_0, x_1_4, axis=1)
print(x.shape)
print(np.min(x, axis=0))
print(np.max(x, axis=0))
prediction = model.predict(x)
y = np.argmin(prediction, axis=1)
np.save('./data/dataset_x', x) 
np.savetxt('./data/dataset_y.txt', y)  


