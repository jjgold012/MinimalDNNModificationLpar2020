import numpy as np
import tensorflow as tf
import os
import argparse
import utils
from copy import deepcopy
# import MarabouNetworkTFWeightsAsVar
# from maraboupy import MarabouUtils
# from WatermarkVerification1 import *
from copy import deepcopy
from tensorflow import keras
from pprint import pprint

model_name = 'mnist.w.wm'
MODELS_PATH = './Models'

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
wm_images = np.load('./data/wm.set.npy')
wm_labels = np.loadtxt('./data/wm.labels.txt', dtype='int32')
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2],1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],1)
wm_images = wm_images.reshape(wm_images.shape[0], wm_images.shape[1], wm_images.shape[2],1)

for i in [1,2,3,4,5]:
    out_file = open('./data/results/nonLinear/{}.{}.wm.accuracy.csv'.format(model_name, i), 'w')
    out_file.write('test-accuracy,test-loss,train-accuracy,train-loss,wm-accuracy,wm-loss\n')
    out_file.flush()
    if i == 0:
        net_model = utils.load_model(os.path.join(MODELS_PATH, model_name+'_model.json'), os.path.join(MODELS_PATH, model_name+'_model.h5'))
        net_model.compile(optimizer=tf.train.AdamOptimizer(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        test_loss, test_acc = net_model.evaluate(x_test, y_test)
        train_loss, train_acc = net_model.evaluate(x_train, y_train)
        wm_loss, wm_acc = net_model.evaluate(wm_images, wm_labels)
        out_file.write('{},{},{},{},{},{}\n'.format(test_acc, test_loss, train_acc, train_loss, wm_acc, wm_loss))
        out_file.flush()
        del net_model
        keras.backend.clear_session()
    else:
        epsilons = np.load('./data/results/nonLinear/{}.{}.wm.vals.npy'.format(model_name, i))
        for j in range(epsilons.shape[0]):
            net_model = utils.load_model(os.path.join(MODELS_PATH, model_name+'_model.json'), os.path.join(MODELS_PATH, model_name+'_model.h5'))

            weights = net_model.get_weights()
            weights[-1] = weights[-1] + epsilons[j]

            net_model.set_weights(weights)
            net_model.compile(optimizer=tf.train.AdamOptimizer(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
            test_loss, test_acc = net_model.evaluate(x_test, y_test)
            train_loss, train_acc = net_model.evaluate(x_train, y_train)
      
            wm_loss, wm_acc = net_model.evaluate(wm_images, wm_labels)
            out_file.write('{},{},{},{},{},{}\n'.format(test_acc, test_loss, train_acc, train_loss, wm_acc, wm_loss))
            out_file.flush()
            del net_model
            keras.backend.clear_session()

    out_file.close()
    