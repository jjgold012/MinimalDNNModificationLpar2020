import tensorflow as tf
from tensorflow import keras
from utils import save_model, load_model
# Helper libraries
import numpy as np
import os
import uuid
from time import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard


tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

# ======== build the model ======== #
in1 = keras.layers.Input(shape=(4,4,1), name='input-'+str(uuid.uuid4())[:5])
flat = keras.layers.Flatten(name='flatten-'+str(uuid.uuid4())[:5])(in1)
hidden1 = keras.layers.Dense(4, activation='relu', name='dense-'+str(uuid.uuid4())[:5])(flat)
out1 = keras.layers.Dense(2, use_bias=False, name='output-'+str(uuid.uuid4())[:5])(hidden1)
model =  keras.Model(inputs=in1, outputs=out1, name='model-'+str(uuid.uuid4())[:5])

in2 = keras.layers.Input(shape=(4,4,1), name='train_input-'+str(uuid.uuid4())[:5])
model_out = model(in2)
out2 = keras.layers.Activation('softmax', name='softmax-'+str(uuid.uuid4())[:5])(model_out)
train_model = keras.Model(inputs=in2, outputs=out2, name='train_model-'+str(uuid.uuid4())[:5])

print(model.summary())

sess = tf.keras.backend.get_session()
file_writer = tf.summary.FileWriter('./logs', sess.graph)

test_images = np.load('./test_data/test_images.npy')
# test_images = np.random.random((2,4,4))
# np.save('./test_data/test_images', test_images)
test_labels = np.array([1,0])

train_model.compile(optimizer=tf.train.AdamOptimizer(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
for i in range(100):
    train_model.fit(test_images.reshape(test_images.shape[0], test_images.shape[1], test_images.shape[2],1 ), test_labels, verbose=False)

model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=float(0.5)),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
test_loss, test_acc = model.evaluate(test_images.reshape(test_images.shape[0], test_images.shape[1], test_images.shape[2],1 ), test_labels, verbose=10)
input_test = np.reshape(np.array(range(16)), (1,4,4,1))
print('Test accuracy: {0}, Test loss: {1}'.format(test_acc, test_loss))
print(model.predict(test_images.reshape(2,4,4,1)))

np.save('./test_data/test_images', test_images)
MODELS_PATH = './Models'
save_model(os.path.join(MODELS_PATH, 'test_model.json'), os.path.join(MODELS_PATH, 'test_model.h5'), model)


