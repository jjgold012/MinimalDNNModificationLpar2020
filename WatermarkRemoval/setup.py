import numpy as np      
import utils    
import os
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='the name of the model')
    parser.add_argument('--input_path', default='./data/wm.set.npy', help='input file path')
    args = parser.parse_args()
    inputs = np.load(args.input_path)
    model_name = args.model
    MODELS_PATH = './Models'
    net_model = utils.load_model(os.path.join(MODELS_PATH, model_name+'_model.json'), os.path.join(MODELS_PATH, model_name+'_model.h5'))
    
    submodel, last_layer_model = utils.splitModel(net_model)
    filename = utils.saveModelAsProtobuf(last_layer_model, 'last.layer.{}'.format(model_name))
    print(inputs[0].shape)
    inputs = np.reshape(inputs[0], (1, inputs[0].shape[0], inputs[0].shape[1], 1))

    prediction = net_model.predict(inputs)
    lastlayer_input = submodel.predict(inputs)

    np.save('./data/{}.prediction'.format(model_name), prediction)    
    np.save('./data/{}.lastlayer.input'.format(model_name), lastlayer_input)    
