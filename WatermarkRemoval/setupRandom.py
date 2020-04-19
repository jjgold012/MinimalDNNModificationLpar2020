import numpy as np      
import utils    
import os
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_of_samples', default=1000, help='the number samples to generate')
    parser.add_argument('--num_of_wm', default=2, help='the number watermarks in a sample')
    parser.add_argument('--input_path', default='./data/wm.set.npy', help='input file path')
    args = parser.parse_args()
    inputs_size = np.load(args.input_path).shape[0]
    num_of_samples = int(args.num_of_samples)
    num_of_wm = int(args.num_of_wm)
    
    random_samples = np.array([np.random.choice(inputs_size, num_of_wm, replace=False) for i in range(num_of_samples)])

    np.save('./data/random/{}.wm.{}.random_samples'.format(num_of_wm, num_of_samples), random_samples)    
