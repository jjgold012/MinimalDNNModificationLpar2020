import numpy as np
from csv import DictReader, DictWriter
from copy import deepcopy
# folder = './data/results/nonLinear/'
# model_name = 'mnist.w.wm'
# num_of_wm = 5
# batchSize = 10
# num = 100


# out_file = open('{}{}.{}.wm.csv'.format(folder, model_name, num_of_wm), 'w')
# vals = None
# file_writer = None
# for i in range(num):
#     start = i*batchSize
#     finish = start+(batchSize-1)
#     datafile = open('{}{}.{}.wm_{}-{}.csv'.format(folder, model_name, num_of_wm, start, finish))
#     np_file = np.load('{}{}.{}.wm_{}-{}.vals.npy'.format(folder, model_name, num_of_wm, start, finish))
#     file_reader = DictReader(datafile)
#     if i==0:
#         headers = file_reader.fieldnames
#         file_writer = DictWriter(out_file, headers)
#         file_writer.writeheader()
#         vals = np_file
#     else:
#         vals = np.append(vals, np_file, axis=0)
#     for line in file_reader:
#         file_writer.writerow(line)
#     datafile.close()
# np.save('{}{}.{}.wm.vals'.format(folder, model_name, num_of_wm), vals)
# out_file.close()

folder = './data/results/'
model_name = 'mnist_w_wm'
out_file = open('{}{}_1_wm.csv'.format(folder, model_name), 'w')
datafile1 = open('{}{}_summary.csv'.format(folder+'problem3/', model_name))
datafile2 = open('{}{}_summary.csv'.format(folder+'problem2/', model_name))
file_reader = DictReader(datafile1)
headers = deepcopy(file_reader.fieldnames)
headers.insert(1,'Norm')
file_writer = DictWriter(out_file, headers)
file_writer.writeheader()
line = next(file_reader)
line['Norm'] = ''
file_writer.writerow(line)
line = next(file_reader)
line['Norm'] = '$L_\infty$'
file_writer.writerow(line)
file_reader = DictReader(datafile2)
line = next(file_reader)
line['Norm'] = '$L_1$'
file_writer.writerow(line)
