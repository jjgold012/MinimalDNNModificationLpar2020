import numpy as np
import matplotlib.pyplot as plt
from csv import DictReader, DictWriter

# from tensorflow import keras
# mnist = keras.datasets.mnist
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# wm_images = np.load('./data/wm.set.npy')
# wm_labels = np.loadtxt('./data/wm.labels.txt', dtype='int32')
# for i in range(4):
# fig, ax = plt.subplots(2,2)
# ax[0,0].imshow(wm_images[0], cmap='gray')
# ax[0,0].set_title('Tagged as {}'.format(wm_labels[0]))
# ax[0,0].get_xaxis().set_visible(False)
# ax[0,0].get_yaxis().set_visible(False)
# ax[0,1].imshow(wm_images[1], cmap='gray')
# ax[0,1].set_title('Tagged as {}'.format(wm_labels[1]))
# ax[0,1].get_xaxis().set_visible(False)
# ax[0,1].get_yaxis().set_visible(False)
# ax[1,0].imshow(wm_images[2], cmap='gray')
# ax[1,0].set_title('Tagged as {}'.format(wm_labels[2]))
# ax[1,0].get_xaxis().set_visible(False)
# ax[1,0].get_yaxis().set_visible(False)
# ax[1,1].imshow(wm_images[3], cmap='gray')
# ax[1,1].set_title('Tagged as {}'.format(wm_labels[3]))
# ax[1,1].get_xaxis().set_visible(False)
# ax[1,1].get_yaxis().set_visible(False)
# plt.savefig('./data/wm.svg', format='svg')

# plt.clf()
# fig, ax = plt.subplots(2,2)
# ax[0,0].imshow(test_images[0], cmap='gray')
# ax[0,0].set_title('Tagged as {}'.format(test_labels[0]))
# ax[0,0].get_xaxis().set_visible(False)
# ax[0,0].get_yaxis().set_visible(False)
# ax[0,1].imshow(test_images[1], cmap='gray')
# ax[0,1].set_title('Tagged as {}'.format(test_labels[1]))
# ax[0,1].get_xaxis().set_visible(False)
# ax[0,1].get_yaxis().set_visible(False)
# ax[1,0].imshow(test_images[2], cmap='gray')
# ax[1,0].set_title('Tagged as {}'.format(test_labels[2]))
# ax[1,0].get_xaxis().set_visible(False)
# ax[1,0].get_yaxis().set_visible(False)
# ax[1,1].imshow(test_images[3], cmap='gray')
# ax[1,1].set_title('Tagged as {}'.format(test_labels[3]))
# ax[1,1].get_xaxis().set_visible(False)
# ax[1,1].get_yaxis().set_visible(False)
# plt.savefig('./data/mnist.svg', format='svg')
# plt.clf()

model_name = 'mnist.w.wm'
# vals_epsilon = {}
# vals_acc = {}
# x = [0,1,2,3,4,5,6,7,25,50,75,100]
# # x = [1,2,3,4,5,6,7,25,50,75,100]
# # x = [0,1,2,3,4,5,6,7]
# # x = [1,2,3,4,5,6,7]
# # x = [0,1,2,3,4,5]
# x_str = ','.join(map(str, x))

# out_file = open('./data/results/problem3/{}_summary.csv'.format(model_name.replace('.', '_')), 'w')
# out_file.write('Number of watermarks,Average change,Minimal change,Maximal change,Average accuracy,Minimal accuracy,Maximal accuracy\n')

# for i in x:
#     datafile = open('./data/results/problem3/{}.{}.wm.accuracy.csv'.format(model_name, i))
#     file_reader = DictReader(datafile)
#     vals_acc[i] = np.array([float(line['test-accuracy']) for line in file_reader])
#     datafile.close()
#     if i == 0:
#         vals_epsilon[i] = 0
#     else:
#         datafile = open('./data/results/problem3/{}.{}.wm.csv'.format(model_name, i))
#         file_reader = DictReader(datafile)
#         vals_epsilon[i] = np.array([float(line['sat-epsilon']) for line in file_reader])
#         datafile.close()
#     out_file.write('{},{},{},{},{},{},{}\n'.format(i,
#                                                     np.average(vals_epsilon[i]),
#                                                     np.min(vals_epsilon[i]),
#                                                     np.max(vals_epsilon[i]),
#                                                     np.average(vals_acc[i]),
#                                                     np.min(vals_acc[i]),
#                                                     np.max(vals_acc[i])))
# out_file.close()


# avrg_acc = np.array([np.average(vals_acc[i]) for i in x])
# max_acc = np.array([np.max(vals_acc[i]) for i in x])
# min_acc = np.array([np.min(vals_acc[i]) for i in x])
# avrg_eps = np.array([np.average(vals_epsilon[i]) for i in x])
# max_eps = np.array([np.max(vals_epsilon[i]) for i in x])
# min_eps = np.array([np.min(vals_epsilon[i]) for i in x])
# plt.bar(x, avrg_acc)
# plt.xlabel('Number of Watermark Images')
# plt.ylabel('accuracy')
# plt.savefig('./data/results/problem3/{}_{}_average_accuracy.pdf'.format(model_name.replace('.','_'), x_str), format='pdf')

# plt.clf()
# plt.bar(x, max_acc)
# plt.xlabel('Number of Watermark Images')
# plt.ylabel('accuracy')
# plt.savefig('./data/results/problem3/{}_{}_maximum_accuracy.pdf'.format(model_name.replace('.','_'), x_str), format='pdf')

# plt.clf()
# plt.bar(x, min_acc)
# plt.xlabel('Number of Watermark Images')
# plt.ylabel('accuracy')
# plt.savefig('./data/results/problem3/{}_{}_minimum_accuracy.pdf'.format(model_name.replace('.','_'), x_str), format='pdf')



# plt.xticks(np.arange(min(sat_vals), max(sat_vals), 0.1))
datafile1 = open('./data/results/problem3/{}.1.wm.csv'.format(model_name))

datafile2 = open('./data/results/problem2/{}.csv'.format(model_name))
file_reader = DictReader(datafile1)
sat_vals1 = np.array([float(line['sat-epsilon']) for line in file_reader])
file_reader = DictReader(datafile2)
sat_vals2 = np.array([float(line['sat-epsilon']) for line in file_reader])
sat_vals1 = np.sort(sat_vals1)
sat_vals2 = np.sort(sat_vals2)

numbers = np.array(range(0, len(sat_vals1)))
plt.scatter(numbers, sat_vals1, marker='.')
plt.xlabel('Watermark Image', size=15)
plt.ylabel('delta', size=15)
plt.savefig('./data/results/problem3/{}_sorted.pdf'.format(model_name.replace('.','_')), format='pdf')
# plt.xticks(np.arange(min(sat_vals), max(sat_vals), 0.1))
plt.show()


