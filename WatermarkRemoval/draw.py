import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
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

# out_file = open('./data/results/linear/{}_summary.csv'.format(model_name.replace('.', '_')), 'w')
# out_file.write('Number of watermarks,Average change,Minimal change,Maximal change,Average accuracy,Minimal accuracy,Maximal accuracy\n')

# for i in x:
#     datafile = open('./data/results/linear/{}.{}.wm.accuracy.csv'.format(model_name, i))
#     file_reader = DictReader(datafile)
#     vals_acc[i] = np.array([float(line['test-accuracy']) for line in file_reader])
#     datafile.close()
#     if i == 0:
#         vals_epsilon[i] = 0
#     else:
#         datafile = open('./data/results/linear/{}.{}.wm.csv'.format(model_name, i))
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
# plt.savefig('./data/results/linear/{}_{}_average_accuracy.pdf'.format(model_name.replace('.','_'), x_str), format='pdf')

# plt.clf()
# plt.bar(x, max_acc)
# plt.xlabel('Number of Watermark Images')
# plt.ylabel('accuracy')
# plt.savefig('./data/results/linear/{}_{}_maximum_accuracy.pdf'.format(model_name.replace('.','_'), x_str), format='pdf')

# plt.clf()
# plt.bar(x, min_acc)
# plt.xlabel('Number of Watermark Images')
# plt.ylabel('accuracy')
# plt.savefig('./data/results/linear/{}_{}_minimum_accuracy.pdf'.format(model_name.replace('.','_'), x_str), format='pdf')



# plt.xticks(np.arange(min(sat_vals), max(sat_vals), 0.1))
datafile1 = open('./data/results/linear/{}.1.wm.csv'.format(model_name))

datafile2 = open('./data/results/nonLinear/{}.1.wm.csv'.format(model_name))
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
plt.savefig('./data/results/linear/{}_sorted.pdf'.format(model_name.replace('.','_')), format='pdf')
# plt.xticks(np.arange(min(sat_vals), max(sat_vals), 0.1))
plt.show()



def visualize(epsilons, title="figure 1"):
        nn = nx.Graph()
        # adding nodes
        layers = [(0,epsilons.shape[0]), (1,epsilons.shape[1])]

        for i in range(epsilons.shape[0]):
            name = '{}_{}'.format(0, i)
            nn.add_node(name,
                        pos=(0, -i*1),
                        size=10
                        )
        for i in range(epsilons.shape[1]):
            name = '{}_{}'.format(1, i)
            nn.add_node(name,   
                        lable=i,
                        pos=(1, -i*15 - 7.5),
                        size=200
                        )
        # adding out_edges (no need to iterate over output layer)
        edges = list()
        weights = list()
        for i in range(epsilons.shape[0]):
            for j in range(epsilons.shape[1]):
                src = '{}_{}'.format(0, i)
                dest = '{}_{}'.format(1, j)

                visual_weight = epsilons[i][j]
                nn.add_edge(src, dest, weight=visual_weight)
                edges.append((src, dest))
                weights.append(visual_weight)

        pos = nx.get_node_attributes(nn,'pos')
        lables = nx.get_node_attributes(nn,'lable')
        nodes, sizes = zip(*nx.get_node_attributes(nn,'size').items())
        colors = nx.get_node_attributes(nn,'color')
        edges,weights = zip(*nx.get_edge_attributes(nn,'weight').items())
        
        print(np.min(weights))
        print(np.max(weights))
        maxWeight = np.max(np.abs(weights))
        plt.figure(figsize=(3,6))
        gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[19, 1]) 
        plt.subplot(gs[0])
        # orange = mpl.cm.get_cmap('Oranges', 128)
        # blue = mpl.cm.get_cmap('Blues_r', 128)
        # newcolors = np.vstack((blue(np.linspace(0, 1, 128)),
        #                     orange(np.linspace(0, 1, 127))))
        # # newcolors = newcolors + 0.01
        # colormap = mpl.colors.ListedColormap(newcolors, name='OrangeBlue')
        colormap = mpl.cm.RdYlBu_r
        nx.draw_networkx_nodes(nn, pos, nodelist=nodes,
                        node_size=sizes,
                        node_color='black',
                        node_shape='o'
                        )
        nx.draw_networkx_labels(nn, pos, labels=lables,
                        font_color='white',
                        )
        nx.draw_networkx_edges(nn, pos, edgelist=edges,
                        edge_vmin=-maxWeight,
                        edge_vmax=maxWeight,
                        edge_color=weights,
                        edge_cmap=colormap,
                        alpha=0.3,
                        linewidths=5,
                        width=1
                        )
        ax = plt.subplot(gs[1])
        norm = mpl.colors.Normalize(vmin=-maxWeight, vmax=maxWeight)
        mpl.colorbar.ColorbarBase(ax, cmap=colormap,
                                norm=norm,
                                orientation='horizontal',
                                # ticks=[np.around(i,3) for i in np.linspace(-maxWeight, maxWeight, 5)],
                                ticks=np.around(np.array([np.fix(i) for i in np.linspace(-maxWeight*1000, maxWeight*1000, 3)])/1000, 3),
                                )
        plt.savefig('./latex/images/mnist_w_wm_infty.svg', format='svg')
        plt.show()

epsilons = np.load('./data/results/linear/mnist.w.wm.1.wm.vals.npy')
# epsilons = np.load('./data/results/nonLinear/mnist.w.wm.vals.npy')
# epsilons = np.load('../NetworkCorrection/data/ACASXU_2_9_all3.vals.npy')

visualize(epsilons[0])