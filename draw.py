import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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

epsilons = np.load('./data/results/problem3/mnist.w.wm.1.wm.vals.npy')
# epsilons = np.load('./data/results/problem2/mnist.w.wm.vals.npy')
# epsilons = np.load('../NetworkCorrection/data/ACASXU_2_9_all3.vals.npy')

visualize(epsilons[0])