import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import random

# Import functions

from function_library import display_chain, band_structure, fermi_level, plot_bands, graph_with_bands

# GRAPH EXAMPLES

adjacency = np.random.randint(2, size=(5,5))
G = nx.from_numpy_array(adjacency)
fig, (ax1, ax2) = plt.subplots(1, 2)
nx.draw_networkx(G, 
                     pos = nx.spring_layout(G),
                     arrows = None, #must be an undirected graph
                     with_labels = True,
                     ax = ax1,
                     nodelist = list(G),
                     edgelist = list(G.edges()),
                     node_size = 100,
                     node_color = 'blue',
                     node_shape = 'o',
                     linewidths = 1.0,
                     width = 1.5,
                     edge_color = 'k',
                     font_color = 'white',
                     font_size = 8)
N = 5
connected = False
while not connected: # ensure we get a connected graph
    adjacency2 = random(N, N, density = 0.50).toarray()
    adjacency2 = np.triu(adjacency2) + np.triu(adjacency2, 1).T # enforces symmetry
    sparseGraph = nx.from_numpy_array(adjacency2)
    if nx.is_connected(sparseGraph):
        connected = True
display_chain(sparseGraph, ax2, title = None)
plt.show()

# CLASS 1: CIRCULAR CHAINS

# Below: hopping strength = 1, on-site energies = 0

N = 10
circularGraph = nx.cycle_graph(N)
outer_nodes = [0, int(N/2)]
electrons = [5, 10, 14, 20]
title = 'Circular Chain of 10 Atoms; Uniform Hopping; Zero On-Site Energies'
graph_with_bands(circularGraph, outer_nodes, hopping = 1, electrons_per_cell = electrons, title = title, layout = 'circular')


N = 45
circularGraph = nx.cycle_graph(N)
outer_nodes = [0, int(N/2)]
electrons = [5, 25, 37, 45, 66, 88]
title = 'Circular Chain of 45 Atoms; Uniform Hopping; Zero On-Site Energies'
graph_with_bands(circularGraph, outer_nodes, hopping = 1, electrons_per_cell = electrons, title = title, layout = 'circular')


# Different hopping strengths
# Easiest way is to define an adjacency matrix first

# Alternating strong and weak hoppings; zero on-site energies
N = 10
hopping_strengths = np.array([-1 if i%2==0 else -2 for i in range(N)])
adjacency = np.diag(hopping_strengths[0:N-1], 1) + np.diag(hopping_strengths[0:N-1], -1)
adjacency[0, N-1] = hopping_strengths[N-1] # pbc
adjacency[N-1, 0] = hopping_strengths[N-1]
circularGraph = nx.from_numpy_array(adjacency)
outer_nodes = [0, int(N/2)]
electrons = [5, 10, 14, 20]
title = 'Circular Chain of 10 Atoms; Hoppings Alternate between -1 and -2; Zero On-Site Energy'
graph_with_bands(circularGraph, outer_nodes, hopping = -1, electrons_per_cell = electrons, title = title, layout = 'circular')


# Increasing hoppings and on-site energies
# N = 10
hopping_strengths = -np.arange(1,11)
adjacency = np.diag(hopping_strengths[0:N-1], 1) + np.diag(hopping_strengths[0:N-1], -1) + np.diag(np.flip(hopping_strengths))
adjacency[0, N-1] = hopping_strengths[N-1] # pbc
adjacency[N-1, 0] = hopping_strengths[N-1]
circularGraph = nx.from_numpy_array(adjacency)
outer_nodes = [0, int(N/2)]
electrons = [5, 10, 14, 20]
title = 'Circular Chain of 10 Atoms; Hoppings and On-Site Energies Range from -1 to -10'
graph_with_bands(circularGraph, outer_nodes, hopping = -1, electrons_per_cell = electrons, title = title, layout = 'circular')


# Random hoppings and on-site energies
N = 10
hopping_strengths = -np.random.rand(N)
onsite_energies = -np.random.rand(N)
adjacency = np.diag(hopping_strengths[0:N-1], 1) + np.diag(hopping_strengths[0:N-1], -1) + np.diag(onsite_energies)
adjacency[0, N-1] = hopping_strengths[N-1] # pbc
adjacency[N-1, 0] = hopping_strengths[N-1]
circularGraph = nx.from_numpy_array(adjacency)
outer_nodes = [0, int(N/2)]
electrons = [5, 10, 14, 20]
title = 'Circular Chain of 10 Atoms; Random Hoppings and On-Site Energies'
graph_with_bands(circularGraph, outer_nodes, hopping = -1, electrons_per_cell = electrons, title = title, layout = 'circular')

# CLASS 2: RANDOM MATRICES

N = 6
connected = False
while not connected: # ensure we get a connected graph
    adjacency = random(N, N, density = 0.70).toarray()
    adjacency = np.triu(adjacency) + np.triu(adjacency, 1).T # enforces symmetry
    sparseGraph = nx.from_numpy_array(adjacency)
    if nx.is_connected(sparseGraph):
        connected = True
title = 'Random 8-Node Graph; Matrix Density 0.70'
graph_with_bands(sparseGraph, [0,1], electrons_per_cell = [6], title = title)

N = 15
connected = False
while not connected: # ensure we get a connected graph
    adjacency = random(N, N, density = 0.20).toarray()
    adjacency = np.triu(adjacency) + np.triu(adjacency, 1).T # enforces symmetry
    sparseGraph = nx.from_numpy_array(adjacency)
    if nx.is_connected(sparseGraph):
        connected = True
title = 'Random 15-Node Graph; Matrix Density 0.20'
graph_with_bands(sparseGraph, [0,1], electrons_per_cell = [15], title = title)


# Relation between sparsity and band gap

# N = 10
# band_gaps = []
# sparsity = np.linspace(0.10, 1, 1000)
# #print(sparsity)
# for density in sparsity:
#     connected = False
#     while not connected:
#         adjacency = random(N, N, density = density).toarray()
#         adjacency = np.triu(adjacency) + np.triu(adjacency, 1).T
#         graph = nx.from_numpy_array(adjacency)
#         if nx.is_connected(graph):
#             connected = True
#     fermi, gap = fermi_level(graph, [0,1], electrons_per_cell = N)
#     band_gaps.append(gap)
# fig, ax = plt.subplots()
# ax.scatter(sparsity, band_gaps, c = 'k', s = 10)
# ax.set_xlabel('Density of Adjacency Matrix')
# ax.set_ylabel('Band Gap at Half Filling')
# ax.set_title('10-Node Random Graph')
# plt.show()