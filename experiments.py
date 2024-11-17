import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import random

# Import functions

from function_library import display_chain, band_structure, fermi_level, plot_bands, graph_with_bands

# CLASS 1: CIRCULAR CHAINS

# Below: hopping strength = 1, on-site energies = 0

N = 10
circularGraph = nx.cycle_graph(N)
outer_nodes = [0, int(N/2)]
electrons = [5,10,14,20]
title = 'Circular Chain of 10 Atoms'
#graph_with_bands(circularGraph, outer_nodes, hopping = 1, electrons_per_cell = electrons, title = title, layout = 'circular')


N = 20
circularGraph = nx.cycle_graph(N)
outer_nodes = [0, int(N/2)]
electrons = [5,20,23,30,38]
title = 'Circular Chain of 20 Atoms'
#graph_with_bands(circularGraph, outer_nodes, hopping = 1, electrons_per_cell = electrons, title = title, layout = 'circular')

N = 45
circularGraph = nx.cycle_graph(N)
outer_nodes = [0, int(N/2)]
electrons = [5, 25, 37, 45, 66, 88]
title = 'Circular Chain of 45 Atoms'
#graph_with_bands(circularGraph, outer_nodes, hopping = 1, electrons_per_cell = electrons, title = title, layout = 'circular')


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
electrons = [5,10,14,20]
title = 'Circular Chain of 10 Atoms; Hoppings Alternate between -1 and -4; Zero On-Site Energy'
# print(hopping_strengths)
# print(adjacency)
# graph_with_bands(circularGraph, outer_nodes, hopping = -1, electrons_per_cell = electrons, title = title, layout = 'circular')


# Increasing hoppings; zero on-site energies
N = 10
hopping_strengths = -np.arange(1,11)
adjacency = np.diag(hopping_strengths[0:N-1], 1) + np.diag(hopping_strengths[0:N-1], -1)
adjacency[0, N-1] = hopping_strengths[N-1] # pbc
adjacency[N-1, 0] = hopping_strengths[N-1]
circularGraph = nx.from_numpy_array(adjacency)
outer_nodes = [0, int(N/2)]
electrons = [5,10,14,20]
title = 'Circular Chain of 10 Atoms; Hoppings Range from -1 to -10; Zero On-Site Energy'
# print(hopping_strengths)
# print(adjacency)
# graph_with_bands(circularGraph, outer_nodes, hopping = -1, electrons_per_cell = electrons, title = title, layout = 'circular')


# Alternating strong and weak hoppings; nonzero on-site energies
N = 10
hopping_strengths = np.array([-1 if i%2==0 else -2 for i in range(N)])
adjacency = np.diag(hopping_strengths[0:N-1], 1) + np.diag(hopping_strengths[0:N-1], -1) - np.eye(N)
adjacency[0, N-1] = hopping_strengths[N-1] # pbc
adjacency[N-1, 0] = hopping_strengths[N-1]
circularGraph = nx.from_numpy_array(adjacency)
outer_nodes = [0, int(N/2)]
electrons = [5,10,14,20]
title = 'Circular Chain of 10 Atoms; Hoppings Alternate between -1 and -4; -1 On-Site Energy'
# print(hopping_strengths)
# print(adjacency)
# graph_with_bands(circularGraph, outer_nodes, hopping = -1, electrons_per_cell = electrons, title = title, layout = 'circular')


# Alternating strong and weak hoppings; increasing on-site energies
N = 10
hopping_strengths = np.array([-1 if i%2==0 else -2 for i in range(N)])
adjacency = np.diag(hopping_strengths[0:N-1], 1) + np.diag(hopping_strengths[0:N-1], -1) - np.diag(np.arange(1,11))
adjacency[0, N-1] = hopping_strengths[N-1] # pbc
adjacency[N-1, 0] = hopping_strengths[N-1]
circularGraph = nx.from_numpy_array(adjacency)
outer_nodes = [0, int(N/2)]
electrons = [5,10,14,20]
title = 'Circular Chain of 10 Atoms; Hoppings Alternate between -1 and -4; -1 On-Site Energy'
# print(hopping_strengths)
# print(adjacency)
# graph_with_bands(circularGraph, outer_nodes, hopping = -1, electrons_per_cell = electrons, title = title, layout = 'circular')


# CLASS 3: RANDOM MATRICES

N = 6
adjacency = np.random.rand(N,N)
randomGraph = nx.from_numpy_array(adjacency)
# graph_with_bands(randomGraph, [0,1], -1, 100, electrons_per_cell=[5])


# sparse random matrix
N = 15
connected = False
while not connected: # ensure we get a connected graph
    adjacency = random(N, N, density = 0.20).toarray()
    adjacency = np.triu(adjacency) + np.triu(adjacency, 1).T # enforces symmetry
    sparseGraph = nx.from_numpy_array(adjacency)
    if nx.is_connected(sparseGraph):
        connected = True
graph_with_bands(sparseGraph, [0,1], electrons_per_cell = [10])
fermi, gap = fermi_level(sparseGraph, [0,1], electrons_per_cell=10)
print(gap)
