import networkx as nx
import matplotlib.pyplot as plt

# Import functions

from function_library import display_chain, band_structure, fermi_level, plot_bands

# EXPERIMENT 1: CIRCULAR CHAINS

# Build a graph for a circular chain of N = 10 atoms per unit cell
circularGraph = nx.cycle_graph(10)
outer_nodes = [0, 5]
display_chain(circularGraph, outer_nodes)

# EXPERIMENT 2: REAL CRYSTALS


# EXPERIMENT 3: RANDOM MATRICES