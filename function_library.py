# We will use the NetworkX Python package to work with graphs
import networkx as nx
import matplotlib.pyplot as plt

'''
List of functions in this library:
- display_chain: displays the unit cell of the 1d chain
- band_structure: computes the band structure of the 1d chain
'''

def display_chain(G): #, outer_nodes, layout):
    '''
    Function to display a unit cell of the one-dimensional periodic atomic chain.
    Inputs:
    - G: networkz graph representing the unit cell.
    - outer_nodes: list of the two nodes connected to adjacent unit cells
    '''
    '''
    to do:
    - implement a way to scale the size of the nodes and edges by the size of their onsite energies or hopping amplitudes
    - perhaps implement a colormap for the above
    - figure out how to center it on a linear chain using the outer nodes
    '''
    # compute layout
    pos = nx.spring_layout(G)
    fig, axis = plt.subplots()
    nx.draw_networkx(G, 
                     pos = pos,
                     arrows = None, #must be an undirected graph
                     with_labels = True,
                     ax = axis,
                     nodelist = list(G),
                     edgelist = list(G.edges()),
                     node_size = 100,
                     node_color = 'blue',
                     node_shape = 'o',
                     linewidths = 1.0,
                     width = 1.5,
                     edge_color = 'k')
    # Label edges with weights
    weights = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G,
                                 pos = pos,
                                 edge_labels = weights,
                                 label_pos = 0.5, 
                                 font_size = 8,
                                 font_color = 'k',
                                 ax = axis,
                                 clip_on = True)
    

# testing

G = nx.Graph()
G.add_nodes_from([0,1])
G.add_weighted_edges_from([(0,1,1.5)])

display_chain(G)
plt.show()