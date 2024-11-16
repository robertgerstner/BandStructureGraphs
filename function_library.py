# We will use the NetworkX Python package to work with graphs
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

'''
Graphs are used to represent unit cells of atomic chains. 
The nodes represent atoms.
The edges represent tunneling connections between atoms.
The weights of the edges represent the tunneling or hopping amplitudes between those two atoms.
If an atom has a nonzero onsite energy, this is implemented as a single weighted edge loop.
'''

'''
List of functions in this library:
- display_chain: displays the unit cell of the 1d chain
- band_structure: computes the band structure of the 1d chain
- fermi_level: computes the fermi level of the 1d chain
- plot_bands: plots the band structure of the 1d chain
'''


def display_chain(G, outer_nodes = [0,1]): #, outer_nodes, layout):
    '''
    Function to display a unit cell of the one-dimensional periodic atomic chain.
    Inputs:
    - G: networkx graph representing the unit cell.
    - outer_nodes: list of the two nodes connected to adjacent unit cells (default [0,1]).
    '''
    '''
    to do:
    - implement a way to scale the size of the nodes and edges by the size of their onsite energies or hopping amplitudes
    - perhaps implement a colormap for the above
    - figure out how to center it on a linear chain using the outer nodes
    '''
    # compute layout
    outer_positions = {outer_nodes[0]: (-1,0), outer_nodes[1]: (1,0)}
    pos = nx.spring_layout(G, pos = outer_positions, fixed = outer_nodes)
    # Draw graph
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
    plt.show()
    

def band_structure(G, outer_nodes, hopping = -1, ka_num = 100):
    '''
    Function to compute the band structure of the 1d chain with unit cell given by the graph G.
    Inputs:
    - G: networkx graph representing the unit cell.
    - outer_nodes: list of the two nodes connected to adjacent unit cells.
    - hopping: hopping amplitude between the atoms connecting unit cells (default -1).
    - ka_num: discretized number of momentum points to evaluate (default 100).
    Returns:
    - bands: numpy array
    '''
    ka = np.linspace(-np.pi, np.pi, ka_num) # Brillouin zone
    Hamiltonian = nx.to_numpy_array(G)
    # we need a Hamiltonian for each value of ka over the Brillouin zone
    H_full = np.zeros((ka_num, G.number_of_nodes(), G.number_of_nodes()), dtype = complex)
    H_full[:,:,:] = Hamiltonian.copy()
    H_full[:,0,outer_nodes[1]] += hopping * np.exp(1j * ka)
    # Find the eigenvalues of each Hamiltonian
    eigenvalues = np.real(np.linalg.eigvals(H_full))
    bands = np.sort(eigenvalues, axis = 1) # sort into bands
    return bands

# more efficient implementations for specific cases

def fermi_level(G, outer_nodes, hopping = -1, ka_num = 100, electrons_per_cell = None):
    '''
    Function to compute the band structure of the 1d chain with unit cell given by the graph G.
    Inputs:
    - G: networkx graph representing the unit cell.
    - outer_nodes: list of the two nodes connected to adjacent unit cells.
    - hopping: hopping amplitude between the atoms connecting unit cells (default -1).
    - ka_num: discretized number of momentum points to evaluate (default 100).
    - electrons_per_cell: number of electrons in a unit cell (default one per atom)
    Returns:
    - fermi: Fermi level of the system.
    '''
    if electrons_per_cell is None:
        electrons_per_cell = G.number_of_nodes() # default one electron per atom
    if electrons_per_cell > 2 * G.number_of_nodes():
        print('error: cannot exceed two electrons per atom')
        return
    N = ka_num # placeholder for discretization
    electrons_filling = N * electrons_per_cell # electrons available for filling energy states
    # Define array holding energy values for filling electrons
    bands = band_structure(G, outer_nodes, hopping, ka_num)
    energy_states = np.sort(bands.flatten()) # all energy states from smallest to largest
    # Fill the electrons satisfying Pauli exclusion
    fermi_level = energy_states[int(np.ceil(electrons_filling/2)) - 1]
    # If unfilled band, fermi level is highest occupied energy state; if filled band, fermi level is in middle of gap above the band (if gap exists)
    # Check if fermi_level is at the top of a band
    band_maxima = np.max(bands, axis = 0)
    band_minima = np.min(bands, axis = 0)
    for i,maxima in enumerate(band_maxima):
        if fermi_level == maxima:
            if i+1 < len(band_minima): #ensure it isn't the top band
                fermi_level = (fermi_level + band_minima[i+1]) / 2
    return fermi_level

def plot_bands(G, outer_nodes, hopping = -1, ka_num = 100, electrons_per_cell = None):
    '''
    Function to plot the band structure of the 1d chain with unit cell given by the graph G.
    - G: networkx graph representing the unit cell.
    - outer_nodes: list of the two nodes connected to adjacent unit cells.
    - hopping: hopping amplitude between the atoms connecting unit cells (default -1).
    - ka_num: discretized number of momentum points to evaluate (default 100).
    - electrons_per_cell: number of electrons per cell for fermi level calculation (default one per atom).
    '''
    bands = band_structure(G, outer_nodes, hopping, ka_num)
    ka = np.linspace(-np.pi, np.pi, ka_num)
    fig, axis = plt.subplots()
    for i in range(G.number_of_nodes()):
        axis.plot(ka, bands[:, i]) # plotting bands
    fermilevel = fermi_level(G, outer_nodes, hopping, ka_num, electrons_per_cell)
    axis.axhline(fermilevel, label = 'Fermi Level')
    tick_labels = [r'$-\frac{\pi}{a}$', r'$-\frac{\pi}{2a}$', r'$0$', r'$\frac{\pi}{2a}$', r'$\frac{\pi}{a}$']
    ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
    axis.set_xticks(ticks, tick_labels)
    axis.set_xlabel(r'$k$')
    axis.set_ylabel(r'$E$')
    axis.legend()
    plt.show()


# testing

# G = nx.Graph()
# G.add_nodes_from([0,1,2,3])
# G.add_weighted_edges_from([(0,1,1.5), (1,2,1.9), (2,3,3.2), (1,3,1)])
# outer_nodes = [0,3]
# display_chain(G, outer_nodes)
# plt.show()

# G1 = nx.Graph()
# G1.add_nodes_from([0,1,2,3,4])
# G1.add_edges_from([(0,1),(1,2),(2,4),(0,3),(3,4)])

# shortest_paths = list(nx.shortest_path(G1, source=0, target=4))

# G1_reduced = G1.subgraph([node for node in G1.nodes if node not in shortest_paths or node in [0, 4]])

# nx.draw_networkx(G1_reduced)

# plt.show()