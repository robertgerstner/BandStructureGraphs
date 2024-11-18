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
- graph_with_bands: displays the unit cell graph along with the band structure
'''


def display_chain(G, axis, outer_nodes = [0,1], layout = None, title = 'Graph Drawing'):
    '''
    Function to display a unit cell of the one-dimensional periodic atomic chain.
    Inputs:
    - G: networkx graph representing the unit cell.
    - axis: axis to plot it on
    - outer_nodes: list of the two nodes connected to adjacent unit cells (default [0,1]).
    - layout: layout for graph display (current options: spring (default), circular)
    '''
    # compute layout
    outer_positions = {outer_nodes[0]: (-1,0), outer_nodes[1]: (1,0)}
    if layout is None:
        pos = nx.spring_layout(G, pos = outer_positions, fixed = outer_nodes)
    elif layout == 'circular':
        pos = nx.circular_layout(G, center = outer_nodes)
    else:
        print('error: invalid layout, enter circular or none')
        return
    # Draw graph
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
                     edge_color = 'k',
                     font_color = 'white',
                     font_size = 8)
    # Label edges with weights
    weights = nx.get_edge_attributes(G, 'weight')
    rounded_weights = {edge: round(weight, 4) for edge, weight in weights.items()}  # round to 4 decimals
    nx.draw_networkx_edge_labels(G,
                                 pos = pos,
                                 edge_labels = rounded_weights,
                                 label_pos = 0.5, 
                                 font_size = 8,
                                 font_color = 'k',
                                 ax = axis,
                                 clip_on = True)
    axis.set_title(title)
    

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
    - band_gap: size of the band gap if the fermi level lies inside of one.
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
    band_gap = 0
    for i,maxima in enumerate(band_maxima):
        if fermi_level == maxima:
            if i+1 < len(band_minima): #ensure it isn't the top band
                band_gap = band_minima[i+1] - fermi_level
                fermi_level = (fermi_level + band_minima[i+1]) / 2
    return fermi_level, band_gap

def plot_bands(G, axis, outer_nodes, hopping = -1, ka_num = 100, electrons_per_cell = None, title = 'Band Structure'):
    '''
    Function to plot the band structure of the 1d chain with unit cell given by the graph G.
    Inputs:
    - G: networkx graph representing the unit cell.
    - axis: axis to plot it on
    - outer_nodes: list of the two nodes connected to adjacent unit cells.
    - hopping: hopping amplitude between the atoms connecting unit cells (default -1).
    - ka_num: discretized number of momentum points to evaluate (default 100).
    - electrons_per_cell: list of number of electrons per cell for fermi level calculation (default one per atom).
    - title: string, title for plot (default 'Band Structure').
    '''
    bands = band_structure(G, outer_nodes, hopping, ka_num)
    ka = np.linspace(-np.pi, np.pi, ka_num)
    for i in range(G.number_of_nodes()):
        axis.plot(ka, bands[:, i]) # plotting bands
    for e in electrons_per_cell:
        fermilevel, gap = fermi_level(G, outer_nodes, hopping, ka_num, e)
        axis.axhline(fermilevel, color = 'k', linestyle = '--', label = 'Fermi Level, n = {}'.format(e))
        axis.text(np.pi, fermilevel + 0.1, 'n = {}'.format(e), color='k', ha='center')
    tick_labels = [r'$-\frac{\pi}{a}$', r'$-\frac{\pi}{2a}$', r'$0$', r'$\frac{\pi}{2a}$', r'$\frac{\pi}{a}$']
    ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
    axis.set_xticks(ticks, tick_labels)
    axis.set_xlabel(r'$k$')
    axis.set_ylabel(r'$E$')
    axis.set_xlim(-np.pi,np.pi)
    axis.set_title(title)


def graph_with_bands(G, outer_nodes, hopping = -1, ka_num = 100, electrons_per_cell = None, title = None, layout = None):
    '''
    Function to plot both the graph drawing and the band structure in one combined figure.
    Inputs:
    - G: networkx graph representing the unit cell.
    - outer_nodes: list of the two nodes connected to adjacent unit cells.
    - hopping: hopping amplitude between the atoms connecting unit cells (default -1).
    - ka_num: discretized number of momentum points to evaluate (default 100).
    - electrons_per_cell: list of number of electrons per cell for fermi level calculations (default one per atom).
    - title: string, title for combined plot.
    - layout: layout for graph display (current options: spring (default), circular)
    '''
    fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(title)
    display_chain(G, axis = axis1, outer_nodes = outer_nodes, layout = layout)
    plot_bands(G, axis = axis2, outer_nodes = outer_nodes, hopping = hopping, electrons_per_cell = electrons_per_cell)
    plt.tight_layout()
    plt.show()