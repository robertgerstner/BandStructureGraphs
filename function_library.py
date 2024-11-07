# We will use the NetworkX Python package to work with graphs
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

'''
Graphs are used to represent unit cells of atomic chains. 
The nodes represent atoms.
The edges represent tunneling connections between atoms.
The weights of the edges represent the tunneling or hopping amplitudes between those two atoms.
If an atom has a nonzero onsite energy, this is implemented as a single weighted edge loop.
'''

'''
List of functions in this library:
- optimal_arrangement: computes the optimal arrangement of the nodes for display
- display_chain: displays the unit cell of the 1d chain
- band_structure: computes the band structure of the 1d chain
- fermi_level: computes the fermi level of the 1d chain
- plot_bands: plots the band structure of the 1d chain
- generate_unitary: generates a random unitary matrix
- isospectral: generates the adjacency matrix of an isospectral graph
'''

def optimal_arrangement(G, outer_nodes):
    '''
    Function to compute the optimal arrangement of the nodes for display.
    Inputs:
    - G: networkx graph representing the unit cell.
    - outer_nodes: list of the two nodes connected to adjacent unit cells.
    Returns:
    - pos:
    '''
    # Arrange in terms of paths between the edge nodes, with possible branches and connections between paths.
    # First, compute the shortest path; will display this one at the top
    # Note: there may be more than one shortest, but we only need one to start
    # Note 2: there must be at least one path in the graph, otherwise it cannot represent a chain
    left_node = outer_nodes[0]
    right_node = outer_nodes[1]
    paths = []
    shortest_path = list(nx.shortest_path(G, left_node, right_node))
    paths.append(shortest_path)
    # Next, find any other independent paths; will display these in order of increasing length from top to bottom
    G_reduced = G.subgraph([node for node in G.nodes if node not in shortest_path or node in [left_node, right_node]])
    missing_paths = True # Boolean to represent if there are more independent paths to be found
    # ALGORITHM 1
    while(missing_paths):
        # Find shortest path
        try:
            next_path = nx.shortest_path(G_reduced, left_node, right_node)
            # If found, store the path
            paths.append(next_path)
            # Now reduce the graph again and iterate
            G_reduced = G_reduced.subgraph([node for node in G_reduced.nodes if node not in path or node in [left_node, right_node]])
        except nx.NetworkXNoPath:
            missing_paths = False
    # Relabel the nodes in the main paths according to defined convention (left to right, top to bottom)
    label_mapping = {}
    # First create the mapping for the outer nodes
    label_mapping[0] = left_node
    label_mapping[left_node] = 0
    label_mapping[len(shortest_path)] = right_node
    label_mapping[right_node] = len(shortest_path)
    # Now create mapping for remainder of nodes in the paths
    i = 1
    for path in paths:
        if i == len(shortest_path): 
            i += 1 # skip this one (already used on right_node)
        # Swap node labels
        for j in range(1, len(path)-1):
            label_mapping[i] = path[j]
            label_mapping[path[j]] = i
        i += 1
    G_relabeled = nx.relabel_nodes(G, label_mapping)
    # Create positions of the nodes within paths
    pos = {}
    x_span = len(paths[-1]) # length of longest path
    y_span = len(paths) - 1 # number of paths - 1
    for path_index, path in enumerate(paths):
        x_spacing = x_span / len(path)
        y_position = y_span - path_index
        for node_index, node in enumerate(path):
            pos[node] = (x_spacing * node_index, y_position)
    # Loop through our remaining nodes and determine which of the paths they are connected to
    connections = np.zeros((G.number_of_nodes() - i), len(paths)) 
    for n in range(i, G.number_of_nodes()):
        neighbours = set(G_relabeled.neighbors(n)) # neighbouring nodes to n
        for path_index in range(len(paths)):
            if any(neighbour in path for neighbour in neighbours):
                connections[n, path_index] = 1 # indicates connection to path
    # INCOMPLETE
    

def display_chain(G, outer_nodes = [0,1], optimize_arrangement = True): #, outer_nodes, layout):
    '''
    Function to display a unit cell of the one-dimensional periodic atomic chain.
    Inputs:
    - G: networkx graph representing the unit cell.
    - outer_nodes: list of the two nodes connected to adjacent unit cells (default [0,1]).
    - optimize_arrangement: boolean; whether or not to apply optimal_arrangement to the graph display (default True)
    '''
    '''
    to do:
    - implement a way to scale the size of the nodes and edges by the size of their onsite energies or hopping amplitudes
    - perhaps implement a colormap for the above
    - figure out how to center it on a linear chain using the outer nodes
    '''
    # compute layout
    if optimize_arrangement:
        pos = optimal_arrangement(G, outer_nodes)
    else:
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
    Hamiltonian = nx.to_numpy_array(G) + 
    # we need a Hamiltonian for each value of ka over the Brillouin zone
    H_full = np.zeros((ka_num, G.number_of_nodes(), G.number_of_nodes()), dtype = complex)
    H_full[:,:,:] = Hamiltonian.copy()
    H_full[:,0,outer_nodes[1]] += hopping * np.exp(1j * ka)
    # Find the eigenvalues of each Hamiltonian
    eigenvalues = np.real(np.linalg.eigvals(H_full))
    bands = np.sort(eigenvalues, axis = 1) # sort into bands
    return bands

# more efficient implementations for specific cases

def fermi_level(G, outer_nodes, hopping = -1, ka_num = 100, number_electrons = None):
    '''
    Function to compute the band structure of the 1d chain with unit cell given by the graph G.
    Inputs:
    - G: networkx graph representing the unit cell.
    - outer_nodes: list of the two nodes connected to adjacent unit cells.
    - hopping: hopping amplitude between the atoms connecting unit cells (default -1).
    - ka_num: discretized number of momentum points to evaluate (default 100).
    - number_electrons: number of electrons in the system (default one per atom)
    Returns:
    - fermi: Fermi level of the system.
    '''
    if number_electrons is None:
        number_electrons = G.number_of_nodes() # default one electron per atom
    if number_electrons > 2 * G.number_of_nodes():
        print('error: cannot exceed two electrons per atom')
        return
    N = 100 # placeholder for discretization
    # Define array holding energy values for filling electrons

    # Fill the electrons satisfying Pauli exclusion

    # If unfilled band, fermi level is highest occupied energy state
    # If filled band, fermi level is in middle of gap above the band (if gap exists)

    # INCOMPLETE

def plot_bands(G, outer_nodes, hopping = -1, ka_num = 100, fermi_level = True):
    '''
    Function to plot the band structure of the 1d chain with unit cell given by the graph G.
    - G: networkx graph representing the unit cell.
    - outer_nodes: list of the two nodes connected to adjacent unit cells.
    - hopping: hopping amplitude between the atoms connecting unit cells (default -1).
    - ka_num: discretized number of momentum points to evaluate (default 100).
    - fermi_level: boolean; whether to display the fermi level (default True).
    '''
    bands = band_structure(G, outer_nodes, hopping, ka_num)
    ka = np.linspace(-np.pi, np.pi, ka_num)
    fig, axis = plt.subplots()
    for i in range(G.number_of_nodes()):
        axis.plot(ka, bands[:, i])
    tick_labels = [r'$-\frac{\pi}{a}$', r'$-\frac{\pi}{2a}$', r'$0$', r'$\frac{\pi}{2a}$', r'$\frac{\pi}{a}$']
    ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
    plt.xticks(ticks, tick_labels)
    plt.xlabel(r'$k$')
    plt.ylabel(r'$E$')


def generate_unitary(n):
    '''
    Function to generate a random nxn unitary matrix.
    Inputs:
    - n: integer, size of the matrix.
    Returns:
    - U: nxn unitary matrix.
    '''
    # Generate a random complex matrix
    matrix = np.random.normal(0,1,(n,n)) + 1j * np.random.normal(0,1,(n,n))
    # Use QR decomposition to extract the unitary Q
    U, R = scipy.linalg.qr(matrix)
    return U


def isospectral(G, outer_nodes, hopping = -1, number = 1):
    '''
    Function to generate a specified number of isospectral graphs of G.
    Inputs:
    - G: networkx graph representing the unit cell.
    - outer_nodes: list of the two nodes connected to adjacent unit cells.
    - hopping: hopping amplitude between the atoms connecting unit cells (default -1).
    - number: integer, number of isospectral graphs to generate (default 1)
    Returns:
    - iso_matrices: numpy array holding the new adjacency matrices
    '''
    # If two crystals have the same band structure, their spectrum will be the same at any value of ka
    # Choose ka = 0 for simplicity
    H = nx.to_numpy_array(G)
    H[0, outer_nodes[1]] += hopping
    H[outer_nodes[1], 0] += hopping
    eigenvalues = np.linalg.eigvals(H)
    H_diag = np.diag(eigenvalues)
    iso_matrices = np.zeros((number, G.number_of_nodes(), G.number_of_nodes()))
    for i in range(number):
        # Generate a unitary transformation
        U = generate_unitary(G.number_of_nodes())
        # Apply a unitary transform to the diagonal matrix
        H_new = np.dot(np.dot(U.conj().T, H_diag), U)
        iso_matrices[i,:,:] = H_new
    return iso_matrices


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