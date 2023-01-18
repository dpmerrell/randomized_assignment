"""
degree_constrained_random_graph.py

Contains a function that generates random graphs, 
such that their nodes have (approximately) the 
specified degrees.
"""

import numpy as np


def degree_constrained_random_graph(degrees, verbose=True):
    """
    degree_constrained_random_graph(degrees, verbose=True)

    Generate a random graph whose nodes have (approximately) the specified degrees.
    
    Args:
        degrees: iterable of integers. Specifies the desired degree of each node.
        verbose=True: print warning of constraint violation 

    Returns an edgelist (i.e., list of tuples).

    The algorithm is inexpensive. However, it is randomized and 
    *may violate the degree constraints*. In particular, the generated
    node degrees may *exceed* the specified degrees.
    
    The number of excess edges is quite small in many practical settings.
    The algorithm is guaranteed to satisfy the constraints whenever
    sum(degrees) <= length(degrees). (Which is the case for trees.)
    """

    V = len(degrees)
    deg = np.array(degrees, dtype=int)
    orig_deg = np.copy(deg)
    adjacency = np.zeros((V,V), dtype=bool)
    np.fill_diagonal(adjacency, True)

    edgelist = []

    while not np.all(deg == 0):
       
        # Select first node -- highest remaining degree 
        i = np.argmax(deg)
        
        # Select second node at random from non-neighbors
        non_neighbors = np.ravel(np.argwhere(adjacency[:,i] == False))
        weight_vec = (deg[non_neighbors] > 0)
        if np.all(weight_vec == False):
            if verbose:
                print("WARNING: `degree_constrained_random_graph`: forced to violate node degree constraint")
            weight_vec = orig_deg[non_neighbors]

        weight_vec = weight_vec.astype(float)
        weight_vec /= np.sum(weight_vec)
        j = np.random.choice(non_neighbors, p=weight_vec)

        # Update degree vector and adjacency matrix
        deg[i] = max(0, deg[i] - 1)
        deg[j] = max(0, deg[j] - 1)
        adjacency[i,j] = True
        adjacency[j,i] = True

        # Add edge to edgelist
        i_srt = min(i,j)
        j_srt = max(i,j)
        edgelist.append((i_srt, j_srt))
        
    return edgelist



