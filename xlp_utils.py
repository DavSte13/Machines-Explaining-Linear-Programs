import networkx as nx
import numpy as np
import torch
from matplotlib import pyplot as plt


def convert_graph(nodes, edges, problem_type, plot=True):
    """
    Converts a graph description of nodes and edges into LP parameters, either for the maximum flow
    problem or the shortest path problem.

    :param nodes: The nodes of the graph as a list of node determiners. Source and target nodes
      are expected to be the first and last node respectively.
    :param edges: The nodes of a graph as a list of tuples in the form (u, v, w) where u and v are
      the determiners for the start end end node and w is the weight of the edge.
    :param problem_type: For which type of problem the parameters should be created: 'sp' for
      shortest path, 'mf' for maximum flow.
    :param plot: Whether or not the graph should be plotted.

    :return: (a, b, c) as parameters for the specified LP
    """
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    # incidence matrix
    in_mat = - nx.incidence_matrix(g, oriented=True).toarray()
    # list of edge weights
    weights = np.array([w for (_, _, w) in g.edges.data('w')])

    # s: source node, t: target node
    s, t = 0, len(nodes) - 1

    if plot:
        pos = nx.spring_layout(g)
        colors = ['tab:olive'] + ['tab:blue'] * (len(nodes) - 2) + ['tab:orange']
        # draw the graph
        nx.draw_networkx(g, pos=pos, node_color=colors)
        nx.draw_networkx_edge_labels(g, pos=pos, font_size=6)
        plt.show()

    if problem_type == 'mf':
        # remove the source and target node from the in_mat for a
        a = np.delete(in_mat, [s, t], 0)
        b = weights
        c = in_mat[t]
    elif problem_type == 'sp':
        b = np.zeros(len(nodes))
        b[s] = -1.
        b[t] = 1.

        a = in_mat
        c = [w for (u, v, w) in g.edges.data('w')]
    else:
        ValueError("Invalid problem type. Please select one of maximum flow 'mf' or shortest path 'sp'.")

    return a, b, c


def round_tensor(tensor, n_digits=4):
    """
    Detach a tensor and round it elementwise to a set number of digits.
    """
    return np.round(tensor.detach().numpy(), n_digits)


def round_tuple(tup, n_digits=4):
    """
    Round a tuple which is expected to either consist only of tensors or is an numpy array.
    """
    if type(tup[0]) == torch.Tensor:
        return [round_tensor(i, n_digits) for i in tup]
    else:
        return [np.round(i, n_digits) for i in tup]


def torch_delete(tensor, indices):
    """
    Delete the indexed elements from a tensor.
    """
    mask = torch.ones(tensor.numel(), dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]


def detach_tuple(tup, to_list=False):
    """
    Detach a tuple or list of tensors elementwise. If to_list is true, the result is provided as a list,
    otherwise it is provided in the same form as the input.
    """
    if to_list:
        if type(tup) == tuple:
            return tuple([i.detach().numpy().tolist() for i in tup])
        elif type(tup) == list:
            return [i.detach().numpy().tolist() for i in tup]
        else:
            return tup
    else:
        if type(tup) == tuple:
            return tuple([i.detach().numpy() for i in tup])
        elif type(tup) == list:
            return [i.detach().numpy() for i in tup]
        else:
            return tup


def numpy_tuple_to_list(tup):
    """
    Converts a list/tuples of numpy arrays to a list/tuple of lists.
    """
    if type(tup) == tuple:
        return tuple([i.tolist() for i in tup])
    elif type(tup) == list:
        return [i.tolist() for i in tup]


def round_array_with_none(arr, digits):
    """
    Round a numpy array to the specified number of digits. The array is allowed to include
    None values, which are preserved as such.
    """
    for idx, x in np.ndenumerate(arr):
        if x is not None:
            arr[idx] = round(x, digits)
    return arr
