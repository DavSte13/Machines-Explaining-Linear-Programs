import networkx as nx
import numpy as np
import torch
from matplotlib import pyplot as plt


def convert_graph(nodes, edges, problem_type, plot=True):
    """
    Converts the given graph into parameters a, b and c for a LP
    problem_type specifies if the graph should be converted for the maximum flow 'mf' or the shortest path problem 'sp'
    Source and target nodes are expected to be the first and last node respectively
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
    return np.round(tensor.detach().numpy(), n_digits)


def round_tuple(tuple, n_digits=4):
    if type(tuple[0]) == torch.Tensor:
        return [round_tensor(i, n_digits) for i in tuple]
    else:
        return [np.round(i, n_digits) for i in tuple]


def torch_delete(tensor, indices):
    mask = torch.ones(tensor.numel(), dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]


def detach_tuple(tup, to_list=False):
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
    if type(tup) == tuple:
        return tuple([i.tolist() for i in tup])
    elif type(tup) == list:
        return [i.tolist() for i in tup]


def round_array_with_none(arr, digits):
    for idx, x in np.ndenumerate(arr):
        if x is not None:
            arr[idx] = round(x, digits)
    return arr

