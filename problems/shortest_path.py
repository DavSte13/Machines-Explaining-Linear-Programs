import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from cvxpylayers.torch import CvxpyLayer


class ShortestPath(nn.Module):
    """
    Torch module for the cvxpylayers emulation of the shortest path problem.
    The objective function can be handed over as a parameter to the constructor.

    objective:    find the shortest path from source to target
    subject to:
    """

    def __init__(self, m, n, objective='cost'):
        super().__init__()
        self.m = m  # number of edges
        self.n = n  # number of nodes

        x = cp.Variable(m)

        A = cp.Parameter(shape=(n, m))
        A_out = cp.Parameter(shape=(n, m))
        b = cp.Parameter(n)
        c = cp.Parameter(m)

        obj = cp.Minimize(c.T @ x)
        constr = [x >= 0,
                  A @ x == b,
                  A_out @ x <= np.ones(n)]

        prob = cp.Problem(obj, constr)
        assert prob.is_dpp()

        self.cvxpylayer = CvxpyLayer(prob, parameters=[A, A_out, b, c], variables=[x])

        self.objective = objective

    def forward(self, A, b, c):
        """
        :param A: Incidence matrix for all nodes
        :param b: Vector of zeros with a one at the source and a -1 at the target indices
        :param c: edge weights
        :return: objective function applied to the optimal solution of the LP
        """

        A_out = A.clip(0)  # matrix of outgoing edges (per node)

        solution, = self.cvxpylayer.forward(A, A_out, b, c)

        if self.objective == 'opt':
            result = solution
        elif self.objective == 'cost':
            result = solution.mul(c).sum()
        elif self.objective == 'sum':
            result = solution.sum()
        elif self.objective == 'l2':
            result = solution.dist(torch.tensor(np.zeros(self.m)))
        else:
            raise ValueError("Unknown objective function")

        return result

# setup the graph
V = np.arange(5)
E = np.array([(0, 1, {'w': 1.}), (0, 2, {'w': 2.}), (1, 3, {'w': 3.}), (2, 3, {'w': 1.})])
n, m = len(V), len(E)

G = nx.DiGraph()
G.add_nodes_from(V)
G.add_edges_from(E)
# s: source node, t: target node
s, t = 0, 3


in_mat = - nx.incidence_matrix(G, oriented=True).toarray()
b = np.zeros(n)
b[s] = 1.
b[t] = -1.

# third constraint (sum of outgoing edges per nodes <= 1)
A_outgoing = in_mat.clip(0)  # matrix of outgoing edges (per node)

# list of edge weights
c = np.array([w for (u, v, w) in G.edges.data('w')])

A_tch = torch.tensor(in_mat, requires_grad=True)
b_tch = torch.tensor(b, requires_grad=True)
c_tch = torch.tensor(c, requires_grad=True)

sp = ShortestPath(m, n, objective='opt')
result = sp.forward(A_tch, b_tch, c_tch)
print("Result:", result)

jac_A, jac_b, jac_c = torch.autograd.functional.jacobian(sp, (A_tch, b_tch, c_tch))

# print(jac_A)
# print(jac_b)
print(jac_c)

