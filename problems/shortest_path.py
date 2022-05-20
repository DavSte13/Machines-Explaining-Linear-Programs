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
    subject to:   Constraints ensuring the graph structure is used correctly

    :param m: Number of edges.
    :param n: Number of nodes.
    :param target: Type of the target function, options are 'opt' for the optimal solution and 'cost'
      for the objective function
    """

    def __init__(self, m, n, target='cost'):
        super().__init__()
        self.m = m  # number of edges
        self.n = n  # number of nodes

        # create a cvxpy formulation of the problem.
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

        self.target = target

    def forward(self, A, b, c):
        """
        :param A: Incidence matrix for all nodes
        :param b: Vector of zeros with 1 at the source and -1 at the target indices
        :param c: edge weights
        :return: objective function applied to the optimal solution of the LP
        """

        A_out = A.clip(0)  # matrix of outgoing edges (per node)

        solution, = self.cvxpylayer.forward(A, A_out, b, c)

        if self.target == 'opt':
            result = solution
        elif self.target == 'cost':
            result = solution.mul(c).sum()
        else:
            raise ValueError("Unknown objective function")

        return result
