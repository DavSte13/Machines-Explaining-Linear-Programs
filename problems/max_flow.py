import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn
from cvxpylayers.torch import CvxpyLayer


class MaxFlow(nn.Module):
    """
    Torch module for the cvxpylayers emulation of the maximum flow problem.
    The objective function can be handed over as a parameter to the constructor.

    objective:    maximize the flow through the network
    subject to:   incoming and outgoing flow is the same for all nodes except source and sink
                  the flow at each edge does not exceed its capacity
    """

    def __init__(self, m, n, objective='cost', reduce_edges=False):
        super().__init__()
        if reduce_edges:
            m = m - 1

        self.m = m  # number of edges
        self.n = n  # number of nodes

        x = cp.Variable(m)

        c = cp.Parameter(shape=m)
        b = cp.Parameter(shape=m)
        A = cp.Parameter(shape=(n - 2, m))

        obj = cp.Maximize(-c.T @ x)
        constraints = [x >= np.zeros(x.shape),
                       x <= b,
                       A @ x == np.zeros(n - 2)]

        prob = cp.Problem(obj, constraints)
        assert prob.is_dpp()

        self.cvxpylayer = CvxpyLayer(prob, parameters=[A, b, c], variables=[x])

        self.objective = objective

    def forward(self, A, b, c):
        """
        :param A: Incidence matrix for all nodes except source and sink
        :param b: capacity limits for each edge
        :param c: Incidence matrix row for the sink
        :return: objective function applied to the optimal solution of the LP
        """

        solution, = self.cvxpylayer.forward(A, b, c)

        if self.objective == 'opt':
            result = solution
        elif self.objective == 'cost':
            result = solution.mul(-c).sum()
        elif self.objective == 'sum':
            result = solution.sum()
        elif self.objective == 'l2':
            result = solution.dist(torch.tensor(np.zeros(self.m)))
        else:
            raise ValueError("Unknown objective function")

        return result

