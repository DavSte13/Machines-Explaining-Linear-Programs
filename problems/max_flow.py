import cvxpy as cp
import numpy as np
import torch.nn as nn
from cvxpylayers.torch import CvxpyLayer


class MaxFlow(nn.Module):
    """
    Torch module for the cvxpylayers emulation of the maximum flow problem.
    The objective function can be handed over as a parameter to the constructor.

    objective:    maximize the flow through the network
    subject to:   incoming and outgoing flow is the same for all nodes except source and sink
                  the flow at each edge does not exceed its capacity

    :param m: Number of edges
    :param n: Number of nodes
    :param target: Target function for the LP, possible values are 'opt' for the optimal solution
      and 'cost' for the objective function.
    :param reduce_dimension: If the dimension of the problem should be reduced. This is used for
      the occlusion attributions, and effectively removes one edge from the problem.
    """

    def __init__(self, m, n, target='cost', reduce_dimension=False):
        super().__init__()
        if reduce_dimension:
            m = m - 1

        self.m = m  # number of edges
        self.n = n  # number of nodes

        # create the cvxpy problem formulation
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

        self.target = target

    def forward(self, A, b, c):
        """
        :param A: Incidence matrix for all nodes except source and sink
        :param b: capacity limits for each edge
        :param c: Incidence matrix row for the sink
        :return: target function of the LP
        """

        solution, = self.cvxpylayer.forward(A, b, c)

        if self.target == 'opt':
            result = solution
        elif self.target == 'cost':
            result = solution.mul(-c).sum()
        else:
            raise ValueError("Unknown objective function")

        return result
