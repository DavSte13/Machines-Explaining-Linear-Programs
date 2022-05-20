import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn
from cvxpylayers.torch import CvxpyLayer


class Knapsack(nn.Module):
    """
    Torch module for the cvxpylayers emulation of the knapsack problem.
    The objective function can be handed over as a parameter to the constructor.

    === This problem is an integer problem, but the following module will treat it as a
        linear program. There exists a separate ILP module. ===

    objective:      find the best combination of items (value) to take into the backpack
    subject to:     do not violate the total amount of space in the backpack
    """

    def __init__(self, n, max_copies=1, objective='opt'):
        super().__init__()
        self.n = n  # number of items

        x = cp.Variable(n)

        A = cp.Parameter(n)
        b = cp.Parameter(1)
        c = cp.Parameter(n)

        obj = cp.Maximize(c.T @ x)
        constr = [x >= 0,
                  A.T @ x <= b,
                  x <= np.ones(n) * max_copies]

        prob = cp.Problem(obj, constr)
        assert prob.is_dpp()

        self.cvxpylayer = CvxpyLayer(prob, parameters=[A, b, c], variables=[x])

        self.objective = objective

    def forward(self, A, b, c):
        """
        :param A: vector of item weights
        :param b: total capacity of the knapsack
        :param c: vector of item values
        :return: objective function applied to the optimal solution of the LP
        """

        solution, = self.cvxpylayer.forward(A, b, c)

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
