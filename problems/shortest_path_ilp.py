import torch
import torch.nn as nn
from CombOptNet.models.comboptnet import CombOptNetModule
import numpy as np


class ShortestPath(nn.Module):
    """
    Torch module for the cvxpylayers emulation of the shortest path problem.
    The objective function can be handed over as a parameter to the constructor.

    objective:    find the shortest path from source to target

    :param target: The target function, can either be the objective function 'cost' or the optimal solution 'opt'.
    """

    def __init__(self, target='cost'):
        super().__init__()

        self.ilp_module = CombOptNetModule(variable_range={'lb': 0, 'ub': 1}, tau=0.5)

        self.target = target

    def forward(self, A, b, c):
        """
        :param A: Incidence matrix for all nodes.
        :param b: Vector of zeros with a one at the source and a -1 at the target indices.
        :param c: Vector of all edge weights.
        :return: Target function applied to the optimal solution of the LP.
        """
        c = c.view(1, -1)
        constraints_1 = torch.hstack((A, b))
        constraints_2 = -constraints_1
        constraints = torch.vstack((constraints_1, constraints_2))

        solution = self.ilp_module.forward(c, constraints)

        if self.target == 'opt':
            result = solution
        elif self.target == 'cost':
            result = solution.mul(c).sum()
        elif self.target == 'sum':
            result = solution.sum()
        elif self.target == 'l2':
            result = solution.dist(torch.tensor(np.zeros(self.m)))
        else:
            raise ValueError("Unknown objective function")

        return result
