import torch
import torch.nn as nn
from models.comboptnet import CombOptNetModule


class Knapsack(nn.Module):
    """
    Torch module for the comboptnet emulation of the knapsack problem.
    The target function can be handed over as a parameter to the constructor.

    objective:      find the best combination of items (value) to take into the backpack
    subject to:     do not violate the total amount of space in the backpack

    :param target: The target function, can either be the objective function 'cost' or the optimal solution 'opt'.
    :param max_copies: The maximum number of each item which can be put into the knapsack (default: 1).
    """

    def __init__(self, target='opt', max_copies=1):
        super().__init__()
        var_ranges = {'lb': 0, 'ub': max_copies}

        self.ilp_module = CombOptNetModule(var_ranges, tau=0.5)

        self.target = target

    def forward(self, A, b, c):
        """
        :param A: vector of item weights.
        :param b: total capacity of the knapsack.
        :param c: vector of item values.
        :return: target function applied to the optimal solution of the LP.
        """
        # make c and b negative to be in line with the ilp formulation of comboptnet
        c = -c.view(1, -1)
        b = -b
        constraints = torch.hstack((A, b))
        constraints = constraints.unsqueeze(0)

        solution = self.ilp_module.forward(c, constraints)

        if self.target == 'opt':
            result = solution
        elif self.target == 'cost':
            result = solution.mul(c).sum()
        else:
            raise ValueError("Unknown objective function")

        return result
