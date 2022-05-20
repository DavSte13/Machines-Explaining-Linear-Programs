import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn

from cvxpylayers.torch import CvxpyLayer


class DietProblem(nn.Module):
    """
       Torch module for the cvxpylayers emulation of the diet problem.
       The objective function can be handed over as a parameter to the constructor.

        objective:    minimize the cost of food per day
        subject to:   get enough of each nutrient
                      do not eat too much of an individual food
       """

    def __init__(self, num_foods, num_nutrients, objective='cost'):
        super().__init__()

        # number of different foods:
        self.n = num_foods
        # number of different nutrients:
        self.m = num_nutrients

        # create the cvxpy parameters
        c = cp.Parameter(nonneg=True, shape=self.n)  # cost of each food
        A = cp.Parameter(nonneg=True, shape=(self.m, self.n))  # nutrients per food
        b = cp.Parameter(nonneg=True, shape=self.m)  # minimum nutrients
        lim = cp.Parameter(nonneg=True, shape=self.n)  # maximum amounts per food

        # setup the problem:
        x = cp.Variable(self.n)
        constraints = [A @ x >= b,
                       x <= lim,
                       x >= np.zeros(self.n)]
        obj = cp.Minimize(c.T @ x)
        prob = cp.Problem(obj, constraints)

        assert prob.is_dpp()

        self.cvxpylayer = CvxpyLayer(prob, parameters=[A, b, c, lim], variables=[x])

        self.objective = objective

    def forward(self, A, b, c, lim):
        """
        :param A: nutrients per food
        :param b: minimum nutrients
        :param c: cost of each food
        :param lim: maxmimum amounts for each food
        :return: objective function applied to the optimal solution of the LP
        """

        solution, = self.cvxpylayer.forward(A, b, c, lim)

        if self.objective == 'opt':
            result = solution
        elif self.objective == 'cost':
            result = solution.mul(c).sum()
        elif self.objective == 'sum':
            result = solution.sum()
        elif self.objective == 'l2':
            result = solution.dist(torch.tensor(np.zeros(self.n)))
        else:
            raise ValueError("Unknown objective function")

        return result


n, m = 4, 1

A_tch = torch.tensor(np.random.random((m, n)), requires_grad=True)
b_tch = torch.tensor(np.ones(m), requires_grad=True)
c_tch = torch.tensor(np.random.random(n).T, requires_grad=True)
l_tch = torch.tensor(np.random.random(n), requires_grad=True)

diet_problem = DietProblem(4, 1, 'opt')
result = diet_problem.forward(A_tch, b_tch, c_tch, l_tch)


print("Result:", result)

# compute the gradients:
jac_A, jac_b, jac_c, jac_l = torch.autograd.functional.jacobian(diet_problem, (A_tch, b_tch, c_tch, l_tch))

# print("Gradients wrt A:", jac_A)
# print("Gradients wrt b:", jac_b)
# print("Gradients wrt c:", jac_c)
# print("Gradients wrt lim:", jac_l)

