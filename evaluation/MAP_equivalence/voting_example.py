import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn
from cvxpylayers.torch import CvxpyLayer


class MAP_LP(nn.Module):
    """
    Torch module for the cvxpylayers emulation of the basic linear problem.
    The objective function can later on be handed over as a parameter to the constructor.
    reduce_dimensions can be used for granger causal attributions (reduces the number of constraints by 1)
    m can be used to specify the number of constraints (if not given, it is assumed to be the same as n).
    """

    def __init__(self, objective='cost', reduce_edge=False):
        super().__init__()

        if reduce_edge:
            mu_dim = 10
        else:
            mu_dim = 14
        # inititalize the CVXPY problem
        # create the variables and parameters
        mu = cp.Variable(mu_dim)

        theta = cp.Parameter(shape=mu_dim)

        constraints = [
            mu >= np.zeros(mu_dim),
            mu <= np.ones(mu_dim),
            mu[0] + mu[1] == 1,
            mu[2] + mu[3] == 1,
            mu[4] + mu[5] == 1,
            mu[2] == mu[6] + mu[7],
            mu[3] == mu[8] + mu[9],
            mu[4] == mu[6] + mu[8],
            mu[5] == mu[7] + mu[9]
            ]

        if not reduce_edge:
            constraints.extend([
            mu[0] == mu[6] + mu[7],
            mu[1] == mu[8] + mu[9],
            mu[2] == mu[6] + mu[8],
            mu[3] == mu[7] + mu[9],

            mu[2] == mu[10] + mu[11],
            mu[3] == mu[12] + mu[13],
            mu[4] == mu[10] + mu[12],
            mu[5] == mu[11] + mu[13]
            ])

        obj = cp.Maximize(theta.T @ mu)
        prob = cp.Problem(obj, constraints)

        assert prob.is_dpp()

        self.cvxpylayer = CvxpyLayer(prob, parameters=[theta], variables=[mu])

        self.objective = objective

    def forward(self, theta):
        """
        :param theta: value for the parameter A (as torch tensor)
        :return: The objective function applied to the optimal solution of the LP
        """

        solution,  = self.cvxpylayer.forward(theta)

        if self.objective == 'opt':
            result = solution
        elif self.objective == 'cost':
            result = solution.mul(theta).sum()
        else:
            raise ValueError("Unknown objective function")

        return result


map_cost = MAP_LP('cost')
map_opt = MAP_LP('opt')
theta = torch.tensor([1., 0.3, 0.9, 0.9, 0.7, 1., 1.3, 0., 0., 1.3, 1.3, 0., 0., 1.3], requires_grad=True)
theta_masked = torch.tensor([1., 0.3, 0.9, 0.9, 0.7, 1., 1.3, 0., 0., 1.3], requires_grad=True)

map_cost_reduced = MAP_LP('opt', True)
masked_res = map_cost_reduced.forward(theta_masked)
print(masked_res)

res = map_cost.forward(theta)
print("Result:\n", res, "\n")
res.backward()
grad = np.around(theta.grad.detach().numpy(), 4)
print("Objective map derivative:\n", grad, "\n")

jac = torch.autograd.functional.jacobian(map_opt, theta)
print("Solution map derivative:\n", np.around(jac.detach().numpy(), 4))
