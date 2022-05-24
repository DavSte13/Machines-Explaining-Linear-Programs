import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn
from cvxpylayers.torch import CvxpyLayer
import argparse

parser = argparse.ArgumentParser(description='Show the attribution methods results on the MAP example.')
args = parser.parse_args()


class MapLp(nn.Module):
    """
    Torch module for the MAP voting example. The LP emulates a small graph with three nodes. Each node has its
    individual preference to vote either for 1 or 0. Two nodes connected by edges are more likely to vote for the
    same result.

    :param target: The target function, can either be the objective function 'cost' or the optimal solution 'opt'.
    :param reduce_edge: 'no': do not remove any edges, '01': remove the edge between x0 and x1, '12': remove
      the edge between x1 and x2
    """

    def __init__(self, target='cost', reduce_edge='no'):
        super().__init__()

        if reduce_edge != 'no':
            mu_dim = 10
        else:
            mu_dim = 14
        # initialize the CVXPY problem
        # create the variables and parameters
        mu = cp.Variable(mu_dim)
        theta = cp.Parameter(shape=mu_dim)

        # consistency of the nodes
        constraints = [
            mu >= np.zeros(mu_dim),
            mu <= np.ones(mu_dim),
            mu[0] + mu[1] == 1,
            mu[2] + mu[3] == 1,
            mu[4] + mu[5] == 1
        ]
        # consistency of the edges
        if reduce_edge == 'no':
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
        elif reduce_edge == '01':
            constraints.extend([
                mu[2] == mu[6] + mu[7],
                mu[3] == mu[8] + mu[9],
                mu[4] == mu[6] + mu[8],
                mu[5] == mu[7] + mu[9]
            ])
        elif reduce_edge == '12':
            constraints.extend([
                mu[0] == mu[6] + mu[7],
                mu[1] == mu[8] + mu[9],
                mu[2] == mu[6] + mu[8],
                mu[3] == mu[7] + mu[9]
            ])
        else:
            raise ValueError("Unknown value for reduce_edge given.")

        obj = cp.Maximize(theta.T @ mu)
        prob = cp.Problem(obj, constraints)

        assert prob.is_dpp()

        self.cvxpylayer = CvxpyLayer(prob, parameters=[theta], variables=[mu])

        self.target = target

    def forward(self, theta):
        """
        :param theta: value for the parameter theta (as torch tensor).
        :return: The target function of the LP.
        """

        solution, = self.cvxpylayer.forward(theta)

        if self.target == 'opt':
            result = solution
        elif self.target == 'cost':
            result = solution.mul(theta).sum()
        else:
            raise ValueError("Unknown objective function")

        return result


def evaluate_map():
    map_cost = MapLp('cost')
    map_opt = MapLp('opt')
    theta = torch.tensor([1., 0.3, 0.9, 0.9, 0.7, 1., 1.3, 0., 0., 1.3, 1.3, 0., 0., 1.3], requires_grad=True)
    # as both edges have the same values in theta, removing either of them results in the same theta_masked.
    theta_masked = torch.tensor([1., 0.3, 0.9, 0.9, 0.7, 1., 1.3, 0., 0., 1.3], requires_grad=True)

    res_cost = map_cost.forward(theta)
    res_opt = np.around(map_opt.forward(theta).detach().numpy(), 4)
    print("Optimal solution:\n", res_opt)
    print(f"X1 = {res_opt[1]} \t"
          f"X2 = {res_opt[3]} \t"
          f"X3 = {res_opt[5]}\n")

    # occlusion attributions
    map_red01 = MapLp('opt', '01')
    map_red12 = MapLp('opt', '12')
    masked_res01 = np.around(map_red01.forward(theta_masked).detach().numpy(), 2)
    masked_res12 = np.around(map_red12.forward(theta_masked).detach().numpy(), 2)

    print("Optimal solution with masked edge 01:", masked_res01)
    print(f"X1 = {masked_res01[1]} \t"
          f"X2 = {masked_res01[3]} \t"
          f"X3 = {masked_res01[5]}")
    print(f"Occlusion attributions for edge 01: \t{res_opt[1] - masked_res01[1]}, "
          f"{res_opt[3] - masked_res01[3]}, "
          f"{res_opt[5] - masked_res01[5]}\n")

    print("Optimal solution with masked edge 02:", masked_res12)
    print(f"X1 = {masked_res12[1]} \t"
          f"X2 = {masked_res12[3]} \t"
          f"X3 = {masked_res12[5]}")
    print(f"Occlusion attributions for edge 01: \t{res_opt[1] - masked_res12[1]}, "
          f"{res_opt[3] - masked_res12[3]}, "
          f"{res_opt[5] - masked_res12[5]}\n")

    # gradients for the objective function
    res_cost.backward()
    grad = np.around(theta.grad.detach().numpy(), 4)
    print("Objective function derivative:\n", grad, "\n")
    # the gradients of the objective function do not provide further insights, as they are the same
    # as the optimal solution

    # gradients for the optimal solution
    jac = torch.autograd.functional.jacobian(map_opt, theta)
    print("Optimal solution derivative:\n", np.around(jac.detach().numpy(), 4))
    # the gradients of the optimal solutions are all zero, which causes all gradient based methods to fail.


evaluate_map()
