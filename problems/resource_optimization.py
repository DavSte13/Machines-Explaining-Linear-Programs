import math

import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn
from cvxpylayers.torch import CvxpyLayer

import matplotlib.pyplot as plt


class BasicLP(nn.Module):
    """
    Torch module for the cvxpylayers emulation of the basic linear problem.
    The objective function can later on be handed over as a parameter to the constructor.
    reduce_dimensions can be used for granger causal attributions (reduces the number of constraints by 1)
    m can be used to specify the number of constraints (if not given, it is assumed to be the same as n).
    """

    def __init__(self, n, objective='cost', reduce_dimension=False, m=None, x_geq_zero=True, maximize=True):
        super().__init__()

        if m is None:
            m = n
        self.n = n
        self.m = m
        # inititalize the CVXPY problem
        # create the variables and parameters
        x = cp.Variable(n)
        if not reduce_dimension:
            b = cp.Parameter(shape=(m,))
            A = cp.Parameter(shape=(m, n))
        else:
            b = cp.Parameter(shape=(m - 1,))
            A = cp.Parameter(shape=(m - 1, n))
            self.m -= 1

        c = cp.Parameter(shape=n)

        if x_geq_zero:
            constraints = [A @ x <= b,
                                x >= np.zeros(n)]
        else:
            constraints = [A @ x <= b]

        # Form objective and problem
        if maximize:
            obj = cp.Maximize(c.T @ x)
        else:
            obj = cp.Minimize(c.T @ x)
        prob = cp.Problem(obj, constraints)

        assert prob.is_dpp()

        self.cvxpylayer = CvxpyLayer(prob, parameters=[A, b, c], variables=[x])

        self.objective = objective

    def forward(self, A, b, c):
        """
        :param A: value for the parameter A (as torch tensor)
        :param b: value for the parameter b (as torch tensor)
        :return: The objective function applied to the optimal solution of the LP
        """

        solution,  = self.cvxpylayer.forward(A, b, c)

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

    def base_plot(self, A, b, c, title=None):
        """
        Create a base plot for the problem and the constraints. Returns the plot object as well as several intermediate
        results (e.g. the constraint lines)
        """
        # set the objective to "optimal solution"
        if self.objective != 'opt':
            previous_obj = self.objective
            self.objective = 'opt'
        else:
            previous_obj = self.objective

        # find the optimal solution
        opt_solution = self.forward(A, b, c).detach().numpy()

        # convert torch parameters to numpy arrays for further use
        A = A.detach().numpy()
        b = b.detach().numpy()
        c = c.detach().numpy()

        # define the grid
        grid_steps = 34
        x1 = np.linspace(-2, 15, grid_steps)
        x2 = np.linspace(-2, 15, grid_steps)
        xx, yy = np.meshgrid(x1, x2, sparse=False)

        # plot the axis
        plt.axvline(0, color='black', linewidth='0.5')
        plt.axhline(0, color='black', linewidth='0.5')
        plt.axis([-2, 15, -2, 15])

        # compute and plot the constraints
        color_list = ['tab:pink', 'tab:red', 'tab:green', 'tab:olive', 'tab:purple', 'tab:brown', 'tab:gray']
        constraint_information = []     # track the start, median and end point of each constraint
        for i in range(self.m):
            if A[i][1] == 0:
                tmp = np.ones(x2.shape) * b[i]
                plt.plot(tmp, x2, label=f'Constraint {i + 1}', color=color_list[i])
                constraint_information.append([[tmp[0], x2[0]], [tmp[grid_steps - 1], x2[grid_steps - 1]]])
            else:
                tmp = (b[i] - x1 * A[i][0]) / A[i][1]
                plt.plot(x1, tmp, label=f'Constraint {i + 1}', color=color_list[i])
                constraint_information.append([[x1[0], tmp[0]], [x1[grid_steps - 1], tmp[grid_steps - 1]]])

        # compute and plot the general solution value
        sol = xx * c[0] + yy * c[1]
        cf = plt.pcolormesh(xx, yy, sol, shading='auto')
        plt.colorbar(cf)

        # plot the optimal solution
        plt.scatter(opt_solution[0], opt_solution[1], c='tab:orange', s=20, label='Optimal solution', zorder=10)

        if title is not None:
            plt.title(title)

        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend()

        # reset the objective
        self.objective = previous_obj

        return plt, constraint_information, color_list

    def plot(self, A, b, c, title=None):
        """
        Plot the problem (this is just a function which calls show on the base_plot
        """

        self.base_plot(A, b, c, title)[0].show()

    def plot_attributions(self, A, b, c, attributions, title=None):
        """
        Extend the base plot of the problem by additional information about the attributions.
        Currently only works for cost attributions
        :param attributions: tuple of attributions for A and b
        """
        attr_A = attributions[0]
        attr_b = attributions[1]
        # normalize the attributions to a new interval (minimum and maximum size of an attribution vector
        old_max = np.max([np.max(np.abs(attr_A)), np.max(np.abs(attr_b))])
        old_min = np.min([np.min(np.abs(attr_A)), np.min(np.abs(attr_b))])
        new_min, new_max = 0.5, 2

        # linear normalization
        attr_A_norm = np.multiply(np.sign(attr_A),
                                  (np.abs(attr_A) - old_min) * (new_max - new_min) / (old_max - old_min) + new_min)
        attr_b_norm = np.multiply(np.sign(attr_b),
                                  (np.abs(attr_b) - old_min) * (new_max - new_min) / (old_max - old_min) + new_min)

        plot, c_inf, colors = self.base_plot(A, b, c, title)

        # plot the attributions for all constraints:
        for i in range(self.m):
            # check that the constraint is not parallel to the x1 axis or x2 axis
            if (not c_inf[i][0][0] == c_inf[i][1][0]) and (not c_inf[i][0][1] == c_inf[i][1][1]):
                # compute gradient of the constraint (image-wise)
                m = (c_inf[i][1][1] - c_inf[i][0][1])/(c_inf[i][1][0] - c_inf[i][0][0])
                # intersection with x2-axis:
                b = c_inf[i][0][1]

                # plot the arrows on the axis lines if the attributions are not zero:
                if not attr_A[i][0] == 0:
                    plot.arrow(-b/m, 0, attr_A_norm[i][0], 0.1, length_includes_head=True, color=colors[i],
                               head_width=0.3, width=0.01)
                if not attr_A[i][1] == 0:
                    plot.arrow(0, b, 0.1, attr_A_norm[i][1], length_includes_head=True, color=colors[i], head_width=0.3,
                               width=0.01)

                # plot the arrow for b:
                if not attr_b[i] == 0:
                    length = math.sqrt(1/2)*abs(attr_b_norm[i])
                    sign = np.sign(attr_b_norm[i])
                    plot.arrow(-b/(2 * m), b/2, sign * length, sign * length, length_includes_head=True,
                               color=colors[i], head_width=0.2)

            elif c_inf[i][0][0] == c_inf[i][1][0]:
                # plot arrows on the x1 axis line/the upper end of the plot
                if not attr_A[i][0] == 0:
                    plot.arrow(c_inf[i][0][0], 0, attr_A_norm[i][0], 0.1, length_includes_head=True, color=colors[i],
                               head_width=0.3, width=0.01)
                if not attr_A[i][1] == 0:
                    plot.arrow(c_inf[i][1][0], c_inf[i][1][1], attr_A_norm[i][1], -0.1, length_includes_head=True,
                               color=colors[i], head_width=0.3, width=0.01)

                # plot the arrow for b:
                if not attr_b[i] == 0:
                    plot.arrow(c_inf[i][0][0], c_inf[i][1][1]/2, attr_b_norm[i], 0, length_includes_head=True,
                               color=colors[i], head_width=0.2)

            elif c_inf[i][0][0] == c_inf[i][1][0]:
                # plot arrows on the x2 axis line/the right end of the plot
                if not attr_A[i][0] == 0:
                    plot.arrow(0, c_inf[i][0][1], 0.1, attr_A_norm[i][1], length_includes_head=True, color=colors[i],
                               head_width=0.2)
                if not attr_A[i][1] == 0:
                    plot.arrow(c_inf[i][1][0], c_inf[i][1][1], -0.1, attr_A_norm[i][0], length_includes_head=True,
                               color=colors[i], head_width=0.2)

                # plot the arrow for b:
                if not attr_b[i] == 0:
                    plot.arrow(c_inf[i][1][0]/2, c_inf[i][0][1], 0, attr_b_norm[i], length_includes_head=True,
                               color=colors[i], head_width=0.2)

        plot.show()

    def plot_with_gradients(self, A, b, c, A_new=None, b_new=None, include_gradients=False, show_grad=()):
        """
        Solve the problem and compute the gradients w.r.t. the optimal solution.
        Plot both the problem and the given gradients
        """
        # set the objective to "optimal solution"
        if self.objective != 'opt':
            previous_obj = self.objective
            self.objective = 'opt'
        else:
            previous_obj = self.objective

        # find the optimal solution
        opt_solution = self.forward(A, b, c).detach().numpy()

        if include_gradients:
            jac_A, jac_b, _ = torch.autograd.functional.jacobian(self, (A, b, c))
            jac_A = jac_A.detach().numpy()
            jac_b = jac_b.detach().numpy()

        if A_new is None:
            A_new = A
        if b_new is None:
            b_new = b

        opt_solution_new = self.forward(A_new, b_new, c).detach().numpy()

        A = A.detach().numpy()
        b = b.detach().numpy()
        c = c.detach().numpy()
        A_new = A
        b_new = b

        # define the grid
        x1 = np.linspace(-2, 15, 35)
        x2 = np.linspace(-2, 15, 35)
        xx, yy = np.meshgrid(x1, x2, sparse=False)

        # compute the constraints
        x2_c1 = (b[0] - x1 * A[0][0]) / A[0][1]
        x2_c2 = (b[1] - x1 * A[1][0]) / A[1][1]

        if include_gradients:
            x2_c1_new = (b_new[0] - x1 * A_new[0][0]) / A_new[0][1]
            x2_c2_new = (b_new[1] - x1 * A_new[1][0]) / A_new[1][1]

        # compute the general solution value
        sol = xx * c[0] + yy * c[1]

        # plot the axis
        plt.axvline(0, color='black', linewidth='0.5')
        plt.axhline(0, color='black', linewidth='0.5')
        plt.axis([-2, 15, -2, 15])

        # plot the constraints
        plt.plot(x1, x2_c1, label='Constraint 1', color='magenta')
        plt.plot(x1, x2_c2, label='Constraint 2', color='red')

        if include_gradients:
            if not np.array_equal(x2_c1_new, x2_c1):
                plt.plot(x1, x2_c1_new, label='New constraint 1', color='magenta', linestyle='--')
            if not np.array_equal(x2_c2_new, x2_c2):
                plt.plot(x1, x2_c2_new, label='New constraint 2', color='red', linestyle='--')

        # plot the solution value
        cf = plt.pcolormesh(xx, yy, sol, shading='auto')
        plt.colorbar(cf)

        if include_gradients:
            # plot the gradient
            if "b0" in show_grad:
                if not np.isclose(jac_b[0][0], jac_b[1][0], 0, atol=1e-04):
                    head = 0.15
                else:
                    head = 0.05
                plt.arrow(opt_solution[0], opt_solution[1], jac_b[0][0], jac_b[1][0],
                          length_includes_head=True, zorder=12, color='white', head_width=head, label='Gradient b0')

            if "b1" in show_grad:
                if not np.isclose(jac_b[0][1], jac_b[1][1], 0, atol=1e-04):
                    head = 0.15
                else:
                    head = 0.05
                plt.arrow(opt_solution[0], opt_solution[1], jac_b[0][1], jac_b[1][1],
                          length_includes_head=True, zorder=12, color='silver', head_width=head, label='Gradient b1')

            if "a00" in show_grad:
                if not np.isclose(jac_A[0][0][0], jac_A[1][0][0], 0, atol=1e-04):
                    head = 0.15
                else:
                    head = 0.05
                plt.arrow(opt_solution[0], opt_solution[1], jac_A[0][0][0], jac_A[1][0][0],
                          length_includes_head=True, zorder=12, color='coral', head_width=head, label='Gradient a00')

            if "a01" in show_grad:
                if not np.isclose(jac_A[0][0][1], jac_A[1][0][1], 0, atol=1e-04):
                    head = 0.15
                else:
                    head = 0.05
                plt.arrow(opt_solution[0], opt_solution[1], jac_A[0][0][1], jac_A[1][0][1],
                          length_includes_head=True, zorder=12, color='peru', head_width=head, label='Gradient a01')

            if "a10" in show_grad:
                if not np.isclose(jac_A[0][1][0], jac_A[1][1][0], 0, atol=1e-04):
                    head = 0.15
                else:
                    head = 0.05
                plt.arrow(opt_solution[0], opt_solution[1], jac_A[0][1][0], jac_A[1][1][0],
                          length_includes_head=True, zorder=12, color='lime', head_width=head, label='Gradient a10')

            if "a11" in show_grad:
                if not np.isclose(jac_A[0][1][1], jac_A[1][1][1], 0, atol=1e-04):
                    head = 0.15
                else:
                    head = 0.05
                plt.arrow(opt_solution[0], opt_solution[1], jac_A[0][1][1], jac_A[1][1][1],
                          length_includes_head=True, zorder=12, color='cyan', head_width=head, label='Gradient a11')

        # plot the optimal solution
        plt.scatter(opt_solution[0], opt_solution[1], c='tab:orange', s=30, label='Optimal solution', zorder=10)

        if include_gradients:
            plt.scatter(opt_solution_new[0], opt_solution_new[1], c='tab:orange', s=30, label='New Optimal solution',
                        zorder=10, marker='s')

        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend()
        plt.show()

        # reset the objective
        self.objective = previous_obj
