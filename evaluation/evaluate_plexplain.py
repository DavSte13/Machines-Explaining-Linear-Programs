import torch
import numpy as np
import argparse
from problems.plexplain import PlexPlain
from attribution_methods.gradientxinput import GradientXInput
from attribution_methods.integrated_gradients import IntegratedGradients

parser = argparse.ArgumentParser(description='Evaluate attribution methods on the plexplain.')
parser.add_argument("method", help="Select the method to evaluate on the plexplain LP. One of: Gradients 'grad', "
                                   "Gradient times Input 'gxi', Integrated Gradients 'ig' or Occlusion 'occ'.",
                    choices=['grad', 'gxi', 'ig', 'occ'])

args = parser.parse_args()


def print_nice(inp, res, attr, base=None, is_ig=False, is_opt=False):
    if not is_ig:
        attr = attr.detach().numpy()
    else:
        base = base.detach().numpy()

    inp = inp.detach().numpy()

    print("============= Input =============")
    print("Cost PV:\t\t", inp[0])
    print("Cost bat:\t\t", inp[1])
    print("Cost energy:\t", inp[2])
    print("Total demand:\t", inp[3])

    print("\n============= Result =============")
    print(res.detach().numpy())

    if not is_opt:
        print("\n============= Attributions =============")
        print("Cost PV:\t\t", attr[0])
        print("Cost bat:\t\t", attr[1])
        print("Cost energy:\t", attr[2])
        print("Total demand:\t", attr[3])
    else:
        print("\n============= Attributions =============")
        print(attr)
        print("Cap bat:\t\t", attr[-2])
        print("Cap pv:\t\t", attr[-1])

    if is_ig:
        print("\n============= Baseline =============")
        print("Cost PV:\t\t", base[0])
        print("Cost bat:\t\t", base[1])
        print("Cost energy:\t", base[2])
        print("Total demand\t:", base[3])


def evaluate_plexplain(method):
    """
    computes attributions for the plexplain LP.
    For the gradient based methods, the attributions are computed on 8-hour intervals. For occlusion, the
    attributions for each month are computed.

    :param method: One of ('grad', 'gxi', 'ig', 'occ')
    """
    # define the model and the base parameters
    plexplain = PlexPlain(use_days=True)
    gxi = GradientXInput(plexplain)
    ig = IntegratedGradients(plexplain)

    lifetime = 10
    cost_pv = 1000.
    cost_bat = 300.
    cost_buy = 0.25
    dem_tot = 3500.

    base = torch.tensor([cost_pv / 100 / lifetime, cost_bat / 100 / lifetime, cost_buy / 100, dem_tot / 100],
                        requires_grad=True)

    inputs = torch.tensor([cost_pv / lifetime, cost_bat / lifetime, cost_buy, dem_tot], requires_grad=True)
    tensor_inputs = (torch.tensor([cost_pv / lifetime], requires_grad=True),
                     torch.tensor([cost_bat / lifetime], requires_grad=True),
                     torch.tensor([cost_buy / lifetime], requires_grad=True),
                     torch.tensor([dem_tot / lifetime], requires_grad=True))

    if method == 'grad':
        # solve the problem
        result = plexplain.forward(inputs, False)
        # compute the gradients for the model
        result.backward()
        attributions_grad = inputs.grad
        print_nice(inputs, result, attributions_grad)

    if method == 'gxi':
        # solve the problem
        result = plexplain.forward(inputs, False)
        # compute the gradient times input attributions for the model
        attributions_gxi, = gxi.attribute([inputs], no_jacobian=True)
        print_nice(inputs, result, attributions_gxi)

    if method == 'ig':
        # solve the problem
        result = plexplain.forward(inputs, False)
        # compute the IG attributions for the model
        attributions_ig, = ig.attribute(inputs, base, no_jacobian=True, steps=20, plexplain=True)[0]
        print_nice(inputs, result, attributions_ig, base, is_ig=True)

    if method == 'occ':
        # solve the problem
        result = plexplain.forward(inputs, False)
        solution = plexplain.forward(inputs, True)

        # compute the occlusion attributions
        cap_pv = solution[6]
        cap_bat = solution[5]
        opex = torch.sum(solution[4]) * cost_buy
        own_gen = torch.sum(solution[0]) / dem_tot
        print(f"============= Cap PV =============\n{cap_pv}")
        print(f"============= Cap bat =============\n{cap_bat}")
        print(f"============= OPEX =============\n{opex}")
        print(f"============= Own gen =============\n{own_gen}")

        # computing the attribution values per month
        occ = np.ones([12, 5])
        for m in range(12):
            reduced_model = PlexPlain(reduce_dimension=True, month=m + 1)
            red_result, red_solution = reduced_model.forward(inputs, False), reduced_model.forward(inputs, True)

            occ[m][0] = result - red_result
            occ[m][1] = cap_pv - red_solution[6]
            occ[m][2] = cap_bat - red_solution[5]
            occ[m][3] = opex - torch.sum(red_solution[4]) * cost_buy
            occ[m][4] = own_gen - torch.sum(red_solution[0]) / dem_tot

        np.set_printoptions(suppress=True)

        print("============= Occlusion attributions: =============")
        print("\tTOTEX\tcap PV\t cap bat OPEX \t  own generation")
        print(occ.round(4))


evaluate_plexplain(args.method)
