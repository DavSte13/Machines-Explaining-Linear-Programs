import torch
import numpy as np

import xlp_utils
from problems.plexplain import PlexPlain
from attribution_methods.gradientxinput import GradientXInput
from attribution_methods.integrated_gradients import IntegratedGradients


def print_nice(inp, res, attr, base=None, is_ig=False, is_opt=False):
    if not is_ig:
        attr = xlp_utils.detach_tuple(attr)
    else:
        base = xlp_utils.detach_tuple(base)

    inp = xlp_utils.detach_tuple(inp)

    print("============= Input =============")
    print("Cost PV:\t\t", inp[0])
    print("Cost bat:\t\t", inp[1])
    print("Cost energy:\t", inp[2])
    print("Total demand\t:", inp[3])

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
        print("Cap bat:\t\t", attr[-2])
        print("Cap pv:\t\t", attr[-1])

    if is_ig:
        print("\n============= Baseline =============")
        print("Cost PV:\t\t", base[0])
        print("Cost bat:\t\t", base[1])
        print("Cost energy:\t", base[2])
        print("Total demand\t:", base[3])


# define the model and the base parameters
plexplain = PlexPlain(use_days=True)
gxi = GradientXInput(plexplain)
ig = IntegratedGradients(plexplain)

lifetime = 10
cost_pv = torch.tensor([1000.], requires_grad=True)
cost_bat = torch.tensor([300.], requires_grad=True)
cost_buy = torch.tensor([0.25], requires_grad=True)
dem_tot = torch.tensor([3500.], requires_grad=True)
inputs = torch.tensor([1000./lifetime, 200./lifetime, 0.25, 3500.], requires_grad=True)
base = torch.tensor([1000./100/lifetime, 200./100/lifetime, 0.25/100, 3500./100], requires_grad=True)

# solve the problem
result = plexplain.forward(inputs, False)
solution = plexplain.forward(inputs)
# print the problem solution
# print(result)
# print(solution[-2:])

# compute the gradients for the model
# jac = torch.autograd.functional.jacobian(plexplain, inputs)
# result.backward()
# attributions_grad = [i.grad for i in inp]
# print_nice(inp, result, attributions_grad)

# compute the gradient times input attributions for the model
# attributions_gxi = gxi.attribute([inputs], no_jacobian=True)
# print_nice(inputs, result, attributions_gxi)

# compute the IG attributions for the model (attributions for the optimal solutions require changes in the IG function)
# attributions_ig = ig.attribute(inputs, base, no_jacobian=True, steps=20)[0]
# print_nice(inputs, result, attributions_ig, base, is_ig=True, is_opt=True)

# compute the GCA
cap_pv = solution[6]
cap_bat = solution[5]
opex = torch.sum(solution[4]) * cost_buy
own_gen = torch.sum(solution[0]) / dem_tot
print(f"============= Cap PV =============\n{cap_pv}")
print(f"============= Cap bat =============\n{cap_bat}")
print(f"============= OPEX =============\n{opex}")
print(f"============= Own gen =============\n{own_gen}")

# computing the granger causal attributions
gca = np.ones([12, 5])
for m in range(12):
    reduced_model = PlexPlain(reduce_dimension=True, month=m+1)
    red_result, red_solution = reduced_model.forward(inputs, False), reduced_model.forward(inputs, True)

    gca[m][0] = result - red_result
    gca[m][1] = cap_pv - red_solution[6]
    gca[m][2] = cap_bat - red_solution[5]
    gca[m][3] = opex - torch.sum(red_solution[4]) * cost_buy
    gca[m][4] = own_gen - torch.sum(red_solution[0]) / dem_tot

np.set_printoptions(suppress=True)

print(gca.round(4))



