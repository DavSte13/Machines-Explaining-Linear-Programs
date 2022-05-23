import torch
import pandas as pd
from problems.knapsack import Knapsack
from attribution_methods.gradientxinput import GradientXInput
from attribution_methods.integrated_gradients import IntegratedGradients
from attribution_methods.occlusion import Occlusion
import xlp_utils


def evaluate_knapsack(evaluation_function, input_parameters, attribution_method, baselines, descriptions):
    """
    :param evaluation_function: String which signals how the optimal solution should be processed: 'opt' - remains as
      the optimal solution; 'cost' - apply the objective function to the optimal solution.
    :param input_parameters: List of: tuple of the parameters for the problem: (items, limit), items is a list of
      [value, weight] pairs and limit is an integer.
    :param attribution_method: List of: Which attribution method should be applied, one of: 'grad': gradients; '
      gxi': gradient times input; 'ig': integrated gradients; 'occ': Occlusion.
    :param baselines: List of: None or a valid baseline. For each entry in attribution methods which is 'ig', a baseline
      has to be given.
    :param descriptions: list of: strings which are added to the resulting dataframe entry as a description.
    """
    num_runs = len(input_parameters)

    # the problem specification based on CombOptNet does not require specific parameter shapes, therefore it is not
    # necessary to create a reduced problem for the granger causal attributions
    ks = Knapsack(evaluation_function)

    # prepare the attribution methods
    gxi = GradientXInput(ks)
    ig = IntegratedGradients(ks)
    occ = Occlusion(ks, ks, evaluation_function, reduce_rows=False)

    # prepare the result dataframe
    col_list = ['method', 'eval_func', 'input', 'result', 'attributions', 'baseline', 'description']
    attributions = pd.DataFrame(columns=col_list)

    # compute the attributions for the specified attribution method and all inputs
    for i in range(num_runs):
        # extract the problem parameters from the inputs
        items, limit = input_parameters[i]
        a = torch.tensor([items[:, 1]])
        b = torch.tensor([[limit]])
        c = torch.tensor(items[:, 0])

        inp = (a, b, c)

        result = ks.forward(*inp).detach().numpy().tolist()
        # create attributions and detach them
        if attribution_method[i] == 'grad':
            attr = xlp_utils.detach_tuple(torch.autograd.functional.jacobian(ks, inp), to_list=True)
            # convert to list to make data types unified
            attr = list(attr)
        elif attribution_method[i] == 'gxi':
            attr = xlp_utils.detach_tuple(gxi.attribute(inp), to_list=True)
            # convert to list to make data types unified
            attr = list(attr)
        elif attribution_method[i] == 'ig':
            # extract the problem baseline from the input
            base_items, base_limit = baselines[i]
            a_base = torch.tensor([base_items[:, 1]])
            b_base = torch.tensor([[base_limit]])
            c_base = torch.tensor(base_items[:, 0])

            base = (a_base, b_base, c_base)

            attr, _ = ig.attribute(inp, base, steps=100)
            # convert the numpy arrays to lists for storing later
            attr = xlp_utils.numpy_tuple_to_list(attr)
        elif attribution_method[i] == 'occ':
            attr = xlp_utils.detach_tuple(occ.attribute(inp), to_list=True)

        # add the results to the attributions dataframe and detach them
        inp = [input_parameters[i][0].tolist(), input_parameters[i][1]]
        if baselines[i] is not None:
            base = [baselines[i][0].tolist(), baselines[i][1]]
        else:
            base = None
        values = pd.DataFrame([[attribution_method[i], evaluation_function, inp, result, attr,
                                base, descriptions[i]]], columns=col_list)
        attributions = attributions.append(values, ignore_index=True)

    return attributions
