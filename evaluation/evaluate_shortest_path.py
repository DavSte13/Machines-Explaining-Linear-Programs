import torch
import pandas as pd
from problems.shortest_path import ShortestPath
from attribution_methods.gradientxinput import GradientXInput
from attribution_methods.integrated_gradients import IntegratedGradients
from attribution_methods.occlusion import Occlusion
import xlp_utils


def evaluate_sp(evaluation_function, input_parameters, attribution_method, baselines, descriptions):
    """
    :param evaluation_function: String which signals how the optimal solution should be processed: 'opt' - remains as
      the optimal solution; 'cost' - apply the objective function to the optimal solution.
    :param input_parameters: List of: tuple of the parameters for the problem: (a, b, c) -
      all values for a, b and c are required to have the same shapes.
    :param attribution_method: List of: Which attribution method should be applied, one of: 'grad': gradients;
      'gxi': gradient times input; 'ig': integrated gradients; 'occ': occlusion.
    :param baselines: List of: None or a valid baseline. For each entry in attribution methods which is 'ig', a baseline
      has to be given.
    :param descriptions: List of: strings which are added to the resulting dataframe entry as a description.
    """
    num_runs = len(input_parameters)

    # the problem specification based on CombOptNet does not require specific parameter shapes, therefore it is not
    # necessary to create a reduced problem for the granger causal attributions
    sp = ShortestPath(evaluation_function)

    # prepare the attribution methods
    gxi = GradientXInput(sp)
    ig = IntegratedGradients(sp)
    occ = Occlusion(sp, sp, evaluation_function, reduce_rows=False)

    # prepare the result dataframe
    col_list = ['method', 'eval_func', 'input', 'result', 'attributions', 'baseline', 'description']
    attributions = pd.DataFrame(columns=col_list)

    # compute the attributions for the specified attribution method and all inputs
    for i in range(num_runs):
        # get input parameters from graph
        a, b, c = xlp_utils.convert_graph(*input_parameters[i], 'sp', plot=False)
        a_tch = torch.tensor(a, requires_grad=True)
        b_tch = torch.tensor(b, requires_grad=True).view(1, -1).t()
        c_tch = torch.tensor(c, requires_grad=True)
        inp = (a_tch, b_tch, c_tch)

        result = sp.forward(*inp).detach().numpy().tolist()
        # create attributions and detach them
        if attribution_method[i] == 'grad':
            attr = xlp_utils.detach_tuple(torch.autograd.functional.jacobian(sp, inp), to_list=True)
            # convert to list to make data types unified
            attr = list(attr)
        elif attribution_method[i] == 'gxi':
            attr = xlp_utils.detach_tuple(gxi.attribute(inp), to_list=True)
            # convert to list to make data types unified
            attr = list(attr)
        elif attribution_method[i] == 'ig':
            # get the baseline parameters from the baseline graph
            base_a, base_b, base_c = xlp_utils.convert_graph(*baselines[i], 'sp', plot=False)
            base = (torch.Tensor(base_a), torch.Tensor(base_b).view(1, -1).t(), torch.Tensor(base_c))

            attr, _ = ig.attribute(inp, base, steps=100)
            # convert the numpy arrays to lists for storing later
            attr = xlp_utils.numpy_tuple_to_list(attr)
        elif attribution_method[i] == 'occ':
            attr = xlp_utils.detach_tuple(occ.attribute(inp), to_list=True)

        # add the results to the attributions dataframe and detach them
        inp = xlp_utils.numpy_tuple_to_list(input_parameters[i])
        base = xlp_utils.numpy_tuple_to_list(baselines[i])
        values = pd.DataFrame([[attribution_method[i], evaluation_function, inp, result, attr,
                                base, descriptions[i]]], columns=col_list)
        attributions = attributions.append(values, ignore_index=True)

    return attributions
