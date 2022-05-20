import torch
import pandas as pd
import numpy as np
from problems.resource_optimization import ResourceOpt
from attribution_methods.gradientxinput import GradientXInput
from attribution_methods.integrated_gradients import IntegratedGradients
from attribution_methods.occlusion import Occlusion
import xlp_utils


def evaluate_ro(evaluation_function, input_parameters, attribution_method, baselines, descriptions):
    """
    :param evaluation_function: String which signals how the optimal solution should be processed: 'opt' - remains as
        the optimal solution; 'cost' - apply the cost function to the optimal solution.
    :param input_parameters: list of: tuple of the parameters for the problem: (a, b, c) -
        all values for a, b and c are required to have the same shapes
    :param attribution_method: list of: Which attribution method should be applied, one of: 'grad': gradients; 'gxi': gradient
        times input; 'ig': integrated gradients; 'occ': occlusion
    :param baselines: list of: None or a valid baseline. For each entry in attribution methods which is 'ig', a baseline
        has to be given.
    :param descriptions: list of: strings which are added to the resulting dataframe entry as a description
    """
    num_runs = len(input_parameters)

    m, n = input_parameters[0][0].shape
    ro = ResourceOpt(n, evaluation_function, reduce_dimension=False, m=m)
    ro_reduced = ResourceOpt(n, evaluation_function, reduce_dimension=True, m=m)

    # prepare the attribution methods
    gxi = GradientXInput(ro)
    ig = IntegratedGradients(ro)
    occ = Occlusion(ro, ro_reduced, evaluation_function, reduce_rows=True)

    # prepare the result dataframe
    col_list = ['method', 'eval_func', 'input', 'result', 'attributions', 'baseline', 'description']
    attributions = pd.DataFrame(columns=col_list)

    # compute the attributions for the specified attribution method and all inputs
    for i in range(num_runs):

        # compute the result of the LP
        result = np.around(ro.forward(*input_parameters[i]).detach().numpy(), 4)
        result = result.tolist()

        # create attributions and detach them
        if attribution_method[i] == 'grad':
            attr = xlp_utils.detach_tuple(torch.autograd.functional.jacobian(ro, input_parameters[i]), to_list=True)
            # convert to list to make data types unified
            attr = list(attr)
        elif attribution_method[i] == 'gxi':
            attr = xlp_utils.detach_tuple(gxi.attribute(input_parameters[i]), to_list=True)
            # convert to list to make data types unified and convert the entries to lists for storing later
            attr = list(attr)
        elif attribution_method[i] == 'ig':
            attr, _ = ig.attribute(input_parameters[i], baselines[i], steps=100)
            # convert the numpy arrays to lists for storing later
            attr = xlp_utils.numpy_tuple_to_list(attr)
        elif attribution_method[i] == 'occ':
            attr = xlp_utils.detach_tuple(occ.attribute(input_parameters[i]), to_list=True)

        # add the results to the attributions dataframe and detach them
        inp = xlp_utils.detach_tuple(input_parameters[i], to_list=True)
        base = xlp_utils.detach_tuple(baselines[i], to_list=True)
        values = pd.DataFrame([[attribution_method[i], evaluation_function, inp, result, attr,
                                base, descriptions[i]]], columns=col_list)
        attributions = attributions.append(values, ignore_index=True)

    return attributions
